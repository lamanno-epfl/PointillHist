import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.nn import SimpleConv

from ._models import PointillHistNet

__all__ = [
    "networks",
    "setup_training",
    "load_model",
    "calibrate_distance_scalers",
    "estimate_initial_dispersions",
    "estimate_initial_dispersions_normal",
    "estimate_initial_dispersions_lognormal",
]

DIST_SCALER_EDGE_TYPES = {
    "cells__is_close_to__cells": ("cells", "is_close_to", "cells"),
    "cells__is_watched_by__longrange_grid": ("cells", "is_watched_by", "longrange_grid"),
    "longrange_grid__is_close_to__longrange_grid": (
        "longrange_grid", "is_close_to", "longrange_grid",
    ),
}


def networks(graphs, **kwargs):
    """
    Build a :class:`PointillHistNet` whose sizes are read from ``graphs``.

    ``n_classes``, ``n_genes``, ``n_timepoints``, ``n_sections`` and
    ``n_conditions`` come from the graph attributes set by ``generate_graphs``;
    ``**kwargs`` are the architecture options of :class:`PointillHistNet`.
    """
    if len(graphs) == 0:
        raise ValueError("networks() needs at least one graph.")
    missing = [name for name in ("cell_types", "genes", "timepoint", "section", "condition")
               if not all(hasattr(g, name) for g in graphs)]
    if missing:
        raise ValueError(
            f"graphs lack the attribute(s) {missing}; build them with ph.pp.generate_graphs."
        )
    return PointillHistNet(
        n_classes=len(graphs[0].cell_types),
        n_genes=len(graphs[0].genes),
        n_timepoints=max(int(g.timepoint) for g in graphs) + 1,
        n_sections=max(int(g.section) for g in graphs) + 1,
        n_conditions=max(int(g.condition) for g in graphs) + 1,
        **kwargs,
    )


@torch.no_grad()
def calibrate_distance_scalers(net, graphs, factor=2.0, device=None):
    """
    Set ``net.dist_scalers`` from the edge lengths observed in ``graphs``.

    For each spatial edge type the per-graph median edge length is measured and
    the scaler is set to ``factor`` times the median of those medians. The
    reverse grid->cell edge shares the scale of the forward one.
    """
    if device is None:
        device = next(net.parameters()).device

    per_key_graph_medians = {k: [] for k in DIST_SCALER_EDGE_TYPES}

    for graph in graphs:
        for key, (src_type, rel, dst_type) in DIST_SCALER_EDGE_TYPES.items():
            edge_index = graph[(src_type, rel, dst_type)].edge_index.to(device)
            src_pos = graph[src_type]["pos"].to(device)
            dst_pos = graph[dst_type]["pos"].to(device)
            lengths = (src_pos[edge_index[0]] - dst_pos[edge_index[1]]).norm(dim=1)
            per_key_graph_medians[key].append(lengths.median().item())

    new_scalers = {}
    for key, medians in per_key_graph_medians.items():
        base = torch.tensor(medians, device=device).median().item()
        new_scalers[key] = float(factor * base)

    for key, value in new_scalers.items():
        if key not in net.dist_scalers:
            raise KeyError(
                f"{key} not found in net.dist_scalers. Available: {list(net.dist_scalers.keys())}"
            )
        net.dist_scalers[key].data = torch.tensor(
            value, device=device, dtype=net.dist_scalers[key].dtype
        )

    reverse_key = "longrange_grid__rev_watched_by__cells"
    forward_key = "cells__is_watched_by__longrange_grid"
    if forward_key in new_scalers and reverse_key in net.dist_scalers:
        value = new_scalers[forward_key]
        net.dist_scalers[reverse_key].data = torch.tensor(
            value, device=device, dtype=net.dist_scalers[reverse_key].dtype
        )
        new_scalers[reverse_key] = value

    return new_scalers


def _aggregated_counts(graphs):
    """Stack the per-cell expression of every graph (on CPU)."""
    expressions = []
    for graph in graphs:
        graph = graph.cpu()
        try:
            counts = graph["cells"].x
        except AttributeError:
            counts = SimpleConv(aggr='sum', flow="source_to_target")(
                (graph["dots"].x, graph["cells"].pos),
                graph["dots", "could_come_from", "cells"].edge_index,
            )
        expressions.append(counts)
    return torch.cat(expressions, dim=0)


def estimate_initial_dispersions(graphs: list, device: torch.device) -> torch.Tensor:
    """
    Method-of-moments dispersion per gene over all graphs:
    ``mean**2 / (var - mean)`` where ``var > mean``, else ``mean``.
    Cells with total expression <= 1 are ignored.
    """
    expressions = _aggregated_counts(graphs)
    expressions = expressions[expressions.sum(dim=1) > 1]

    gene_means = expressions.mean(dim=0)
    gene_vars = expressions.var(dim=0)
    eps = 1e-8

    dispersion_estimates = torch.where(
        gene_vars > gene_means,
        gene_means**2 / (gene_vars - gene_means + eps),
        gene_means,
    )
    return dispersion_estimates.to(device)


def estimate_initial_dispersions_normal(
    graphs: list,
    device: torch.device,
    min_sigma: float = 0.05,
) -> torch.Tensor:
    """Per-gene standard deviation for the Normal likelihood, floored at ``min_sigma``."""
    expressions = _aggregated_counts(graphs)
    gene_vars = expressions.var(dim=0, unbiased=False)  # MLE variance
    sigma = torch.sqrt(gene_vars.clamp_min(0.0)).clamp_min(min_sigma)
    return sigma.to(device)


def estimate_initial_dispersions_lognormal(
    graphs: list,
    device: torch.device,
    eps: float = 1e-3,
    min_sigma: float = 0.05,
) -> torch.Tensor:
    """
    Per-gene standard deviation of ``log(x + eps)`` for the log-normal
    likelihood, floored at ``min_sigma``. Cells with zero total expression are ignored.
    """
    expressions = _aggregated_counts(graphs)
    expressions = expressions[expressions.sum(dim=1) > 0]
    y = torch.log(expressions.clamp_min(eps))
    gene_vars = y.var(dim=0, unbiased=False)
    sigma = torch.sqrt(gene_vars.clamp_min(0.0)).clamp_min(min_sigma)
    return sigma.to(device)


def setup_training(
    net: nn.Module,
    graphs: list,
    device: torch.device = torch.device("cpu"),
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    T_0: float = 50,
    eta_min: float = 3e-4,
    cell_loss_type: str = 'zip',
):
    """
    Initialise ``net.dispersion`` and ``net.dist_scalers`` from ``graphs`` and
    return ``(optimizer, scheduler)``: AdamW and CosineAnnealingWarmRestarts.
    """
    if cell_loss_type in ('nb', 'zinb', 'poisson', 'zip'):
        init_disp = estimate_initial_dispersions(graphs, device)
    elif cell_loss_type == 'normal':
        init_disp = estimate_initial_dispersions_normal(graphs, device)
    elif cell_loss_type == 'log-normal':
        init_disp = estimate_initial_dispersions_lognormal(graphs, device)
    else:
        raise ValueError(
            f"Unknown cell loss type: {cell_loss_type}. Possible values are: "
            "['zip', 'zinb', 'nb', 'poisson', 'normal', 'log-normal']."
        )
    net.dispersion = nn.Parameter(init_disp.to(device))

    calibrate_distance_scalers(net, graphs, factor=2.0)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=eta_min)

    return optimizer, scheduler


def load_model(model_path: str, graphs, device=None, **kwargs) -> nn.Module:
    """
    Rebuild the network for ``graphs`` (see :func:`networks`) and load the
    state dict saved at ``model_path``. ``device=None`` -> cuda if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = networks(graphs, **kwargs)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    return net
