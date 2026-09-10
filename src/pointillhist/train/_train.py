import copy
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..preprocessing._reference import load_reference
from ._setup import setup_training
from ._losses import total_loss

__all__ = ["train"]

#: Order must match the tuple returned by :func:`total_loss`.
LOSS_KEYS = [
    "total_loss",
    "grid_loss",
    "grid_cosine_loss",
    "density_loss",
    "cell_loss",
    "anatomical_loss",
]


def _lambda_anatomical_schedule(epoch, num_epochs, lambda_anatomical, lambda_anatomical_init,
                                lambda_anatomical_warmup, lambda_anatomical_expand):
    """Compute effective lambda_anatomical for the current epoch.

    Schedules
    ---------
    'constant'  : lambda_anatomical throughout.
    'cosine'    : lambda_anatomical_init for warmup epochs, then cosine increase
                  from lambda_anatomical_init to lambda_anatomical over
                  the remaining epochs.
    'linear'    : lambda_anatomical_init for warmup epochs, then linear increase
                  from lambda_anatomical_init to lambda_anatomical over
                  the remaining epochs.
    """
    if lambda_anatomical_expand == 'constant':
        return lambda_anatomical
    if epoch <= lambda_anatomical_warmup:
        return lambda_anatomical_init
    t = epoch - lambda_anatomical_warmup
    T = max(1, num_epochs - lambda_anatomical_warmup)
    frac = min(t / T, 1.0)
    if lambda_anatomical_expand == 'cosine':
        scale = 0.5 * (1.0 - np.cos(np.pi * frac))
    elif lambda_anatomical_expand == 'linear':
        scale = frac
    else:
        raise ValueError(f"Unknown lambda_anatomical_expand schedule: {lambda_anatomical_expand}")
    return lambda_anatomical_init + (lambda_anatomical - lambda_anatomical_init) * scale


def _lambda_density_schedule(epoch, num_epochs, lambda_density, lambda_density_min, lambda_density_warmup, lambda_density_decay):
    """Compute effective lambda_density for the current epoch.

    Schedules
    ---------
    'constant'  : lambda_density throughout.
    'cosine'    : constant for warmup epochs, then cosine decay to lambda_density_min.
    'linear'    : constant for warmup epochs, then linear decay to lambda_density_min.
    """
    if lambda_density_decay == 'constant':
        return lambda_density
    t = max(0, epoch - lambda_density_warmup)
    T = max(1, num_epochs - lambda_density_warmup)
    if t <= 0:
        return lambda_density
    frac = min(t / T, 1.0)
    if lambda_density_decay == 'cosine':
        scale = 0.5 * (1.0 + np.cos(np.pi * frac))
    elif lambda_density_decay == 'linear':
        scale = 1.0 - frac
    else:
        raise ValueError(f"Unknown lambda_density_decay schedule: {lambda_density_decay}")
    return lambda_density_min + (lambda_density - lambda_density_min) * scale


def _graph_bytes(graph):
    """Bytes of every tensor stored in a HeteroData (node, edge and graph-level attributes)."""
    return sum(v.numel() * v.element_size() for store in graph.stores for v in store.values() if torch.is_tensor(v))


def _reference(reference, graphs):
    """(K, G) float tensor aligned to the graphs' cell types and genes."""
    if isinstance(reference, (torch.Tensor, np.ndarray)):
        return torch.as_tensor(reference, dtype=torch.float32)
    table = load_reference(reference)
    return torch.tensor(table.loc[graphs[0].cell_types, graphs[0].genes].values, dtype=torch.float32)


def _type_priors(type_priors, graphs):
    """One (K,) prior per timepoint code, indexed by code, each summing to 1."""
    if type_priors is None:
        raise ValueError(
            'type_priors must be "uniform", a DataFrame/csv path (cell types x timepoint '
            "labels) or a (K,) array/Series/tensor, not None"
        )
    cell_types = list(graphs[0].cell_types)
    K = len(cell_types)
    labels = {g.timepoint: g.timepoint_label for g in graphs}
    n_timepoints = max(labels) + 1

    if isinstance(type_priors, str) and type_priors == "uniform":
        return [torch.ones(K) / K] * n_timepoints
    if isinstance(type_priors, str):
        type_priors = pd.read_csv(type_priors, index_col=0)
    if isinstance(type_priors, pd.Series):
        type_priors = type_priors.to_frame()

    if isinstance(type_priors, pd.DataFrame):
        # cell_types are str (load_reference casts the index), so cast the table's labels too
        table = type_priors.rename(index=str, columns=str).loc[cell_types]
        if table.shape[1] == 1:
            columns = {code: table.columns[0] for code in labels}
        else:
            columns = {code: str(label) for code, label in labels.items()}
            missing = sorted(set(columns.values()) - set(table.columns))
            if missing:
                raise ValueError(
                    f"type_priors has no column for timepoint(s) {missing}; "
                    f"available columns: {list(table.columns)}"
                )
        vectors = {code: torch.tensor(table[col].values, dtype=torch.float32)
                   for code, col in columns.items()}
    else:
        vector = torch.as_tensor(type_priors, dtype=torch.float32)
        vectors = {code: vector for code in labels}

    priors = [None] * n_timepoints
    for code, v in vectors.items():
        if v.shape != (K,):
            raise ValueError(f"type_priors must have one entry per cell type ({K}), got shape {tuple(v.shape)}")
        if not torch.isfinite(v).all() or (v < 0).any():
            raise ValueError("type_priors must be finite and non-negative")
        if v.sum() <= 0:
            raise ValueError(f"type_priors for timepoint code {code} sums to zero")
        priors[code] = v / v.sum()
    return priors


def _type_regions(type_regions, graphs):
    """(K, n_regions) float tensor aligned to the graphs' cell types and regions, or None."""
    if type_regions is None:
        return None
    regions = getattr(graphs[0], "regions", None)
    if regions is None:
        raise ValueError("graphs carry no cell regions; pass region_key= to generate_graphs")
    if isinstance(type_regions, str):
        type_regions = pd.read_csv(type_regions, index_col=0)
    if isinstance(type_regions, pd.DataFrame):
        # cell_types and regions are str, so cast the table's labels too
        type_regions = type_regions.rename(index=str, columns=str).loc[list(graphs[0].cell_types), regions].values
    type_regions = torch.as_tensor(type_regions, dtype=torch.float32)
    expected = (len(graphs[0].cell_types), len(regions))
    if tuple(type_regions.shape) != expected:
        raise ValueError(
            f"type_regions must have shape (n_cell_types, n_regions) = {expected}, "
            f"got {tuple(type_regions.shape)}"
        )
    return type_regions


def train(
    net,
    graphs,
    reference,
    type_priors="uniform",
    type_regions=None,
    cell_loss_type='zip',
    temperature=1.0,
    batch_size=1,
    num_epochs=100,
    lr=1e-3,
    weight_decay=1e-7,
    T_0=50,
    eta_min=3e-4,
    lambda_cell_to_grid=10.0,
    lambda_grid_rows=1.0,
    lambda_grid_genes=1.0,
    lambda_density=100.0,
    lambda_density_min=None,
    lambda_density_warmup=None,
    lambda_density_decay='cosine',
    prior_uniform_mix=0.95,
    lambda_anatomical=0.1,
    lambda_anatomical_init=0.0,
    lambda_anatomical_warmup=0,
    lambda_anatomical_expand='cosine',
    checkpoint_dir=None,
    keep_on_device="auto",
    device=None,
):
    """
    Train the network, reporting a running average of each loss term per epoch.

    Parameters
    ----------
    graphs : list of HeteroData
        Output of ``generate_graphs``; sizes, timepoints and regions are read from it.
    reference : DataFrame | csv path | (K, G) tensor/array
        Cell types x genes expression profiles, aligned by name to the graphs.
    type_priors : "uniform" | DataFrame/csv (cell types x timepoint labels) | (K,) array/Series/tensor
        Target cell-type composition per timepoint; columns are normalised to sum 1.
    type_regions : None | DataFrame/csv (cell types x regions) | (K, n_regions) tensor/array
        1 where a cell type is expected in a region; enables the anatomical loss.
    lambda_density_min, lambda_density_warmup : number or None
        Density-prior schedule: ``lambda_density`` for the warm-up epochs, then
        ``lambda_density_decay`` towards ``lambda_density_min``. The defaults are
        a tenth of ``lambda_density`` and half of ``num_epochs``.
    checkpoint_dir : str or None
        If given, the state dict is saved there every 20 epochs.
        A DataFrame/csv is aligned by name; a tensor/array must have its columns
        in the order of ``graphs[0].regions`` (sorted region labels).
    keep_on_device : "auto" | bool
        Where the graphs live during training. True: all of them are moved to
        ``device`` once, before the first epoch, so no graph crosses the
        host-device bus again (fastest; needs device memory for every graph plus
        the working set of one step). False: the graphs stay on the host and a
        shallow copy of each one is sent to the device for its step (one
        host-to-device copy per step, nothing copied back, only one graph on
        the device at a time). "auto" (default): the first epoch runs in the
        copying mode while the peak memory of a step is measured; from the
        second epoch on the graphs stay resident if all of them plus that peak
        fit in 95 % of the device memory, otherwise a note is printed and the
        copying mode continues. Irrelevant on a CPU device.

    Returns
    -------
    net, history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    reference = _reference(reference, graphs).to(device)
    priors = _type_priors(type_priors, graphs)
    type_regions = _type_regions(type_regions, graphs)
    if type_regions is not None:
        type_regions = type_regions.to(device)

    net = net.to(device)
    net.eval()
    print("Initializing (dispersions, distance scalers, optimizer)...")
    optimizer, scheduler = setup_training(
        net, graphs, device=device,
        lr=lr, weight_decay=weight_decay, T_0=T_0, eta_min=eta_min,
        cell_loss_type=cell_loss_type,
    )
    net.temperature.fill_(temperature)

    # default schedule: constant for the first half of the epochs, then cosine decay to a tenth
    if lambda_density_min is None:
        lambda_density_min = lambda_density / 10
    if lambda_density_warmup is None:
        lambda_density_warmup = num_epochs // 2

    history = {k: [] for k in LOSS_KEYS}
    history['lambda_density_eff'] = []
    history['lambda_anatomical_eff'] = []

    data_loader = DataLoader(range(len(graphs)), batch_size=batch_size, shuffle=True)

    net.train()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # where the graphs live: True = resident on the device, False = a copy sent to the device per step,
    # None = "auto" (decided after the first epoch from the measured peak memory of a step)
    if device.type != 'cuda':
        keep = False
    elif keep_on_device == "auto":
        keep = None
    else:
        keep = bool(keep_on_device)
        if keep:
            for graph in graphs:
                graph.to(device)

    bar = tqdm.tqdm(total=num_epochs * len(data_loader), desc="Training")

    for epoch in range(1, num_epochs + 1):
        epoch_sums = dict.fromkeys(LOSS_KEYS, 0.0)
        graphs_processed = 0
        lambda_anatomy = _lambda_anatomical_schedule(
            epoch, num_epochs, lambda_anatomical, lambda_anatomical_init,
            lambda_anatomical_warmup, lambda_anatomical_expand)
        lambda_density_eff = _lambda_density_schedule(
            epoch, num_epochs, lambda_density, lambda_density_min, lambda_density_warmup, lambda_density_decay)

        for batch_indices in data_loader:
            optimizer.zero_grad()
            batch_loss = 0.0

            for i in batch_indices:
                # a resident graph is used in place; otherwise a shallow copy carries this step's tensors to the
                # device while the host copy stays untouched (nothing is copied back, the device copy is freed
                # once the step's autograd graph is released)
                graph = graphs[i] if keep else copy.copy(graphs[i]).to(device)

                cell_dots, logits, *_, scale, (pi_cell, pi_lowrank) = net(graph)

                losses = total_loss(
                    graph=graph,
                    lambda_cell_to_grid=lambda_cell_to_grid,
                    cell_regions=getattr(graph, "cell_regions", None),
                    type_regions=type_regions,
                    counts=cell_dots,
                    logits=logits,
                    scale=scale,
                    reference=reference,
                    cell_loss_type=cell_loss_type,
                    temperature=temperature,
                    type_prior=priors[graph.timepoint],
                    lambda_anatomical=lambda_anatomy,
                    dispersion=net.dispersion,
                    lambda_grid_rows=lambda_grid_rows,
                    lambda_grid_genes=lambda_grid_genes,
                    lambda_density=lambda_density_eff,
                    prior_uniform_mix=prior_uniform_mix,
                    dropout_slope=net.pi_slope,
                    dropout_bias=net.pi_bias,
                    pi_cell=pi_cell, pi_lowrank=pi_lowrank,
                    ambient_logit=net.ambient_logit,
                    ambient_profile_logits=net.ambient_profile_logits,
                    tau_pi=net.pi_logit_temp,
                    gene_detection_bias=net.gene_detection_bias,
                    gene_detection_offset=net.gene_detection_offset,
                )
                batch_loss += losses[0]
                for k, val in zip(LOSS_KEYS, losses):
                    epoch_sums[k] += val.detach().cpu().item()

                del pi_cell, pi_lowrank, cell_dots, logits, scale, graph, losses

            if len(batch_indices) > 0:
                avg_batch_loss = batch_loss / len(batch_indices)
                avg_batch_loss.backward()
                clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                del avg_batch_loss
            del batch_loss

            graphs_processed += len(batch_indices)
            bar.update(1)
            bar.set_postfix(
                epoch=f"{epoch}/{num_epochs}",
                total=f"{epoch_sums['total_loss'] / graphs_processed:.4f}",
                cell=f"{epoch_sums['cell_loss'] / graphs_processed:.4f}",
                grid=f"{epoch_sums['grid_cosine_loss'] / graphs_processed:.4f}",
                density=f"{epoch_sums['density_loss'] / graphs_processed:.4f}",
                anatomical=f"{epoch_sums['anatomical_loss'] / graphs_processed:.4f}",
                lam_d=f"{lambda_density_eff:.1f}",
                lam_anat=f"{lambda_anatomy:.4f}",
            )

        scheduler.step()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if keep is None:      # "auto": the first epoch measured the working set of a step (one graph included)
            index = torch.cuda.current_device() if device.index is None else device.index
            peak = torch.cuda.max_memory_allocated(index)
            resident = sum(_graph_bytes(g) for g in graphs)
            capacity = torch.cuda.mem_get_info(index)[1]
            keep = resident + peak <= 0.95 * capacity
            if keep:
                for graph in graphs:
                    graph.to(device)
                bar.write(f"graphs kept on {device}: {resident / 1e9:.1f} GB of graphs + {peak / 1e9:.1f} GB peak per step "
                          f"fit in 95 % of {capacity / 1e9:.0f} GB")
            else:
                bar.write(f"graphs stay on the host and are copied to {device} for each step: {resident / 1e9:.1f} GB of "
                          f"graphs + {peak / 1e9:.1f} GB peak per step exceed 95 % of {capacity / 1e9:.0f} GB")

        n = len(graphs)
        for k in LOSS_KEYS:
            history[k].append(epoch_sums[k] / n)
        history['lambda_density_eff'].append(lambda_density_eff)
        history['lambda_anatomical_eff'].append(lambda_anatomy)

        if checkpoint_dir and epoch % 20 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"net_checkpoint_epoch_{epoch}.pth")
            torch.save(net.state_dict(), path)

    bar.close()
    return net, history
