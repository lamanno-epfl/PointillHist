import copy
import os

import torch
import torch.distributed as dist
import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

from ..preprocessing._store import DiskGraphs
from ._losses import total_loss
from ._setup import _initialize_from_disk, _optimizer, setup_training
from ._train import (
    LOSS_KEYS,
    _lambda_anatomical_schedule,
    _lambda_density_schedule,
    _reference,
    _resident_bytes,
    _setup_subset,
    _to_device,
    _type_priors,
    _type_regions,
    train,
)

__all__ = ["train_distributed"]


def _average_gradients(params, world_size):
    """Average the gradients over the ranks in one all-reduce; a rank without a gradient contributes zeros,
    and a parameter without a gradient on any rank keeps ``grad=None``."""
    present = torch.tensor([p.grad is not None for p in params], dtype=params[0].dtype, device=params[0].device)
    flat = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in params] + [present])
    dist.all_reduce(flat)
    flat /= world_size
    chunks = flat.split([p.numel() for p in params] + [len(params)])
    for p, grad, count in zip(params, chunks, chunks[-1].tolist()):
        p.grad = grad.view_as(p) if count > 0 else None


def train_distributed(
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
    setup_graphs="auto",
):
    """
    :func:`train` on several processes, one per GPU: ``torchrun --nproc_per_node=<n_gpus> script.py``.

    Takes the same parameters as :func:`train`. ``batch_size`` is per process, so each step averages
    ``n_processes * batch_size`` graphs. Every process must hold the same ``graphs`` (built identically or
    loaded from one saved file). ``device=None`` (or ``"cuda"``) is the GPU ``LOCAL_RANK``. A missing process
    group is created (nccl on GPU, gloo on CPU); end the script with ``torch.distributed.destroy_process_group()``.
    Without torchrun (no process group, ``WORLD_SIZE`` unset or 1) this is exactly :func:`train`.

    Setup: with a list, process 0 estimates the dispersions and distance scalers alone while the others
    wait (a very long setup can exceed the NCCL timeout, see ``setup_graphs``). With a DiskGraphs (one
    folder read by every process) each process reads its share of the setup graphs and the results are
    combined, so no process waits for the others.

    Returns
    -------
    net, history
        On every process; ``history`` averages all the graphs of all the processes. Only rank 0
        shows progress and writes checkpoints; run ``ph.eval.predict`` on rank 0.
    """
    if not (dist.is_available() and dist.is_initialized()) and int(os.environ.get("WORLD_SIZE", 1)) <= 1:
        return train(**locals())

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda":
        if device.index is None:   # modulo: a launcher may show each process only its own GPU
            local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
            device = torch.device("cuda", local_rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    main = rank == 0

    reference = _reference(reference, graphs).to(device)
    priors = _type_priors(type_priors, graphs)
    type_regions = _type_regions(type_regions, graphs)
    if type_regions is not None:
        type_regions = type_regions.to(device)

    net = net.to(device)
    net.eval()
    if main:
        print(f"Initializing (dispersions, distance scalers, optimizer) for {world_size} processes...")
    setup_set = _setup_subset(graphs, setup_graphs)
    if isinstance(graphs, DiskGraphs):
        # every rank reads its share of the setup graphs; the shares are combined
        _initialize_from_disk(net, setup_set, device, cell_loss_type, distributed=True)
        optimizer, scheduler = _optimizer(net, lr, weight_decay, T_0, eta_min)
    else:
        # only rank 0 estimates from the setup graphs; all ranks then start from its parameters and buffers
        optimizer, scheduler = setup_training(
            net, setup_set if main else graphs[:1], device=device,
            lr=lr, weight_decay=weight_decay, T_0=T_0, eta_min=eta_min,
            cell_loss_type=cell_loss_type,
        )
    net.temperature.fill_(temperature)
    for tensor in [*net.parameters(), *net.buffers()]:
        dist.broadcast(tensor.data, src=0)

    if lambda_density_min is None:
        lambda_density_min = lambda_density / 10
    if lambda_density_warmup is None:
        lambda_density_warmup = num_epochs // 2

    history = {k: [] for k in LOSS_KEYS}
    history['lambda_density_eff'] = []
    history['lambda_anatomical_eff'] = []

    # each rank gets an equal share of every shuffled epoch, so all ranks take the same number of steps
    sampler = DistributedSampler(range(len(graphs)), shuffle=True)
    data_loader = DataLoader(range(len(graphs)), batch_size=batch_size, sampler=sampler)
    # the losses use parameters outside forward(), which DistributedDataParallel does not track,
    # so the gradients are averaged by hand after each backward
    params = [p for p in net.parameters() if p.requires_grad]

    net.train()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # graph placement as in train(), decided by each rank for its own device
    if device.type != 'cuda':
        keep = False
    elif keep_on_device == "auto":
        keep = None
    else:
        keep = bool(keep_on_device)
        if keep:
            graphs = _to_device(graphs, device)

    bar = tqdm.tqdm(total=num_epochs * len(data_loader), desc="Training", disable=not main)

    for epoch in range(1, num_epochs + 1):
        sampler.set_epoch(epoch)
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

            avg_batch_loss = batch_loss / len(batch_indices)
            avg_batch_loss.backward()
            _average_gradients(params, world_size)
            clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            del avg_batch_loss, batch_loss

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
            peak = torch.cuda.max_memory_allocated(device)
            resident = _resident_bytes(graphs)
            capacity = torch.cuda.mem_get_info(device)[1]
            keep = resident + peak <= 0.95 * capacity
            if keep:
                graphs = _to_device(graphs, device)
                if main:
                    bar.write(f"graphs kept on {device}: {resident / 1e9:.1f} GB of graphs + {peak / 1e9:.1f} GB peak "
                              f"per step fit in 95 % of {capacity / 1e9:.0f} GB")
            elif main:
                bar.write(f"graphs stay on the host and are copied to {device} for each step: {resident / 1e9:.1f} GB of "
                          f"graphs + {peak / 1e9:.1f} GB peak per step exceed 95 % of {capacity / 1e9:.0f} GB")

        # mean over every graph processed by every rank
        totals = torch.tensor([epoch_sums[k] for k in LOSS_KEYS] + [graphs_processed],
                              dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        *sums, n = totals.tolist()
        for k, total in zip(LOSS_KEYS, sums):
            history[k].append(total / n)
        history['lambda_density_eff'].append(lambda_density_eff)
        history['lambda_anatomical_eff'].append(lambda_anatomy)

        if main and checkpoint_dir and epoch % 20 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"net_checkpoint_epoch_{epoch}.pth")
            torch.save(net.state_dict(), path)

    bar.close()
    return net, history
