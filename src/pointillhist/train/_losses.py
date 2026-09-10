import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv

GLOBAL_EPS = 1e-8

__all__ = [
    "anatomical_loss",
    "grid_composition_loss",
    "nb_cell_loss",
    "zinb_cell_loss",
    "zip_cell_loss",
    "poisson_cell_loss",
    "gaussian_cell_loss",
    "lognormal_cell_loss",
    "total_loss",
]


def anatomical_loss(logits, cell_regions, type_regions, temperature=1.0, gamma=0.25):
    """
    Penalise cell types predicted outside the regions where they are expected.

    Parameters
    ----------
    logits : (n_cells, n_classes)
        Raw cell-type scores.
    cell_regions : (n_cells, n_regions)
        One-hot annotated anatomical region of each cell. Rows that are all
        zero mark unannotated cells and are skipped.
    type_regions : (n_classes, n_regions)
        1 where a cell type is expected to occur in that region, 0 otherwise.
    temperature : float
        Softmax temperature applied to ``logits``.
    gamma : float
        Weight of the specificity term. A type expected in many regions is a
        weaker piece of evidence than one expected in a single region, so
        compatible-but-broad types still pay ``gamma * (1 - 1/n_regions_k)``.
        ``gamma=0`` is a pure binary constraint, ``gamma=1`` full specificity
        weighting; 0.25 is balanced.
    """
    probs = F.softmax(logits / temperature, dim=1)                    # (n_cells, n_classes)
    compatible = (cell_regions @ type_regions.T > 0).float()          # (n_cells, n_classes)

    annotated = cell_regions.sum(dim=1) > 0
    if annotated.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    # Specificity penalty for compatible types: broad types pay more
    n_regions_k = type_regions.sum(dim=1, keepdim=True).T.clamp(min=1)      # (1, n_classes)
    specificity_penalty = compatible[annotated] * (1.0 - 1.0 / n_regions_k)  # (M, n_classes)

    # Full penalty for incompatible types, mild penalty for broad compatible ones
    weight = (1.0 - compatible[annotated]) + gamma * specificity_penalty

    return (probs[annotated] * weight).sum(dim=1).mean()

def grid_composition_loss(
    graph,
    counts,      # (C, G) cell counts
    logits,                    # (C, K) cell-type logits
    reference,             # (K, G) reference
    *,
    type_prior=None,           # (K,) target type density, or None
    # cosine weights
    lambda_row=1.0,            # row-wise (grid vectors)
    lambda_col=1.0,            # col-wise (gene profiles)
    # density regularizer
    lambda_density=100.0,            # weight of density prior
    prior_uniform_mix=0.99,           # convex mix toward uniform for stability
    # composition pipeline
    temperature=1.0,
    ambient_logit=None,        # scalar tensor; if None → ρ=0
    ambient_profile_logits=None,  # (G,); ignored if ρ=0
    rho_cap=0.2,
    # masks / numerics
    std_thr=0.01,
    sum_thr=50,
    epsilon=1e-8,
    # extras
    gene_detection_bias=None,
    gene_detection_offset=None,
):
    """
    Grid-level (hub) double-cosine + density prior, aligned with the mass-preserving
    composition used in cell losses.

    Predicted composition per grid:
        rate = (1-ρ) * (W @ S_norm) + ρ * b
      where W is grid cell-type mixture (row-normalized), S_norm is row-normalized
      (optionally tempered) reference, and b is ambient gene profile.
    """

    device = counts.device
    C, G = counts.shape

    # ---------- 1) Aggregate to grids ----------
    # Grid counts: sum cell counts to each gridpoint
    G_counts = SimpleConv(aggr="sum", flow="source_to_target")(
        (counts, torch.arange(graph["gridpoints"].num_nodes, device=device)[:, None]),
        graph["cells", "is_watched_by", "gridpoints"].edge_index,
    )  # (Ng, G)
    Ng = G_counts.size(0)
    # Row mask: grids that actually have data
    grid_lib = G_counts.sum(dim=1)                    # (Ng,)
    grid_mask = grid_lib > 0

    # Observed composition at grid level
    G_comp = G_counts / (grid_lib.clamp_min(1.0).unsqueeze(1))
    G_comp = G_comp / (G_comp.sum(dim=1, keepdim=True) + GLOBAL_EPS)  # renorm (safety)

    # ---------- 2) Grid cell-type mixture W ----------
    P = F.softmax(logits / temperature, dim=1)  # (C, K)

    grid_type_mass = SimpleConv(aggr="sum", flow="source_to_target")(
        (P, torch.arange(graph["gridpoints"].num_nodes, device=device)[:, None]),
        graph["cells", "is_watched_by", "gridpoints"].edge_index,
    )  # (Ng, K) — unnormalized "mass" of each type per grid

    W_den = grid_type_mass.sum(dim=1, keepdim=True)                     # (Ng,1)
    # Handle empty rows by falling back to uniform over types
    NgK = grid_type_mass.size(1)
    W = grid_type_mass / (W_den + GLOBAL_EPS)                              # (Ng,K)
    if (W_den <= epsilon).any():
        empty_rows = (W_den.squeeze(1) <= epsilon)
        W[empty_rows] = 1.0 / NgK

    # ---------- 3) Mass-preserving predicted composition at grid ----------
    # 1. Apply detection efficiency to the reference S BEFORE normalization
    detection_efficiency = torch.exp(gene_detection_bias).unsqueeze(0) # (1, G)
    S_adjusted = (reference * detection_efficiency + gene_detection_offset.unsqueeze(0)).relu()
    S_norm = S_adjusted / (S_adjusted.sum(dim=1, keepdim=True) + GLOBAL_EPS)  # (K,G), row-normalized
    base_rate = W @ S_norm                                              # (Ng, G) rows sum to 1

    if (ambient_logit is not None) and (ambient_profile_logits is not None):
        rho = torch.sigmoid(ambient_logit) * rho_cap                    # scalar in [0, rho_cap]
        ambient_raw = F.softplus(ambient_profile_logits) + 1e-8   # (G,) >= 0
        b = ambient_raw / ambient_raw.sum()                       # (G,), sums to 1
        rate = (1.0 - rho) * base_rate + rho * b.unsqueeze(0)           # (Ng, G) rows sum to 1
    else:
        rho = torch.tensor(0.0, device=device)
        rate = base_rate

    # ---------- 4) Gene mask (variance across grids + total grid counts) ----------
    gene_std = G_counts.std(dim=0)                                        # (G,)
    gene_sum = G_counts.sum(dim=0)                                      # (G,)
    gene_mask = (gene_std > std_thr) & (gene_sum > sum_thr)
    if not grid_mask.any() or not gene_mask.any():
        # Nothing to compare; return zero loss with grad
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, zero, torch.tensor(0.0, device=device)

    # ---------- 5) Double cosine (composition space) ----------
    # Row-wise: per-grid composition vectors
    pred_rows = F.normalize(rate[grid_mask], p=2, dim=1, eps=epsilon)
    obs_rows  = F.normalize(G_comp[grid_mask], p=2, dim=1, eps=epsilon)
    row_cos   = F.cosine_similarity(pred_rows, obs_rows, dim=1)         # (N_valid_grids,)
    row_term  = 1.0 - row_cos.mean()

    # Column-wise: per-gene profiles across valid grids
    pred_cols = F.normalize(rate[grid_mask][:, gene_mask], p=2, dim=0, eps=epsilon)
    obs_cols  = F.normalize(G_comp[grid_mask][:, gene_mask], p=2, dim=0, eps=epsilon)
    col_cos   = F.cosine_similarity(pred_cols, obs_cols, dim=0)         # (G_valid,)
    col_term  = 1.0 - col_cos.mean()

    corr_loss = lambda_row * row_term + lambda_col * col_term

    # ---------- 6) Density prior over cell types (reverse KL, mode-seeking) ----------
    if type_prior is not None:
        target = type_prior.to(device)                                     # (K,)
        target = target / (target.sum() + GLOBAL_EPS)
        # empirical global type usage from grids:
        emp = grid_type_mass[grid_mask].sum(dim=0)                       # (K,)
        emp = emp / (emp.sum() + GLOBAL_EPS)
        # mix target toward uniform for stability
        uniform = torch.full_like(emp, 1.0 / emp.numel())
        mixed_target = prior_uniform_mix * target + (1.0 - prior_uniform_mix) * uniform
        density_term = (emp * (torch.log(emp + GLOBAL_EPS) - torch.log(mixed_target + GLOBAL_EPS))).mean() 
    else:
        # fallback: encourage non-degenerate usage ≈ uniform
        emp = grid_type_mass[grid_mask].sum(dim=0)
        emp = emp / (emp.sum() + GLOBAL_EPS)
        uniform = torch.full_like(emp, 1.0 / emp.numel())
        density_term = (emp * (torch.log(emp + GLOBAL_EPS) - torch.log(uniform + GLOBAL_EPS))).sum()

    total_loss = corr_loss + lambda_density * density_term

    # ---------- Diagnostics (occasionally) ----------
    return total_loss, corr_loss, density_term

def nb_cell_loss(
    counts,            # (C, G) integer counts
    logits,                          # (C, K)
    scale,                           # (C,1) or (C,)
    reference,                   # (K, G)
    *,
    dispersion,                      # (G,)  θ (MoM) — used directly
    ambient_logit,                   # scalar (learnable)
    ambient_profile_logits,          # (G,)  (learnable)
    temperature=1.0,
    rho_cap=0.0,                     # cap for ambient fraction
    stabilize=True,
    epsilon=1e-8,
    weight_nnz=3.0,                  # extra weight on non-zeros (rare)
    scale_bound=0.5,                 # m in exp(tanh(scale)*m), allows ×[e^-m, e^m]
    gene_detection_bias=None,
    gene_detection_offset=None,
):
    # ---- checks ----
    assert counts.dim() == 2, "counts must be (C,G)"
    C, G = counts.shape
    assert logits.size(0) == C, "logits batch size mismatch"
    assert dispersion.shape == (G,), "dispersion must be (G,)"
    assert ambient_profile_logits.shape == (G,), "ambient_profile_logits must be (G,)"

    k = counts
    device = k.device

    # ---- composition: mass-preserving rate ----
    P = F.softmax(logits / temperature, dim=1)                                # (C,K)
    # 1. Apply detection efficiency to the reference S BEFORE normalization
    detection_efficiency = torch.exp(gene_detection_bias).unsqueeze(0) # (1, G)
    S_adjusted = (reference * detection_efficiency + gene_detection_offset).relu()
    S_norm = S_adjusted / (S_adjusted.sum(dim=1, keepdim=True) + GLOBAL_EPS)  # (K,G), row-normalized
    base_rate = P @ S_norm                                                    # (C,G), rows sum to 1

    rho = torch.sigmoid(ambient_logit) * rho_cap                              # scalar in [0, rho_cap]
    ambient_raw = F.softplus(ambient_profile_logits) + 1e-8   # (G,) >= 0
    b = ambient_raw / ambient_raw.sum()                       # (G,), sums to 1
    rate = (1.0 - rho) * base_rate + rho * b.unsqueeze(0)                     # (C,G), rows sum to 1

    # ---- totals: anchor μ to library size with bounded multiplier ----
    libsize = k.sum(dim=1).clamp_min(1.0)                                     # (C,)
    if scale.dim() == 2 and scale.size(1) == 1:
        scale = scale.squeeze(1)
    elif scale.dim() != 1:
        scale = scale.view(-1)
    assert scale.shape == (C,), "scale must be (C,) after squeeze/view"

    s = torch.tanh(scale) * scale_bound                                       # (C,)
    scale_mult = torch.exp(s)                                                 # (C,) in [e^-m, e^m]
    scale_eff = libsize * scale_mult                                          # (C,)
    mu = (scale_eff.unsqueeze(1) * rate).clamp_min(epsilon)                   # (C,G)

    # ---- dispersion θ (per-gene) from MoM ----
    theta = dispersion.unsqueeze(0).clamp_min(1e-3)                           # (1,G) -> (C,G)

    # ---- NB log pmf ----
    p = theta / (theta + mu)                                                  # (C,G)
    if stabilize:
        log_nb = (torch.lgamma(k + theta + GLOBAL_EPS)
                  - torch.lgamma(theta + GLOBAL_EPS)
                  - torch.lgamma(k + 1.0 + GLOBAL_EPS)
                  + theta * torch.log(p + GLOBAL_EPS)
                  + k * torch.log1p(-p + GLOBAL_EPS))
    else:
        log_nb = (torch.lgamma(k + theta)
                  - torch.lgamma(theta)
                  - torch.lgamma(k + 1.0)
                  + theta * torch.log(p + GLOBAL_EPS)
                  + k * torch.log1p(-p + GLOBAL_EPS))

    # Weight non-zeros (rare) higher if desired
    is_zero = (k == 0)
    loglik = torch.where(is_zero, log_nb, weight_nnz * log_nb)
    loss = -loglik.mean(dim=1).mean()

    # Optional: tiny anchor to keep μ totals near libsize (helps generalization)
    # ---- diagnostics (1% of batches) ----
    return loss

def zinb_cell_loss(
    counts,            # (C, G)
    logits,                          # (C, K)
    scale,                           # (C, 1) or (C,)
    reference,                   # (K, G)
    *,
    # --- ZI / dispersion params ---
    dispersion=None,                 # (G,) θ from MoM (use directly)
    pi_bias,                         # (G,) per-gene zero-inflation bias
    raw_pi_slope,                    # (G,) per-gene zero-inflation slope (softplus'd to <= 0)
    pi_cell=None,                        # (C, 1) cell scalar
    pi_lowrank=None,                          # (C, G) low-rank cell×gene term
    ambient_logit=None,              # scalar Parameter/Tensor
    ambient_profile_logits=None,     # (G,)
    # --- hparams ---
    temperature=1.0,
    rho_cap=0.2,                     # cap for ambient fraction
    scale_bound=0.5,                 # m in exp(tanh(scale)*m), allows ×[e^-m, e^m]
    tau_pi=2.0,                      # temperature on π logits
    stabilize=True,
    epsilon=1e-8,
    weight_nnz=3.0,
    gene_detection_bias=None,
    gene_detection_offset=None,
):
    # ---------- basic checks ----------
    assert counts.dim() == 2, "counts must be (C,G)"
    C, G = counts.shape
    assert logits.size(0) == C, "logits batch size mismatch"
    assert dispersion is not None and dispersion.shape == (G,), "dispersion (G,) required"
    assert pi_cell is not None and pi_cell.shape == (C,1), "pi_cell must be (C,1)"
    assert pi_lowrank is not None and pi_lowrank.shape == (C,G), "pi_lowrank must be (C,G)"
    assert ambient_logit is not None and ambient_profile_logits is not None, "ambient params required"
    assert ambient_profile_logits.shape == (G,), "ambient_profile_logits must be (G,)"

    device = counts.device
    k = counts

    # ---------- mixture over cell types ----------
    P = F.softmax(logits / temperature, dim=1)          # (C, K)
    
    # 1. Apply detection efficiency to the reference S BEFORE normalization
    detection_efficiency = torch.exp(gene_detection_bias).unsqueeze(0) # (1, G)
    S_adjusted = (reference * detection_efficiency + gene_detection_offset).relu()
    S_norm = S_adjusted / (S_adjusted.sum(dim=1, keepdim=True) + GLOBAL_EPS)  # (K,G), row-normalized
    base_rate = P @ S_norm                                                    # (C,G), rows sum to 1

    # Ambient, mass-preserving: rate = (1-ρ)*(P@S) + ρ*b, rows sum to 1
    rho = torch.sigmoid(ambient_logit) * rho_cap                              # scalar in [0, rho_cap]
    ambient_raw = F.softplus(ambient_profile_logits) + 1e-8   # (G,) >= 0
    b = ambient_raw / ambient_raw.sum()                       # (G,), sums to 1
    rate = (1.0 - rho) * base_rate + rho * b.unsqueeze(0)                     # (C, G)

    # ---------- anchor μ to library size ----------
    libsize = k.sum(dim=1).clamp_min(1.0)                                     # (C,)
    if scale.dim() == 2 and scale.size(1) == 1:
        scale = scale.squeeze(1)
    elif scale.dim() != 1:
        scale = scale.view(-1)
    assert scale.shape == (C,), "scale must be (C,) after squeeze/view"
    
    s = torch.tanh(scale) * scale_bound   # (C,)
    scale_mult = torch.exp(s)   # (C,)
    scale_eff = libsize * scale_mult

    mu = (scale_eff.unsqueeze(1) * rate).clamp_min(epsilon)                   # (C, G)

    # ---------- dispersion (θ) from MoM directly ----------
    theta = dispersion.unsqueeze(0).clamp_min(1e-3)                           # (1, G) -> broadcast to (C, G)

    # ---------- π logits: per-cell + low-rank + per-gene bias/slope ----------
    assert pi_bias.shape == (G,), "pi_bias must be (G,)"
    assert raw_pi_slope.shape == (G,), "raw_pi_slope must be (G,)"
    bias_term = pi_bias.unsqueeze(0)                                          # (1,G)
    slope_term = (-F.softplus(raw_pi_slope) - 1e-4).unsqueeze(0)              # (1,G), <= 0

    # π logits with μ-detach to avoid gate-gaming; temperature to avoid saturation
    pi_logits = pi_cell + pi_lowrank + bias_term + slope_term * torch.log1p(mu.detach())   # (C, G)
    if isinstance(tau_pi, torch.Tensor):
        tau = 1.0 + F.softplus(tau_pi)
    else:
        tau = float(tau_pi) if tau_pi >= 2.0 else 2.0
    dropout_prob = torch.sigmoid(pi_logits / tau).clamp(1e-6, 1-1e-6)      # (C, G)

    # ---------- NB log pmf ----------
    p = theta / (theta + mu)        
    if stabilize:
        log_nb = (torch.lgamma(k + theta + GLOBAL_EPS)
                  - torch.lgamma(theta + GLOBAL_EPS)
                  - torch.lgamma(k + 1.0 + GLOBAL_EPS)
                  + theta * torch.log(p + GLOBAL_EPS)
                  + k * torch.log1p(-p + GLOBAL_EPS))
    else:
        log_nb = (torch.lgamma(k + theta)
                  - torch.lgamma(theta)
                  - torch.lgamma(k + 1.0)
                  + theta * torch.log(p + GLOBAL_EPS)
                  + k * torch.log1p(-p + GLOBAL_EPS))

    # ---------- ZINB mixture ----------
    zeros_mask = (k == 0)
    log_nb0 = theta * torch.log(p + GLOBAL_EPS)                                  # log P_NB(k=0)
    nb0     = torch.exp(log_nb0)

    mix0 = dropout_prob + (1.0 - dropout_prob) * nb0
    loglik_zero  = torch.log(mix0 + GLOBAL_EPS)
    loglik_nz    = torch.log1p(-dropout_prob + GLOBAL_EPS) + log_nb

    log_likelihood = torch.where(zeros_mask, loglik_zero,
                                 weight_nnz * loglik_nz)

    loss = -log_likelihood.mean(dim=1).mean()

    # ---------- Diagnostics (1% of the time) ----------
    return loss

def zip_cell_loss(
    counts,            # (C, G) integer counts
    logits,                          # (C, K)
    scale,                           # (C, 1) or (C,)
    reference,                   # (K, G)
    *,
    # --- ZI gate params ---
    pi_bias,                         # (G,) per-gene zero-inflation bias
    raw_pi_slope,                    # (G,) per-gene zero-inflation slope (softplus'd to <= 0)
    pi_cell=None,                        # (C, 1) cell scalar (required)
    pi_lowrank=None,                          # (C, G) low-rank cell×gene term (required)
    ambient_logit=None,              # scalar Parameter/Tensor (required)
    ambient_profile_logits=None,     # (G,) (required)
    # --- hparams ---
    temperature=1.0,
    rho_cap=1.0,                     # cap for ambient fraction
    tau_pi=2.0,                      # temperature on π logits (Tensor allowed, will be >=2)
    epsilon=1e-8,
    weight_nnz=3.0,                  # upweight rare non-zeros
    scale_bound=0.5,                 # m in exp(tanh(scale)*m) → totals ×[e^-m, e^m]
    gene_detection_bias=None,
    gene_detection_offset=None,
):
    # ---------- basic checks ----------
    assert counts.dim() == 2, "counts must be (C,G)"
    C, G = counts.shape
    assert logits.size(0) == C, "logits batch size mismatch"
    assert pi_cell is not None and pi_cell.shape == (C,1), "pi_cell must be (C,1)"
    assert pi_lowrank is not None and pi_lowrank.shape == (C,G), "pi_lowrank must be (C,G)"
    assert ambient_logit is not None and ambient_profile_logits is not None, "ambient params required"
    assert ambient_profile_logits.shape == (G,), "ambient_profile_logits must be (G,)"

    device = counts.device
    k = counts

    # ---------- composition: mass-preserving with ambient ----------
    P = F.softmax(logits / temperature, dim=1)                                # (C,K)

    # 1. Apply detection efficiency to the reference S BEFORE normalization
    detection_efficiency = torch.exp(gene_detection_bias).unsqueeze(0) # (1, G)
    S_adjusted = (reference * detection_efficiency + gene_detection_offset.unsqueeze(0)).relu()
    S_norm = S_adjusted / (S_adjusted.sum(dim=1, keepdim=True) + GLOBAL_EPS)  # (K,G), row-normalized
    base_rate = P @ S_norm                                                    # (C,G), rows sum to 1


    rho = torch.sigmoid(ambient_logit) * rho_cap                              # scalar in [0, rho_cap]
    ambient_raw = F.softplus(ambient_profile_logits) + 1e-8   # (G,) >= 0
    b = ambient_raw / ambient_raw.sum()                       # (G,), sums to 1
    rate = (1.0 - rho) * base_rate + rho * b.unsqueeze(0)                     # (C,G), rows sum to 1

    # ---------- totals: μ anchored to library size with bounded scale ----------
    libsize = k.sum(dim=1).clamp_min(1.0)                                     # (C,)
    if scale.dim() == 2 and scale.size(1) == 1:
        scale = scale.squeeze(1)
    elif scale.dim() != 1:
        scale = scale.view(-1)
    assert scale.shape == (C,), "scale must be (C,) after squeeze/view"

    s = torch.tanh(scale) * scale_bound                                       # (C,)
    scale_mult = torch.exp(s)                                                 # (C,) ∈ [e^-m, e^m]
    scale_eff = libsize * scale_mult                                          # (C,)
    mu = (scale_eff.unsqueeze(1) * rate).clamp_min(epsilon)                   # (C,G)

    # ---------- π logits: per-cell + low-rank + per-gene bias/slope ----------
    assert pi_bias.shape == (G,), "pi_bias must be (G,)"
    assert raw_pi_slope.shape == (G,), "raw_pi_slope must be (G,)"
    bias_term = pi_bias.unsqueeze(0)                                          # (1,G)
    slope_term = (-F.softplus(raw_pi_slope) - 1e-4).unsqueeze(0)              # (1,G), <= 0

    # logits with μ-detach; temperature τ ≥ 2
    pi_logits = pi_cell + pi_lowrank + bias_term + slope_term * torch.log1p(mu.detach())   # (C,G)
    if isinstance(tau_pi, torch.Tensor):
        tau = 1.0 + F.softplus(tau_pi)
    else:
        tau = float(tau_pi) if tau_pi >= 2.0 else 2.0
    pi = torch.sigmoid(pi_logits / tau).clamp(1e-6, 1-1e-6)                   # (C,G)

    # ---------- Poisson log pmf ----------
    # log Pois(k|μ) = k*log μ − μ − log(k!)
    log_pois = k * torch.log(mu + GLOBAL_EPS) - mu - torch.lgamma(k + 1.0)

    # ---------- ZIP mixture ----------
    zeros_mask = (k == 0)
    pois0 = torch.exp(-mu)                                                    # P_Pois(k=0)
    mix0 = pi + (1.0 - pi) * pois0                                            # mixture zero prob

    loglik_zero  = torch.log(mix0 + GLOBAL_EPS)
    loglik_nz    = torch.log1p(-pi + GLOBAL_EPS) + log_pois                      # non-zeros cannot come from the π-mass

    log_likelihood = torch.where(zeros_mask, loglik_zero,
                                 weight_nnz * loglik_nz)
    loss = -log_likelihood.mean(dim=1).mean()

    # Optional: tiny anchor to keep μ_tot near libsize
    # ---------- Diagnostics (sample ~1% of batches) ----------
    return loss

def poisson_cell_loss(
    counts,            # (C, G) integer counts
    logits,                          # (C, K)
    scale,                           # (C, 1) or (C,)
    reference,                   # (K, G)
    *,
    ambient_logit,                   # scalar (learnable)
    ambient_profile_logits,          # (G,)  (learnable)
    temperature=1.0,
    rho_cap=0.2,                     # cap for ambient fraction
    epsilon=1e-8,
    weight_nnz=3.0,                  # upweight rare non-zeros
    scale_bound=0.5,                 # m in exp(tanh(scale)*m) → totals ×[e^-m, e^m]
    gene_detection_bias=None,
    gene_detection_offset=None,
):
    # ---- checks ----
    assert counts.dim() == 2, "counts must be (C,G)"
    C, G = counts.shape
    assert logits.size(0) == C, "logits batch size mismatch"
    assert ambient_profile_logits.shape == (G,), "ambient_profile_logits must be (G,)"

    k = counts

    # ---- composition: mass-preserving rate with ambient ----
    P = F.softmax(logits / temperature, dim=1)                                # (C,K)
    
    # 1. Apply detection efficiency to the reference S BEFORE normalization
    detection_efficiency = torch.exp(gene_detection_bias).unsqueeze(0) # (1, G)
    S_adjusted = (reference * detection_efficiency + gene_detection_offset.unsqueeze(0)).relu()
    S_norm = S_adjusted / (S_adjusted.sum(dim=1, keepdim=True) + GLOBAL_EPS)  # (K,G), row-normalized
    base_rate = P @ S_norm                                                    # (C,G), rows sum to 1

    rho = torch.sigmoid(ambient_logit) * rho_cap                              # scalar in [0, rho_cap]
    ambient_raw = F.softplus(ambient_profile_logits) + 1e-8   # (G,) >= 0
    b = ambient_raw / ambient_raw.sum()                       # (G,), sums to 1
    rate = (1.0 - rho) * base_rate + rho * b.unsqueeze(0)                     # (C,G) rows sum to 1

    # ---- totals: μ anchored to library size with bounded multiplier ----
    libsize = k.sum(dim=1).clamp_min(1.0)                                     # (C,)
    if scale.dim() == 2 and scale.size(1) == 1:
        scale = scale.squeeze(1)
    elif scale.dim() != 1:
        scale = scale.view(-1)
    assert scale.shape == (C,), "scale must be (C,) after squeeze/view"

    s = torch.tanh(scale) * scale_bound                                       # (C,)
    scale_mult = torch.exp(s)                                                 # (C,) ∈ [e^-m, e^m]
    scale_eff = libsize * scale_mult                                          # (C,)

    mu = (scale_eff.unsqueeze(1) * rate).clamp_min(epsilon)                   # (C,G)

    # ---- Poisson log pmf ----
    # log Pois(k|μ) = k*log μ − μ − log(k!)
    log_pois = k * torch.log(mu + GLOBAL_EPS) - mu - torch.lgamma(k + 1.0)

    # Weight non-zeros (rare) higher if desired
    is_zero = (k == 0)
    loglik = torch.where(is_zero, log_pois, weight_nnz * log_pois)
    loss = -loglik.mean(dim=1).mean()

    return loss


def _continuous_rate_and_mu(
    x, logits, scale, reference, ambient_logit, ambient_profile_logits,
    gene_detection_bias, gene_detection_offset, temperature, rho_cap, scale_bound,
):
    """Shared composition and mean for the two continuous-intensity losses.

    Identical in construction to the count losses: a mass-preserving
    composition over cell types, mixed with a learned ambient profile, scaled
    by the per-cell total with a bounded multiplier.
    """
    C = x.shape[0]

    P = F.softmax(logits / temperature, dim=1)                                # (C,K)

    detection_efficiency = torch.exp(gene_detection_bias).unsqueeze(0)        # (1,G)
    S_adjusted = (reference * detection_efficiency + gene_detection_offset.unsqueeze(0)).relu()
    S_norm = S_adjusted / (S_adjusted.sum(dim=1, keepdim=True) + GLOBAL_EPS)  # (K,G)
    base_rate = P @ S_norm                                                    # (C,G), rows sum to 1

    rho = torch.sigmoid(ambient_logit) * rho_cap                              # scalar in [0, rho_cap]
    ambient_raw = F.softplus(ambient_profile_logits) + 1e-8                   # (G,) >= 0
    b = ambient_raw / ambient_raw.sum()                                       # (G,), sums to 1
    rate = (1.0 - rho) * base_rate + rho * b.unsqueeze(0)                     # (C,G), rows sum to 1

    total = x.sum(dim=1).clamp_min(1.0)                                       # (C,)
    if scale.dim() == 2 and scale.size(1) == 1:
        scale = scale.squeeze(1)
    elif scale.dim() != 1:
        scale = scale.view(-1)
    assert scale.shape == (C,), "scale must be (C,) after squeeze/view"

    scale_eff = total * torch.exp(torch.tanh(scale) * scale_bound)            # (C,)
    return rate, scale_eff.unsqueeze(1) * rate                                # (C,G)


def gaussian_cell_loss(
    counts,                          # (C, G) non-negative continuous intensities
    logits,                          # (C, K)
    scale,                           # (C, 1) or (C,)
    reference,                       # (K, G)
    *,
    dispersion,                      # (G,) per-gene standard deviation sigma_g
    ambient_logit,                   # scalar (learnable)
    ambient_profile_logits,          # (G,)  (learnable)
    temperature=1.0,
    rho_cap=0.2,                     # cap for the ambient fraction
    epsilon=1e-8,
    weight_nnz=3.0,                  # extra weight on non-zeros
    scale_bound=0.5,                 # m in exp(tanh(scale)*m) -> totals x[e^-m, e^m]
    gene_detection_bias=None,
    gene_detection_offset=None,
):
    """
    Normal likelihood with a per-gene standard deviation.

        x_{c,g} ~ Normal(mu_{c,g}, sigma_g^2)

    ``mu`` is built exactly as in :func:`nb_cell_loss`, so it is non-negative
    and its row sum is anchored to the per-cell total. The likelihood is
    therefore meant for non-negative continuous intensities, not for
    z-scored data, whose row sums carry no signal.

    ``dispersion[g]`` is read as sigma_g directly; it is taken in absolute
    value and floored so a parameter that drifts negative still receives a
    gradient back towards positive.
    """
    assert counts.dim() == 2, "inputs must be (C,G)"
    C, G = counts.shape
    assert logits.size(0) == C, "logits batch size mismatch"
    assert dispersion.shape == (G,), "dispersion must be (G,)"
    assert ambient_profile_logits.shape == (G,), "ambient_profile_logits must be (G,)"

    x = counts
    _, mu = _continuous_rate_and_mu(
        x, logits, scale, reference, ambient_logit, ambient_profile_logits,
        gene_detection_bias, gene_detection_offset, temperature, rho_cap, scale_bound,
    )

    sigma = dispersion.abs().clamp_min(1e-3).unsqueeze(0)                     # (1,G)

    # log N(x | mu, sigma) = -0.5 * [ (x-mu)^2 / sigma^2 + 2 log sigma + log 2pi ]
    diff = x - mu
    log_gauss = -0.5 * (
        diff * diff / (sigma * sigma + epsilon)
        + 2.0 * torch.log(sigma + epsilon)
        + math.log(2.0 * math.pi)
    )

    is_zero = (x == 0)
    loglik = torch.where(is_zero, log_gauss, weight_nnz * log_gauss)
    return -loglik.mean(dim=1).mean()


def lognormal_cell_loss(
    counts,                          # (C, G) strictly positive intensities
    logits,                          # (C, K)
    scale,                           # (C, 1) or (C,)
    reference,                       # (K, G)
    *,
    dispersion,                      # (G,) per-gene log-space standard deviation
    ambient_logit,                   # scalar (learnable)
    ambient_profile_logits,          # (G,)  (learnable)
    temperature=1.0,
    rho_cap=0.2,                     # cap for the ambient fraction
    epsilon=1e-8,
    weight_nnz=3.0,                  # extra weight on non-zeros
    scale_bound=0.5,                 # m in exp(tanh(scale)*m) -> totals x[e^-m, e^m]
    gene_detection_bias=None,
    gene_detection_offset=None,
):
    """
    Log-normal likelihood with a per-gene log-space standard deviation.

        log x_{c,g} ~ Normal(m_{c,g}, sigma_g^2),  m = log(mu) - sigma^2 / 2

    The offset on ``m`` makes E[x] equal the ``mu`` built by the count losses.
    Zeros are clamped to ``epsilon`` before taking the log, so this likelihood
    only makes sense for strictly positive intensities.

    ``dispersion[g]`` is read as the log-space sigma_g, in absolute value and
    floored, as in :func:`gaussian_cell_loss`.
    """
    assert counts.dim() == 2, "inputs must be (C,G)"
    C, G = counts.shape
    assert logits.size(0) == C, "logits batch size mismatch"
    assert dispersion.shape == (G,), "dispersion must be (G,)"
    assert ambient_profile_logits.shape == (G,), "ambient_profile_logits must be (G,)"

    x = counts
    _, mu = _continuous_rate_and_mu(
        x, logits, scale, reference, ambient_logit, ambient_profile_logits,
        gene_detection_bias, gene_detection_offset, temperature, rho_cap, scale_bound,
    )
    mu = mu.clamp_min(epsilon)

    sigma = dispersion.abs().clamp_min(1e-3).unsqueeze(0)                     # (1,G)
    m = torch.log(mu) - 0.5 * sigma * sigma                                   # (C,G)

    x_clamped = x.clamp_min(epsilon)
    log_x = torch.log(x_clamped)

    # log LogNormal(x | m, sigma) = -(log x - m)^2 / (2 sigma^2) - log(x sigma sqrt(2pi))
    # x_clamped >= epsilon and sigma >= 1e-3, so neither log can see a zero
    log_ln = (
        -0.5 * (log_x - m) * (log_x - m) / (sigma * sigma)
        - log_x - torch.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
    )

    is_zero = (x == 0)
    loglik = torch.where(is_zero, log_ln, weight_nnz * log_ln)
    return -loglik.mean(dim=1).mean()


def total_loss(graph,
               lambda_cell_to_grid,
               cell_regions, type_regions,
               counts,
               logits,
               scale,
               reference,  # reference profiles (cell_types, genes)
               cell_loss_type='zip',
               temperature=1.0,
               type_prior=None,  # (K,) target cell-type composition for this graph's timepoint
               lambda_grid_rows=1.0,
               lambda_grid_genes=1.0,
               lambda_density=100.0,
               prior_uniform_mix=0.9,
               lambda_anatomical=0.25,
               dispersion=None,  # dispersion parameter passed from the network
               dropout_slope=0.7,
               dropout_bias=-2.0,
               pi_cell=None,  # per-cell zero-inflation logit
               pi_lowrank=None,  # low-rank cell x gene zero-inflation term
               ambient_logit=0.0,
               ambient_profile_logits=None,
               tau_pi=2.0,  # Temperature for dropout probability logits
               gene_detection_bias=None,
               gene_detection_offset=None,
               ):

    reference = F.relu(reference)

    # Grid level loss
    grid_loss, grid_cosine_loss, density_loss = grid_composition_loss(
          graph=graph,
          counts=counts,
          logits=logits,
          reference=reference,
          temperature=temperature,
          type_prior=type_prior,
          lambda_row=lambda_grid_rows,
          lambda_col=lambda_grid_genes,
          lambda_density=lambda_density,           # weight of density prior
          prior_uniform_mix=prior_uniform_mix,
          ambient_logit=ambient_logit,        # scalar tensor; if None -> rho=0
          ambient_profile_logits=ambient_profile_logits,  # (G,); ignored if rho=0
          rho_cap=0.2,
          std_thr=0.01,
          sum_thr=10,
          epsilon=1e-8,
          gene_detection_bias=gene_detection_bias,
          gene_detection_offset=gene_detection_offset,
         )

    # Cell level loss
    if cell_loss_type == 'zip':
        cell_loss = zip_cell_loss(
            counts=counts,
            logits=logits,
            scale=scale,
            reference=reference,
            pi_bias=dropout_bias,                 # (G,)
            raw_pi_slope=dropout_slope,           # (G,)
            pi_cell=pi_cell,
            pi_lowrank=pi_lowrank,
            ambient_logit=ambient_logit,
            ambient_profile_logits=ambient_profile_logits,
            temperature=temperature,
            rho_cap=0.2,
            tau_pi=tau_pi,            # learnable Tensor allowed
            epsilon=1e-8,
            weight_nnz=3.0,
            scale_bound=0.7,
            gene_detection_bias=gene_detection_bias,
            gene_detection_offset=gene_detection_offset,
        )

    elif cell_loss_type == 'zinb':
        cell_loss = zinb_cell_loss(
            counts=counts,
            logits=logits,
            scale=scale,                    # (C,1)
            reference=reference,    # (K,G)
            dispersion=dispersion,          # (G,)  MoM theta
            pi_bias=dropout_bias,           # (G,)
            raw_pi_slope=dropout_slope,     # (G,)
            pi_cell=pi_cell, pi_lowrank=pi_lowrank,
            ambient_logit=ambient_logit,
            ambient_profile_logits=ambient_profile_logits,
            temperature=temperature,
            rho_cap=0.2,
            scale_bound=0.7,
            tau_pi=tau_pi,
            weight_nnz=3.0,
            stabilize=True,
            gene_detection_bias=gene_detection_bias,
            gene_detection_offset=gene_detection_offset,
        )

    elif cell_loss_type == 'nb':
        cell_loss = nb_cell_loss(
            counts=counts,
            logits=logits,
            scale=scale,                    # (C,1)
            reference=reference,    # (K,G)
            dispersion=dispersion,          # (G,)  MoM theta
            ambient_logit=ambient_logit,
            ambient_profile_logits=ambient_profile_logits,
            temperature=temperature,
            rho_cap=0.2,
            stabilize=True,
            epsilon=1e-8,
            weight_nnz=3.0,
            scale_bound=0.7,
            gene_detection_bias=gene_detection_bias,
            gene_detection_offset=gene_detection_offset,
        )

    elif cell_loss_type == 'normal':
        cell_loss = gaussian_cell_loss(
            counts=counts,
            logits=logits,
            scale=scale,
            reference=reference,
            dispersion=dispersion,          # (G,) per-gene sigma
            ambient_logit=ambient_logit,
            ambient_profile_logits=ambient_profile_logits,
            temperature=temperature,
            rho_cap=0.2,
            epsilon=1e-8,
            weight_nnz=3.0,
            scale_bound=0.7,
            gene_detection_bias=gene_detection_bias,
            gene_detection_offset=gene_detection_offset,
        )

    elif cell_loss_type == 'log-normal':
        cell_loss = lognormal_cell_loss(
            counts=counts,
            logits=logits,
            scale=scale,
            reference=reference,
            dispersion=dispersion,          # (G,) per-gene log-space sigma
            ambient_logit=ambient_logit,
            ambient_profile_logits=ambient_profile_logits,
            temperature=temperature,
            rho_cap=0.2,
            epsilon=1e-8,
            weight_nnz=3.0,
            scale_bound=0.7,
            gene_detection_bias=gene_detection_bias,
            gene_detection_offset=gene_detection_offset,
        )

    elif cell_loss_type == 'poisson':
        cell_loss = poisson_cell_loss(
            counts=counts,
            logits=logits,
            scale=scale,
            reference=reference,
            ambient_logit=ambient_logit,
            ambient_profile_logits=ambient_profile_logits,
            temperature=temperature,
            rho_cap=0.2,
            epsilon=1e-8,
            weight_nnz=3.0,
            scale_bound=0.7,
            gene_detection_bias=gene_detection_bias,
            gene_detection_offset=gene_detection_offset,
        )

    else:
        raise ValueError(
            f"Unknown cell loss type: {cell_loss_type}. Possible values are: "
            "['zip', 'zinb', 'nb', 'poisson', 'normal', 'log-normal']."
        )

    # Anatomical loss (only when the annotation matrices are provided)
    if cell_regions is not None and type_regions is not None:
        anatomy_loss = anatomical_loss(
            logits, cell_regions, type_regions, temperature=temperature
        )
    else:
        anatomy_loss = torch.tensor(0.0, device=logits.device)

    total_loss_value = grid_loss \
                + lambda_cell_to_grid * cell_loss \
                + lambda_anatomical * anatomy_loss

    return (total_loss_value, grid_loss, grid_cosine_loss, density_loss,
            cell_loss, anatomy_loss)
