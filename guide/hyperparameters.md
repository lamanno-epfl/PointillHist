# Hyperparameter guide

Every setting below has a default that works for typical imaging-based datasets. Start
from the defaults, look at the loss curves and the assignment, and change one setting at a
time. The three families follow the three calls that take settings: `generate_graphs`,
`networks` and `train` (`predict` needs none).

## Graph generation — `ph.pp.generate_graphs`

| parameter | default | how to set it |
|---|---|---|
| `tile_side` | `"auto"` | Side of the square tiles a section is cut into, in the unit of the coordinates. `"auto"` keeps a section whole unless it exceeds 100 000 cells, in which case tiles of at most that many cells are cut with 10 % overlap. Set it yourself to control GPU memory: smaller tiles need less memory per step and let all graphs stay on the GPU. Keep tiles well above the size of the structures of interest, typically tens of thousands of cells. |
| `cell_cell_k_neighbors` | `30` | Neighbours each cell exchanges information with. The default is a safe start even for a 1 000-type atlas (the ABCA-2 example uses it). In practice, due to the long-range message passing between grid hubs, changing it won't change mappings significantly. |
| `grid_spacing` | `"auto"` | Spacing of the fine grid whose local composition is compared with the reference. `"auto"` is ten times the median nearest-neighbour distance between cells, widened below 50 transcripts per cell so that a grid point still pools enough transcripts. A grid point should cover a few dozen cells; widen the spacing for sparse data, as it helps local consistency of mappings through the gird loss. |
| `coarse_grid_side` | `7` | The coarse grid has about `side × side` nodes per tile (a hexagonal lattice with `side` columns) and captures the composition of the whole tile. |

`ph.pp.auto_graph_parameters(sections, genes)` returns the automatic values, so they can
be inspected, logged and passed back explicitly to rebuild identical graphs later. A
network predicts on the graphs it was trained with: every section, timepoint and condition
has its own learned embedding, so new sections are mapped by training on them.

## Model — `ph.tr.networks`

| parameter | default | how to set it |
|---|---|---|
| `timepoints=`, `conditions=` (arguments of `generate_graphs`) | none | One label per section; sections are always a covariate. Use them when sections come from different developmental stages or experimental conditions: each label gets a learned embedding in the network, so differences between groups are absorbed there and implictly modelled in the assignment. Leave them out for a homogeneous set of sections. |
| `gene_dropout` | `0.5` | Fraction of the counts masked in every cell during training, the main regulariser. Keep 0.5 for panels of a few hundred genes or more; lower it to 0.3 for small panels, where masking half of the genes removes too much signal. |

`hidden_size` (128) and `n_heads` (4) set the width of the transformer and rarely need
changing.

## Training — `ph.tr.train`

| parameter | default | how to set it |
|---|---|---|
| `cell_loss_type` | `"zip"` | Likelihood of the counts. `"zip"` (zero-inflated Poisson) for imaging-based data (MERFISH, Xenium, HybISS, EEL) or hundreds of cell types; `"nb"` or `"zinb"` might be a better model when counts are over-dispersed; `"poisson"` for plain counts; `"gamma"`, `"normal"` or `"log-normal"` for continuous values such as protein intensities. |
| `lambda_cell_to_grid` | `10` | Weight of the per-cell likelihood relative to the grid (composition) term. Lower it for sparse datasets with few transcripts per cell, where the counts of a single cell are noisy and the pooled composition of the grid should carry more weight. |
| `lambda_density` | `100` | Weight of the KL term that pulls the overall type composition of each tile towards the prior. Set it high, we suggest 100, for references with hundreds of types; it is annealed down during training (constant for the first half of the epochs, then a cosine decay to a tenth, `lambda_density_min`). For references with dozens of types set it lower, we suggest 10. |
| `type_priors`, `prior_uniform_mix` | `"uniform"`, `0.95` | The prior composition. `"uniform"` is a maximum-entropy prior that keeps every type in play. Pass a table of cell types × timepoint labels (counts or proportions, csv or `DataFrame`) when the expected composition of each stage is known, for example from the reference atlas itself; `prior_uniform_mix` mixes it with 5 % uniform so that no type is ruled out. |
| `type_regions`, `lambda_anatomical` | none, `0.1` | Spatial anatomical prior, see below. |
| `batch_size` | `1` | Graphs per optimiser step. 1 for large tiles; 4 for many small sections, as in the atlas example. |
| `num_epochs` | `100` | 100 is enough for the datasets we ran; the loss curves in `history` show whether fewer would do. |
| `keep_on_device` | `"auto"` | Where the graphs live while training. `"auto"` measures the memory of a step during the first epoch, then keeps all graphs on the GPU when they fit and otherwise copies each one to the GPU for its step (about 1.5× slower). `True` forces the resident mode, `False` the copying mode. |

### Spatial anatomical priors

When cells carry an anatomical annotation, it can restrict which types are allowed where.

1. Add the region of every cell to `obs` of each section (any label; unannotated cells
   stay empty) and pass its column name to `generate_graphs(..., region_key="region")`.
2. Build a binary table with one row per cell type and one column per region, 1 where the
   type may occur (it can be multiple regions), and pass it to `train(..., type_regions=table)`.

The anatomical term then penalises assignments of a type outside its allowed regions. Its
weight `lambda_anatomical` (0.1) ramps up from 0 over training; a larger value enforces the
annotation more strictly, and `type_regions=None` switches the term off.
