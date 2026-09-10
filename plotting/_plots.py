from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

__all__ = [
    "fovs",
    "visualize_interaction_metrics",
    "spatial_interaction",
    "history",
    "spatial_cells",
    "plot_2d",
]

def fovs(graphs, downsample_rate=0.1, figscale=10):
    """
    Plot the tiling of every section into tiles (one panel per section).

    Parameters:
        graphs: List of HeteroData graphs as returned by generate_graphs.
        downsample_rate: Fraction of the core cells of each tile to draw.
        figscale: Figure width in inches (the height follows the panel grid).
    """
    section_tiles = defaultdict(list)
    for graph in graphs:
        section_tiles[graph.section_label].append(graph)

    c = int(np.sqrt(len(section_tiles)))
    r = int(np.ceil(len(section_tiles) / c))
    plt.figure(figsize=(figscale, figscale * r / c))
    gs = gridspec.GridSpec(r, c)
    for n, tiles in enumerate(section_tiles.values()):
        ax = plt.subplot(gs[n])
        counter_cells = 0
        for graph in tiles:
            cells = graph["cells"].pos.cpu().numpy()
            cells_bool = graph["cells"].is_core.cpu().numpy()
            gridpoints = graph["longrange_grid"].pos.cpu().numpy()
            vertexes_fov = graph["vertexes_fov"]
            core_vertexes_fov = graph["core_vertexes_fov"]
            # downsample the cells
            rand_ix_cell = np.random.choice(
                int(cells_bool.sum()),
                int(cells_bool.sum() * downsample_rate),
                replace=False,
            )

            plt.scatter(
                cells[:, 0][cells_bool][rand_ix_cell],
                cells[:, 1][cells_bool][rand_ix_cell],
                s=2 / r,
                alpha=0.4,
                color=plt.cm.terrain((np.random.uniform(0, 25)) / 25),
            )

            plt.scatter(
                gridpoints[:, 0],
                gridpoints[:, 1],
                edgecolors="white",
                facecolors="none",
                s=6 / r,
                alpha=1.0,
                lw=1,
            )
            plt.plot(
                [
                    vertexes_fov[0],
                    vertexes_fov[1],
                    vertexes_fov[1],
                    vertexes_fov[0],
                    vertexes_fov[0],
                ],
                [
                    vertexes_fov[2],
                    vertexes_fov[2],
                    vertexes_fov[3],
                    vertexes_fov[3],
                    vertexes_fov[2],
                ],
                "red",
                linestyle="dashed",
            )
            plt.plot(
                [
                    core_vertexes_fov[0],
                    core_vertexes_fov[1],
                    core_vertexes_fov[1],
                    core_vertexes_fov[0],
                    core_vertexes_fov[0],
                ],
                [
                    core_vertexes_fov[2],
                    core_vertexes_fov[2],
                    core_vertexes_fov[3],
                    core_vertexes_fov[3],
                    core_vertexes_fov[2],
                ],
                "cyan",
            )
            counter_cells += cells_bool.sum()
        ax.set_title(f"cells:{counter_cells}")
        ax.set_facecolor("black")
        ax.set_aspect("equal")
    plt.tight_layout()


def visualize_interaction_metrics(avg_matrices, cell_types, sample_id, threshold=100, if_remove_self=False):
    """
    Visualize the interaction matrix of one section as a heatmap.

    Parameters:
        avg_matrices: Dictionary returned by cell_cell_interactions().
        cell_types: List of cell type names (``result["cell_types"]``).
        sample_id: Section label whose interaction matrix to visualize.
        threshold: Keep only cell types whose row sum exceeds this value.
        if_remove_self: Zero the diagonal and row-normalise.
    """
    interaction = avg_matrices[sample_id]
    cell_labels = np.asarray(cell_types)

    # Filter rows and columns with a sum greater than the threshold.
    rows_keep = interaction.sum(axis=1) > threshold
    cols_keep = rows_keep  # assuming symmetry
    filtered_interaction = interaction[rows_keep][:, cols_keep]
    filtered_yticklabels = cell_labels[rows_keep]
    filtered_xticklabels = cell_labels[cols_keep]

    if if_remove_self:
        # Remove the diagonal elements (self-interactions)
        np.fill_diagonal(filtered_interaction, 0)
        row_sums = filtered_interaction.sum(axis=1, keepdims=True)
        filtered_interaction = filtered_interaction / row_sums

    plt.figure(figsize=(10, 10), dpi=200)
    sns.heatmap(filtered_interaction,
                cmap='hot',
                xticklabels=filtered_xticklabels,
                yticklabels=filtered_yticklabels)
    plt.title(f"Interaction Metrics for {sample_id}")
    plt.show()


def spatial_interaction(
    interaction_matrix,
    result,
    section_id,
    min_cells=100,
    min_weight=150,
    bg_color='grey',
    node_color='orange',
    edge_color='black',
    figsize=8,
    dpi=200,
    point_size=1,
    node_size=20,
    edge_scale=5,
    label_fontsize=5,
    save_path=None,
    show=True
):
    """
    Plot spatial cell-type interaction network for a given section.

    Parameters
    ----------
    interaction_matrix : np.ndarray, shape (C, C)
        Raw interaction weights between C cell types.
    result : dict
        Dictionary returned by predict() (uses all_sections, all_positions,
        all_labels and cell_types).
    section_id : str
        Section label to plot (one of ``result["section_ids"]``).
    min_cells : int
        Filter cell types with at least this many cells.
    min_weight : float
        Only draw edges with weight > min_weight.
    bg_color : str
        Color for all cells background.
    node_color : str
        Color for cell-type node markers.
    edge_color : str
        Color for interaction edges.
    figsize : float
        Width and height of the figure in inches.
    dpi : int
        Resolution of the figure.
    point_size : float
        Size for background points.
    node_size : float
        Size for node markers.
    edge_scale : float
        Scale factor for edge linewidth: lw = weight/max_weight * edge_scale.
    label_fontsize : float
        Font size for node labels.
    save_path : str, optional
        File path to save figure.
    show : bool
        Whether to plt.show().
    """

    in_section = result['all_sections'] == section_id
    if not in_section.any():
        raise ValueError(f"No cells found for section_id={section_id}")
    positions = result['all_positions'][in_section]
    labels = result['all_labels'][in_section]

    # Setup figure
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)

    # Background scatter
    ax.scatter(
        positions[:, 1], positions[:, 0],
        s=point_size, c=bg_color, alpha=0.5, label='All cells'
    )

    # Filter cell types
    mask = labels != -1
    labels_filt = labels[mask]
    positions_filt = positions[mask]
    types, counts = np.unique(labels_filt, return_counts=True)
    filtered = types[counts >= min_cells]

    if len(filtered) == 0:
        raise ValueError(f"No cell types with ≥{min_cells} cells")

    # Median positions per type
    med_pos = {ct: np.median(positions[labels == ct], axis=0)
               for ct in filtered}

    # Filter interaction matrix
    filt_mat = interaction_matrix[np.ix_(filtered, filtered)]

    # Cell type names
    filt_names = np.asarray(result['cell_types'])[filtered]

    # Edges
    max_w = filt_mat.max() if filt_mat.max() > 0 else 1
    for i, ct1 in enumerate(filtered):
        for j, ct2 in enumerate(filtered):
            w = filt_mat[i, j]
            if w > min_weight:
                p1 = med_pos[ct1]
                p2 = med_pos[ct2]
                ax.plot(
                    [p1[1], p2[1]], [p1[0], p2[0]],
                    '-', color=edge_color,
                    lw=(w / max_w) * edge_scale,
                    alpha=0.7
                )

    # Nodes and labels
    for idx, ct in enumerate(filtered):
        p = med_pos[ct]
        ax.scatter(
            p[1], p[0],
            s=node_size, edgecolor='black', facecolor=node_color, zorder=3
        )
        ax.text(
            p[1], p[0], filt_names[idx],
            fontsize=label_fontsize, ha='center', va='center',
            color='white', weight='bold', zorder=4
        )

    # Final touches
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if section_id and not save_path:
        ax.set_title(f"{section_id}", fontsize=10, weight='bold')

    ax.legend(loc='upper right')
    plt.tight_layout()

    # Save/show
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def history(history):
    """
    Plots each training loss history in its own subplot for clarity.

    Parameters
    ----------
    history : dict
        Mapping loss names ('total_loss', 'cell_loss', 'grid_loss',
        'grid_cosine_loss', 'density_loss', 'anatomical_loss') to lists of
        values per epoch. 'anatomical_loss' is plotted only if not all zeros.
    """
    # Define order of possible losses
    candidate_keys = [
        'total_loss', 'cell_loss', 'grid_loss',
        'grid_cosine_loss', 'density_loss', 'anatomical_loss',
    ]
    # Skip missing series, and the anatomical one when it is off
    keys_to_plot = []
    for key in candidate_keys:
        vals = history.get(key)
        if vals is None:
            continue
        if key == 'anatomical_loss' and all(v == 0 for v in vals):
            continue
        keys_to_plot.append(key)

    if not keys_to_plot:
        print("No losses available to plot.")
        return

    num_plots = len(keys_to_plot)
    epochs = list(range(1, len(history[keys_to_plot[0]]) + 1))

    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 3 * num_plots), sharex=True)
    # Ensure axes is iterable
    if num_plots == 1:
        axes = [axes]

    fig.suptitle("Training Loss Curves", fontsize=16, fontweight='bold', y=0.95)

    for ax, key in zip(axes, keys_to_plot):
        values = history[key]
        label = key.replace('_', ' ').title()
        ax.plot(epochs, values, linewidth=2)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.5)

    axes[-1].set_xlabel("Epoch", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_2d(
    embeddings2d,
    key,
    title=None,
    save_path=None,
    show=True,
    figsize_per=5,
    point_size=1,
    alpha=0.8
):
    """
    Plot precomputed 2D embeddings colored by key labels/values.

    Parameters
    ----------
    embeddings2d : array-like of shape (n_points, 2)
    key : array-like of length n_points, or list of such arrays
        Labels or continuous values; list yields subplots.
    title, save_path, show : as in umap().
    figsize_per : float
        Inches per subplot.
    point_size : int
        Scatter marker size.
    alpha : float
        Scatter transparency.
    """
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2)

    keys = key if isinstance(key, (list, tuple)) else [key]
    n_keys = len(keys)
    X = np.asarray(embeddings2d)

    fig, axes = plt.subplots(
        1, n_keys,
        figsize=(figsize_per * n_keys, figsize_per),
        squeeze=False,
        dpi=300
    )
    axes = axes[0]

    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    css_vals = list(mcolors.CSS4_COLORS.values())
    n_css = len(css_vals)

    for ax, vals in zip(axes, keys):
        vals = np.asarray(vals)
        # Determine colors
        if vals.dtype.kind in ('U','S','O') or (
            np.issubdtype(vals.dtype, np.integer)
        ):
            cats, inv = np.unique(vals, return_inverse=True)
            nc = len(cats)
            if nc <= 10:
                palette = sns.color_palette("tab10", n_colors=nc)
                colors = np.array(palette)[inv]
            elif nc <= 20:
                palette = sns.color_palette("hls", n_colors=nc)
                colors = np.array(palette)[inv]
            else:
                colors = np.array(css_vals)[inv % n_css]
        else:
            norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
            cmap = sns.color_palette("viridis", as_cmap=True)
            colors = cmap(norm(vals))

        ax.scatter(
            X[:,0], X[:,1],
            s=point_size,
            c=colors,
            alpha=alpha,
            linewidths=0
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        name = getattr(vals, 'name', None)
        ax.set_title(
            name.replace('_', ' ').title() if name else '',
            fontsize=14,
            fontweight='medium'
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def spatial_cells(results, celltypes, save_path=None, show=True, figscale=30, legend_fontsize=15, legend_markerscale=2, markerscale_bg=2, markerscale_fg=2):
    """
    Plot the predicted cells of every section, colouring the chosen cell types.

    Parameters
    ----------
    results : dict
        Dictionary returned by predict().
    celltypes : list of str
        Cell type names to colour (must be in ``results["cell_types"]``).
    save_path : str, optional
        File path to save the figure (png, svg, pdf).
    show : bool
        Whether to plt.show() at the end.
    figscale : float
        Size (inches) of the figure.
    legend_fontsize, legend_markerscale : legend font size and marker scale.
    markerscale_bg, markerscale_fg : marker sizes of background and coloured cells.
    """

    all_sections = results['all_sections']
    section_ids = results['section_ids']
    all_positions = results['all_positions']
    all_labels = results['all_labels']
    names = list(results['cell_types'])

    N = len(section_ids)
    r = int(np.sqrt(N))
    c = int(np.ceil(N / r))

    celltypes = [names.index(name) for name in celltypes]

    # Create a colormap for each name
    if len(celltypes) <= 10:
        colors = list(mcolors.TABLEAU_COLORS.values())[:len(celltypes)]
    else:
        cmap = plt.get_cmap('tab20', max(20, len(celltypes)))
        colors = [cmap(i) for i in range(len(celltypes))]

    plt.figure(figsize=(figscale, figscale * r / c), dpi=300)
    gs = gridspec.GridSpec(r, c)
    
    for n in range(N):
        ax = plt.subplot(gs[n])
        bool_ix = all_sections == section_ids[n]

        # Plot background points (grey)
        ax.scatter(all_positions[:, 0][bool_ix], all_positions[:, 1][bool_ix], 
                   c='lightgrey', s=markerscale_bg/c)

        # Plot each label in its own color
        for i, name in enumerate(celltypes):
            new_labels = all_labels == name
            
            # Scatter plot for each name with its corresponding color
            ax.scatter(all_positions[:, 0][bool_ix & new_labels], all_positions[:, 1][bool_ix & new_labels],
                       c=colors[i], s=markerscale_fg/c, label=names[name])

        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])

        # Create a proxy artist for each color to add to the legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=5, label=names[name])
                    for i, name in enumerate(celltypes)]
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize=legend_fontsize, markerscale=legend_markerscale, title="Labels", title_fontsize=legend_fontsize)
        ax.set_title(section_ids[n], fontsize=8)
        ax.axis("equal")
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()