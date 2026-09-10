import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import LayerNorm, SimpleConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import remove_self_loops, softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index

__all__ = ["PointillHistNet", "SpatialHGTConv", "FourierDistanceEncoder"]


class FourierDistanceEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        # We assume input is normalized 0-1.
        # Frequencies: 2^0, 2^1, ... 2^(out_dim/2)
        # We want the lowest freq to cover the full 0-1 range (pi),
        # and highest freq to cover 1/2^(N) variations.
        self.register_buffer(
            "freq_bands", 
            2.0 ** torch.linspace(0.0, (out_dim // 2) - 1, steps=out_dim // 2)
        )

    def forward(self, dist):
        # dist: [num_edges] assumed to be in range [0, ~1]
        x = dist.unsqueeze(-1)
        x_proj = x * self.freq_bands * math.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)

class PointillHistNet(nn.Module):
    """
    Heterogeneous graph transformer mapping cells (and a coarse grid) to
    cell-type logits, expression scale and zero-inflation parameters.

    Build it with :func:`networks`, which reads the sizes from the graphs.
    """

    def __init__(self, n_classes: int, n_genes: int, n_timepoints: int = 1, n_sections: int = 1,
                 n_conditions: int = 1,
                 n_heads: int = 4, n_hgt_layers: int = 3,
                 hidden_size: int = 128,
                 time_embed_size: int = 8, section_embed_size: int = 16, condition_embed_size: int = 8,
                 activation: str = "silu",
                 smooth=False, smoothing_factor=0.3,
                 gene_dropout: float = 0.5, cell_edge_dropout: float = 0.2, counts_bypass: bool = True):
        super().__init__()
        
        # ----------------- CONFIGURATION -----------------
        self.n_classes = n_classes
        self.n_genes = n_genes
        self.hidden_size = hidden_size
        self.activation_fn = self._get_activation_fn(activation)
        self.smooth = smooth
        self.smoothing_factor = smoothing_factor
        self.gene_dropout = gene_dropout
        self.cell_edge_dropout = cell_edge_dropout
        self.counts_bypass = counts_bypass
        self.counts_gate = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        # Embedding sizes
        self.time_embed_size = time_embed_size
        self.section_embed_size = section_embed_size
        self.condition_embed_size = condition_embed_size
        
        # ----------------- LOSS PARAMETERS (REQUIRED) -----------------
        self.dispersion = nn.Parameter(torch.ones(n_genes, ), requires_grad=True)
        self.pi_bias  = nn.Parameter(torch.full((self.n_genes,), -2.0), requires_grad=True)
        self.pi_slope = nn.Parameter(torch.full((self.n_genes,), 0.7), requires_grad=True)
        self.ambient_logit = nn.Parameter(torch.tensor(-2.0), requires_grad=True)          
        self.ambient_profile_logits = nn.Parameter(torch.zeros(n_genes), requires_grad=True)  
        self.gene_detection_bias = nn.Parameter(torch.zeros(n_genes), requires_grad=True)
        self.gene_detection_offset = nn.Parameter(torch.zeros(n_genes), requires_grad=False)
        self.pi_logit_temp = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # ----------------- INPUT PROJECTION -----------------
        self.cell_input_proj = nn.Sequential(
            nn.Linear(n_genes, hidden_size), 
            nn.LayerNorm(hidden_size),
            self.activation_fn
        )

        # ----------------- HGT BLOCK -----------------
        node_types = ['cells', 'longrange_grid']
        metadata = (
            node_types,
            [
                ('cells', 'is_close_to', 'cells'),
                ('longrange_grid', 'is_close_to', 'longrange_grid'),
                ('cells', 'is_watched_by', 'longrange_grid'),
                ('longrange_grid', 'rev_watched_by', 'cells') 
            ]
        )

        self.grid_initializer = SimpleConv(aggr="mean")
        # Geometric Edge Encoder
        self.edge_dist_encoder = FourierDistanceEncoder(out_dim=16)

        self.hgt_layers = nn.ModuleList()
        for _ in range(n_hgt_layers):
            self.hgt_layers.append(
                SpatialHGTConv(in_channels=hidden_size,
                        out_channels=hidden_size,
                        metadata=metadata,
                        heads=n_heads,
                        edge_dim=16)
            )

        # ----------------- HEADS & OUTPUTS -----------------
        self.norm_counts = LayerNorm(n_genes) 
        self.norm_cell = LayerNorm(hidden_size) 
        self.norm_grid = LayerNorm(hidden_size) 

        # Global covariate embeddings (Time/Section/Condition)
        self.time_embedding = nn.Embedding(n_timepoints, self.time_embed_size)
        self.section_embedding = nn.Embedding(n_sections, self.section_embed_size)
        self.condition_embedding = nn.Embedding(n_conditions, self.condition_embed_size)
        
        # Calculate Input Dimension for Final MLP
        # It is: Raw_Genes + HGT_Cell + HGT_Grid + Time + Section + Condition
        self.feature_dim = (n_genes + hidden_size + hidden_size + 
                                 time_embed_size + section_embed_size + condition_embed_size)

        # Probabilistic Heads Components
        r = 32
        self.pi_cell_bias = nn.Linear(self.feature_dim, 1)
        self.pi_lowrank_U = nn.Linear(self.feature_dim, r, bias=False)
        self.pi_lowrank_V = nn.Linear(r, self.n_genes, bias=False)
        
        # Scale guessing
        self.scale_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            self.activation_fn,
            nn.Linear(hidden_size, 1),
        )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.LayerNorm(1024),
            self.activation_fn,
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            self.activation_fn,
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            self.activation_fn,
            nn.Linear(512, self.n_classes),
        )

        # ----------------- HELPERS -----------------
        self.agg_dotcell = SimpleConv(aggr="sum", flow="source_to_target")
        self.smooth_conv = SimpleConv(aggr="mean", flow="source_to_target")
        self.count_dropout = torch.nn.Dropout(p=self.gene_dropout)

        self.dist_scalers = nn.ParameterDict({
            # Cell-Cell: usually very local (< 20um)
            "cells__is_close_to__cells": nn.Parameter(torch.tensor(50.0), requires_grad=False),
            
            # Cell-Grid: moderate range (< 50um or half grid spacing)
            "cells__is_watched_by__longrange_grid": nn.Parameter(torch.tensor(300.0), requires_grad=False),
            "longrange_grid__rev_watched_by__cells": nn.Parameter(torch.tensor(300.0), requires_grad=False),
            
            # Grid-Grid: Long range (grid spacing, e.g., 100um)
            "longrange_grid__is_close_to__longrange_grid": nn.Parameter(torch.tensor(600.0), requires_grad=False)
        })

        # Softmax temperature used by the losses and by predict; set by `train`.
        # Registered last so the parameter creation order above is unchanged.
        self.register_buffer("temperature", torch.tensor(1.0))

    def _get_activation_fn(self, name: str) -> nn.Module:
        if name == "relu": return nn.ReLU()
        elif name == "gelu": return nn.GELU()
        elif name == "silu": return nn.SiLU()
        else: return nn.ReLU()

    def forward(self, graph):
        """
        Run the model on one HeteroData graph.

        Returns ``counts_raw, logits, cell_embed, grid_embed, scale, (pi_cell, pi_lowrank)``.
        The timepoint / section / condition embeddings are indexed by the
        graph-level ints ``graph.timepoint``, ``graph.section``, ``graph.condition``.
        """
        # 1. Input Aggregation
        try:
            counts = graph["cells"].x
            # apply dropout
            counts = self.count_dropout(counts)
            counts_raw = graph["cells"].x 
        except AttributeError:
            genes_one_hot = self.count_dropout(graph["dots"].x)
            counts = self.agg_dotcell(
                (genes_one_hot, graph["cells"].pos),
                graph["dots", "could_come_from", "cells"].edge_index,
            )
            counts_raw = self.agg_dotcell(
                (graph["dots"].x, graph["cells"].pos),
                graph["dots", "could_come_from", "cells"].edge_index,
            )

        if self.smooth:
            edge_index_no_self, _ = remove_self_loops(graph["cells", "is_close_to", "cells"].edge_index)
            nbr_mean = self.smooth_conv((counts, graph["cells"].pos), edge_index_no_self)
            counts = self.smoothing_factor * counts + (1 - self.smoothing_factor) * nbr_mean

        # 2. HGT Inputs
        h_cells = self.cell_input_proj(counts)
        
        h_grid = self.grid_initializer(
            (h_cells, graph["longrange_grid"].pos), # Tuple: (Source Feats, Target Coords - dummy here)
            graph["cells", "is_watched_by", "longrange_grid"].edge_index
        )

        x_dict = {
            'cells': h_cells,
            'longrange_grid': h_grid
        }

        # 3. Edge Dropout
        # drop direct cell-cell edges to force usage of the grid bridge
        cell_cell_index = graph["cells", "is_close_to", "cells"].edge_index
        
        if self.training:
            # Drop X% of cell-cell edges
            mask = torch.rand(cell_cell_index.size(1), device=cell_cell_index.device) > self.cell_edge_dropout
            cell_cell_index = cell_cell_index[:, mask]

        edge_index_dict = {
            ('cells', 'is_close_to', 'cells'): cell_cell_index,
            ('longrange_grid', 'is_close_to', 'longrange_grid'): graph["longrange_grid", "is_close_to", "longrange_grid"].edge_index,
            ('cells', 'is_watched_by', 'longrange_grid'): graph["cells", "is_watched_by", "longrange_grid"].edge_index,
            ('longrange_grid', 'rev_watched_by', 'cells'): torch.flip(graph["cells", "is_watched_by", "longrange_grid"].edge_index, [0])
        }

        edge_dist_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_key = "__".join(edge_type) # Helper to match dict keys
            
            # 1. Calc Raw Euclidean Distance
            src_pos = graph[src_type].pos[edge_index[0]]
            dst_pos = graph[dst_type].pos[edge_index[1]]
            raw_dists = torch.norm(src_pos - dst_pos, p=2, dim=-1)
            
            # 2. Normalize by the specific scale of this edge type
            # Clamp to 1.0 to prevent frequency aliasing if outliers exist
            max_scale = self.dist_scalers[edge_key]
            norm_dists = torch.clamp(raw_dists / max_scale, max=1.0)
            
            # 3. Encode
            edge_dist_dict[edge_type] = self.edge_dist_encoder(norm_dists)

        # 4. Run HGT
        for hgt in self.hgt_layers:
            x_dict = hgt(x_dict, edge_index_dict, edge_attr_dict=edge_dist_dict)
            x_dict['cells'] = F.silu(x_dict['cells'])
            x_dict['longrange_grid'] = F.silu(x_dict['longrange_grid'])

        final_cell_embed = x_dict['cells']
        final_grid_embed = x_dict['longrange_grid']

        # 5. Retrieve Grid Features (Back to Cell context)
        final_grid_embed_retrieved = SimpleConv(aggr="mean", flow="source_to_target")(
            (final_grid_embed, torch.arange(graph["cells"].num_nodes, device=h_cells.device)[:, None]),
            edge_index_dict[('longrange_grid', 'rev_watched_by', 'cells')]
        )

        # 6. Prepare Global Embeddings
        xy_positions = graph['cells'].pos[:, :2]
        s_section = torch.tensor(graph.section, device=h_cells.device)
        c_condition = torch.tensor(graph.condition, device=h_cells.device)
        t_time = torch.tensor(graph.timepoint, device=h_cells.device)
        section_embed = self.section_embedding(s_section).repeat(len(xy_positions), 1)
        condition_embed = self.condition_embedding(c_condition).repeat(len(xy_positions), 1)
        time_embed = self.time_embedding(t_time).repeat(len(xy_positions), 1)

        if self.counts_bypass:
            counts_gated = self.norm_counts(counts) * torch.sigmoid(self.counts_gate)
        else: # disable raw gene bypass
            counts_gated = torch.zeros_like(counts) 

        # Final Stack (Concatenate EVERYTHING)
        final_stack = torch.cat(
            [
                counts_gated,                    # Raw Genes
                self.norm_cell(final_cell_embed),      # HGT Cell
                self.norm_grid(final_grid_embed_retrieved), # HGT Grid
                section_embed,                           # Global Context
                time_embed,
                condition_embed
            ],
            dim=-1,
        )

        # 7. Predictions
        scale = self.scale_head(final_stack)
        logits = self.classifier(final_stack)

        # Zero Inflation outputs
        pi_cell = self.pi_cell_bias(final_stack)
        pi_lowrank   = self.pi_lowrank_V(F.silu(self.pi_lowrank_U(final_stack)))

        return counts_raw, logits, final_cell_embed, final_grid_embed, scale, (pi_cell, pi_lowrank)


class SpatialHGTConv(MessagePassing):
    """The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        edge_dim: int = None,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = {key[-1] for key in self.edge_types}

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        # Projection for continuous edge attributes (Geometry)
        if edge_dim is not None:
            self.edge_lin = torch.nn.Linear(edge_dim, heads)
        else:
            self.edge_lin = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],  # Support both.
        edge_attr_dict: Dict[EdgeType, Tensor] = None
    ) -> Dict[NodeType, Optional[Tensor]]:
        """Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[torch.Tensor]]` - The output node
            embeddings for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))

        edge_bias = None
        if self.edge_lin is not None and edge_attr_dict is not None:
            # We must iterate in the exact same order as construct_bipartite_edge_index
            # which iterates over edge_index_dict.keys()
            geo_attrs = []
            for edge_type in edge_index_dict.keys():
                geo_attrs.append(edge_attr_dict[edge_type])
            
            # Stack [Total_Edges, Edge_Dim] and Project to [Total_Edges, Heads]
            geo_tensor = torch.cat(geo_attrs, dim=0)
            edge_bias = self.edge_lin(geo_tensor)

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr, edge_bias=edge_bias)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                edge_bias: Optional[Tensor],
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        # Inject Geometric Bias (Distance) into the attention score
        if edge_bias is not None:
            alpha = alpha + edge_bias
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
    
