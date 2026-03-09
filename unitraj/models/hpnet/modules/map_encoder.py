from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj

from ..layers.graph_attention import GraphAttention
from ..layers.two_layer_mlp import TwoLayerMLP
from ..utils.process_data import compute_angles_lengths_2D
from ..utils.init_weights import init_weights
from ..utils.process_data import wrap_angle
from ..utils.process_data import transform_point_to_local_coordinate
from ..utils.process_data import generate_reachable_matrix

class MapEncoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_hops:int, 
                 num_heads: int,
                 dropout: float) -> None:
        super(MapEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout

        self._l2l_edge_type = ['adjacent', 'predecessor', 'successor']

        self.c_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.c2l_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l2l_emb_layer = TwoLayerMLP(input_dim=7, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.c2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.l2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True)

        self.apply(init_weights)

    def forward(self, data: Batch) -> torch.Tensor:
        #embedding
        c_length = data['centerline']['length']
        c_embs = self.c_emb_layer(input=c_length.unsqueeze(-1))        #[(C1,...,Cb),D]

        l_length = data['lane']['length']
        l_is_intersection = data['lane']['is_intersection']
        l_turn_direction = data['lane']['turn_direction']
        l_traffic_control = data['lane']['traffic_control']
        l_input = torch.stack([l_length, l_is_intersection, l_turn_direction, l_traffic_control], dim=-1)        #[(M1,...,Mb),4]
        l_embs = self.l_emb_layer(input=l_input)                      #[(M1,...,Mb),D]

        #edge
        #c2l
        c2l_position_c = data['centerline']['position']             #[(C1,...,Cb),2]
        c2l_position_l = data['lane']['position']                   #[(M1,...,Mb),2]
        c2l_heading_c = data['centerline']['heading']               #[(C1,...,Cb)]
        c2l_heading_l = data['lane']['heading']                     #[(M1,...,Mb)]
        c2l_edge_index = data['centerline', 'lane']['centerline_to_lane_edge_index']    #[2,(C1,...,Cb)]
        c2l_edge_vector = transform_point_to_local_coordinate(c2l_position_c[c2l_edge_index[0]], c2l_position_l[c2l_edge_index[1]], c2l_heading_l[c2l_edge_index[1]])
        c2l_edge_attr_length, c2l_edge_attr_theta = compute_angles_lengths_2D(c2l_edge_vector)
        c2l_edge_attr_heading = wrap_angle(c2l_heading_c[c2l_edge_index[0]] - c2l_heading_l[c2l_edge_index[1]])
        c2l_edge_attr_input = torch.stack([c2l_edge_attr_length, c2l_edge_attr_theta, c2l_edge_attr_heading], dim=-1)
        c2l_edge_attr_embs = self.c2l_emb_layer(input = c2l_edge_attr_input)

        #l2l
        l2l_position = data['lane']['position']                     #[(M1,...,Mb),2]
        l2l_heading = data['lane']['heading']                       #[(M1,...,Mb)]
        l2l_edge_index = []
        l2l_edge_attr_type = []
        l2l_edge_attr_hop = []
        
        l2l_adjacent_edge_index = data['lane', 'lane']['adjacent_edge_index']
        num_adjacent_edges = l2l_adjacent_edge_index.size(1)
        l2l_edge_index.append(l2l_adjacent_edge_index)
        l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('adjacent')), num_classes=len(self._l2l_edge_type)).to(l2l_adjacent_edge_index.device).repeat(num_adjacent_edges, 1))
        l2l_edge_attr_hop.append(torch.ones(num_adjacent_edges, device=l2l_adjacent_edge_index.device))

        num_lanes = data['lane']['num_nodes']
        l2l_predecessor_edge_index = data['lane', 'lane']['predecessor_edge_index']
        # Compute max_nodes from actual edge indices to handle variable lane counts per sample
        max_pred_node = l2l_predecessor_edge_index.max().item() + 1 if l2l_predecessor_edge_index.numel() > 0 else num_lanes
        max_pred_node = max(max_pred_node, num_lanes)
        l2l_predecessor_edge_index_all = generate_reachable_matrix(l2l_predecessor_edge_index, self.num_hops, max_pred_node)
        for i in range(self.num_hops):
            num_edges_now = l2l_predecessor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_predecessor_edge_index_all[i])
            l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('predecessor')), num_classes=len(self._l2l_edge_type)).to(l2l_predecessor_edge_index.device).repeat(num_edges_now, 1))
            l2l_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=l2l_predecessor_edge_index.device))
        
        l2l_successor_edge_index = data['lane', 'lane']['successor_edge_index']
        # Compute max_nodes from actual edge indices to handle variable lane counts per sample
        max_succ_node = l2l_successor_edge_index.max().item() + 1 if l2l_successor_edge_index.numel() > 0 else num_lanes
        max_succ_node = max(max_succ_node, num_lanes)
        l2l_successor_edge_index_all = generate_reachable_matrix(l2l_successor_edge_index, self.num_hops, max_succ_node)
        for i in range(self.num_hops):
            num_edges_now = l2l_successor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_successor_edge_index_all[i])
            l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('successor')), num_classes=len(self._l2l_edge_type)).to(l2l_successor_edge_index.device).repeat(num_edges_now, 1))
            l2l_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=l2l_successor_edge_index.device))

        l2l_edge_index = torch.cat(l2l_edge_index, dim=1)
        l2l_edge_attr_type = torch.cat(l2l_edge_attr_type, dim=0)
        l2l_edge_attr_hop = torch.cat(l2l_edge_attr_hop, dim=0)
        l2l_edge_vector = transform_point_to_local_coordinate(l2l_position[l2l_edge_index[0]], l2l_position[l2l_edge_index[1]], l2l_heading[l2l_edge_index[1]])
        l2l_edge_attr_length, l2l_edge_attr_theta = compute_angles_lengths_2D(l2l_edge_vector)
        l2l_edge_attr_heading = wrap_angle(l2l_heading[l2l_edge_index[0]] - l2l_heading[l2l_edge_index[1]])
        l2l_edge_attr_input = torch.cat([l2l_edge_attr_length.unsqueeze(-1), l2l_edge_attr_theta.unsqueeze(-1), l2l_edge_attr_heading.unsqueeze(-1), l2l_edge_attr_hop.unsqueeze(-1), l2l_edge_attr_type], dim=-1)
        l2l_edge_attr_embs = self.l2l_emb_layer(input=l2l_edge_attr_input)

        #attention
        #c2l
        l_embs = self.c2l_attn_layer(x = [c_embs, l_embs], edge_index = c2l_edge_index, edge_attr = c2l_edge_attr_embs)         #[(M1,...,Mb),D]

        #l2l
        l_embs = self.l2l_attn_layer(x = l_embs, edge_index = l2l_edge_index, edge_attr = l2l_edge_attr_embs)                   #[(M1,...,Mb),D]

        return l_embs