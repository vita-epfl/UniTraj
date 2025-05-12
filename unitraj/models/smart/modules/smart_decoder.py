import math
import psutil
import os
import pickle
import torch
import torch.nn as nn

# from unitraj.models.smart.modules.agent_decoder import SMARTAgentDecoder
# from unitraj.models.smart.modules.map_decoder import SMARTMapDecoder
from unitraj.models.smart.smart_utils import check_nan_inf
from unitraj.models.smart.layers.attention_layer import AttentionLayer
from unitraj.models.smart.layers import MLPLayer
from unitraj.models.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from typing import Dict, Mapping, Optional
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph
from unitraj.models.smart.smart_utils import (
    angle_between_2d_vectors,
    weight_init,
    wrap_angle,
)


class SMARTDecoder(nn.Module):

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        map_token: Dict,
        token_data: Dict,
        use_intention=False,
        token_size=512,
    ) -> None:
        super(SMARTDecoder, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_token=map_token,
        )
        self.agent_encoder = SMARTAgentDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            token_size=token_size,
            token_data=token_data,
        )
        self.map_enc = None

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}

    def inference(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder.inference(data, map_enc)
        return {**map_enc, **agent_enc}

    def inference_no_map(self, data: HeteroData, map_enc) -> Dict[str, torch.Tensor]:
        agent_enc = self.agent_encoder.inference(data, map_enc)
        return {**map_enc, **agent_enc}


class SMARTMapDecoder(nn.Module):

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        pl2pl_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        map_token,
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if input_dim == 2:
            input_dim_r_pt2pt = 3
        elif input_dim == 3:
            input_dim_r_pt2pt = 4
        else:
            raise ValueError("{} is not a valid dimension".format(input_dim))

        self.type_pt_emb = nn.Embedding(20, hidden_dim)
        # self.side_pt_emb = nn.Embedding(4, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)
        self.light_pl_emb = nn.Embedding(4, hidden_dim)

        self.r_pt2pt_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pt,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.pt2pt_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_size = 1024
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.token_size
        )
        input_dim_token = 22
        self.token_emb = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.map_token = map_token
        self.apply(weight_init)
        self.mask_pt = False

    def maybe_autocast(self, dtype=torch.float32):
        return torch.cuda.amp.autocast(dtype=dtype)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pt_valid_mask = data["pt_token"]["pt_valid_mask"]
        pt_pred_mask = data["pt_token"]["pt_pred_mask"]
        pt_target_mask = data["pt_token"]["pt_target_mask"]
        mask_s = pt_valid_mask

        pos_pt = data["pt_token"]["position"][:, : self.input_dim].contiguous()
        orient_pt = data["pt_token"]["orientation"].contiguous()
        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)
        token_sample_pt = self.map_token["traj_src"].to(pos_pt.device).to(torch.float)
        pt_token_emb_src = self.token_emb(
            token_sample_pt.view(token_sample_pt.shape[0], -1)
        )
        pt_token_emb = pt_token_emb_src[data["pt_token"]["token_idx"]]

        if self.input_dim == 2:
            x_pt = pt_token_emb
        elif self.input_dim == 3:
            x_pt = pt_token_emb
        else:
            raise ValueError("{} is not a valid dimension".format(self.input_dim))

        token2pl = data[("pt_token", "to", "map_polygon")]["edge_index"]
        token_light_type = data["map_polygon"]["light_type"][token2pl[1]]
        # x_pt_categorical_embs = [self.polygon_type_emb(data['pt_token']['pl_type'].long())]
        x_pt_categorical_embs = [
            self.type_pt_emb(data["pt_token"]["type"].long()),
            self.polygon_type_emb(data["pt_token"]["pl_type"].long()),
            self.light_pl_emb(token_light_type.long()),
        ]
        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)
        edge_index_pt2pt = radius_graph(
            x=pos_pt[:, :2],
            r=self.pl2pl_radius,
            batch=data["pt_token"]["batch"] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=100,
        )
        if self.mask_pt:
            edge_index_pt2pt = subgraph(subset=mask_s, edge_index=edge_index_pt2pt)[0]
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        rel_orient_pt2pt = wrap_angle(
            orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]]
        )
        if self.input_dim == 2:
            r_pt2pt = torch.stack(
                [
                    torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                        nbr_vector=rel_pos_pt2pt[:, :2],
                    ),
                    rel_orient_pt2pt,
                ],
                dim=-1,
            )
        elif self.input_dim == 3:
            r_pt2pt = torch.stack(
                [
                    torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                        nbr_vector=rel_pos_pt2pt[:, :2],
                    ),
                    rel_pos_pt2pt[:, -1],
                    rel_orient_pt2pt,
                ],
                dim=-1,
            )
        else:
            raise ValueError("{} is not a valid dimension".format(self.input_dim))
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        next_token_prob = self.token_predict_head(x_pt[pt_pred_mask])
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)
        next_token_index_gt = data["pt_token"]["token_idx"][pt_target_mask]

        return {
            "x_pt": x_pt,
            "map_next_token_idx": next_token_idx,
            "map_next_token_prob": next_token_prob,
            "map_next_token_idx_gt": next_token_index_gt,
            "map_next_token_eval_mask": pt_pred_mask[pt_pred_mask],
        }


def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_front_y = y + 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_front = (left_front_x, left_front_y)

    right_front_x = x + 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_front_y = y + 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_front = (right_front_x, right_front_y)

    right_back_x = x - 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_back_y = y - 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_back = (right_back_x, right_back_y)

    left_back_x = x - 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_back_y = y - 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_back = (left_back_x, left_back_y)
    polygon_contour = [left_front, right_front, right_back, left_back]

    return polygon_contour


class SMARTAgentDecoder(nn.Module):

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        token_data: Dict,
        token_size=512,
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(4, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.token_emb_veh = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_ped = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_cyc = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.fusion_emb = MLPEmbedding(
            input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
        )

        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_size = token_size
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.token_size
        )
        self.trajectory_token = token_data["token"]
        self.trajectory_token_traj = token_data["traj"]
        self.trajectory_token_all = token_data["token_all"]
        self.apply(weight_init)
        self.shift = 5
        self.beam_size = 100
        self.hist_mask = True

    def transform_rel(self, token_traj, prev_pos, prev_heading=None):
        if prev_heading is None:
            diff_xy = prev_pos[:, :, -1, :] - prev_pos[:, :, -2, :]
            prev_heading = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

        num_agent, num_step, traj_num, traj_dim = token_traj.shape
        cos, sin = prev_heading.cos(), prev_heading.sin()
        rot_mat = torch.zeros((num_agent, num_step, 2, 2), device=prev_heading.device)
        rot_mat[:, :, 0, 0] = cos
        rot_mat[:, :, 0, 1] = -sin
        rot_mat[:, :, 1, 0] = sin
        rot_mat[:, :, 1, 1] = cos
        agent_diff_rel = torch.bmm(
            token_traj.view(-1, traj_num, 2), rot_mat.view(-1, 2, 2)
        ).view(num_agent, num_step, traj_num, traj_dim)
        agent_pred_rel = agent_diff_rel + prev_pos[:, :, -1:, :]
        return agent_pred_rel

    def agent_token_embedding(
        self,
        data,
        agent_category,
        agent_token_index,
        pos_a,
        head_vector_a,
        inference=False,
    ):
        num_agent, num_step, traj_dim = pos_a.shape
        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(data["agent"]["num_nodes"], 1, self.input_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )

        agent_type = data["agent"]["type"]
        veh_mask = agent_type == 0  # * agent_category==3
        cyc_mask = agent_type == 2  # * agent_category==3
        ped_mask = agent_type == 1  # * agent_category==3
        trajectory_token_veh = (
            torch.from_numpy(self.trajectory_token["veh"])
            .clone()
            .to(pos_a.device)
            .to(torch.float)
        )
        self.agent_token_emb_veh = self.token_emb_veh(
            trajectory_token_veh.view(trajectory_token_veh.shape[0], -1)
        )
        trajectory_token_ped = (
            torch.from_numpy(self.trajectory_token["ped"])
            .clone()
            .to(pos_a.device)
            .to(torch.float)
        )
        self.agent_token_emb_ped = self.token_emb_ped(
            trajectory_token_ped.view(trajectory_token_ped.shape[0], -1)
        )
        trajectory_token_cyc = (
            torch.from_numpy(self.trajectory_token["cyc"])
            .clone()
            .to(pos_a.device)
            .to(torch.float)
        )
        self.agent_token_emb_cyc = self.token_emb_cyc(
            trajectory_token_cyc.view(trajectory_token_cyc.shape[0], -1)
        )

        if inference:
            agent_token_traj_all = torch.zeros(
                (num_agent, self.token_size, self.shift + 1, 4, 2), device=pos_a.device
            )
            trajectory_token_all_veh = (
                torch.from_numpy(self.trajectory_token_all["veh"])
                .clone()
                .to(pos_a.device)
                .to(torch.float)
            )
            trajectory_token_all_ped = (
                torch.from_numpy(self.trajectory_token_all["ped"])
                .clone()
                .to(pos_a.device)
                .to(torch.float)
            )
            trajectory_token_all_cyc = (
                torch.from_numpy(self.trajectory_token_all["cyc"])
                .clone()
                .to(pos_a.device)
                .to(torch.float)
            )
            agent_token_traj_all[veh_mask] = torch.cat(
                [
                    trajectory_token_all_veh[:, : self.shift],
                    trajectory_token_veh[:, None, ...],
                ],
                dim=1,
            )
            agent_token_traj_all[ped_mask] = torch.cat(
                [
                    trajectory_token_all_ped[:, : self.shift],
                    trajectory_token_ped[:, None, ...],
                ],
                dim=1,
            )
            agent_token_traj_all[cyc_mask] = torch.cat(
                [
                    trajectory_token_all_cyc[:, : self.shift],
                    trajectory_token_cyc[:, None, ...],
                ],
                dim=1,
            )

        agent_token_emb = torch.zeros(
            (num_agent, num_step, self.hidden_dim), device=pos_a.device
        )
        agent_token_emb[veh_mask] = self.agent_token_emb_veh[
            agent_token_index[veh_mask]
        ]
        agent_token_emb[ped_mask] = self.agent_token_emb_ped[
            agent_token_index[ped_mask]
        ]
        agent_token_emb[cyc_mask] = self.agent_token_emb_cyc[
            agent_token_index[cyc_mask]
        ]

        agent_token_traj = torch.zeros(
            (num_agent, num_step, self.token_size, 4, 2), device=pos_a.device
        )
        agent_token_traj[veh_mask] = trajectory_token_veh
        agent_token_traj[ped_mask] = trajectory_token_ped
        agent_token_traj[cyc_mask] = trajectory_token_cyc

        vel = data["agent"]["token_velocity"]

        categorical_embs = [
            self.type_a_emb(data["agent"]["type"].long()).repeat_interleave(
                repeats=num_step, dim=0
            ),
            self.shape_emb(
                data["agent"]["shape"][:, self.num_historical_steps - 1, :]
            ).repeat_interleave(repeats=num_step, dim=0),
        ]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=categorical_embs,
        )
        x_a = x_a.view(-1, num_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (
                feat_a,
                agent_token_traj,
                agent_token_traj_all,
                agent_token_emb,
                categorical_embs,
            )
        else:
            return feat_a, agent_token_traj

    # def agent_predict_next(self, data, agent_category, feat_a):
    #     num_agent, num_step, traj_dim = data['agent']['token_pos'].shape
    #     agent_type = data['agent']['type']
    #     veh_mask = (agent_type == 0)  # * agent_category==3
    #     cyc_mask = (agent_type == 2)  # * agent_category==3
    #     ped_mask = (agent_type == 1)  # * agent_category==3
    #     token_res = torch.zeros((num_agent, num_step, self.token_size), device=agent_category.device)
    #     token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
    #     token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
    #     token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])
    #     return token_res

    # def agent_predict_next_inf(self, data, agent_category, feat_a):
    #     num_agent, traj_dim = feat_a.shape
    #     agent_type = data['agent']['type']

    #     veh_mask = (agent_type == 0)  # * agent_category==3
    #     cyc_mask = (agent_type == 2)  # * agent_category==3
    #     ped_mask = (agent_type == 1)  # * agent_category==3

    #     token_res = torch.zeros((num_agent, self.token_size), device=agent_category.device)
    #     token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
    #     token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
    #     token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])

    #     return token_res

    def build_temporal_edge(
        self, pos_a, head_a, head_vector_a, num_agent, mask, inference_mask=None
    ):
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        hist_mask = mask.clone()

        if self.hist_mask and self.training:
            hist_mask[
                torch.arange(mask.shape[0]).unsqueeze(1),
                torch.randint(0, mask.shape[1], (num_agent, 10)),
            ] = False
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)
        elif inference_mask is not None:
            mask_t = hist_mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]],
                    nbr_vector=rel_pos_t[:, :2],
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask_s):
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self,
        data,
        num_step,
        agent_category,
        pos_a,
        head_a,
        head_vector_a,
        mask,
        batch_s,
        batch_pl,
    ):
        mask_pl2a = mask.clone()
        mask_pl2a = mask_pl2a.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = data["pt_token"]["position"][:, : self.input_dim].contiguous()
        orient_pl = data["pt_token"]["orientation"].contiguous()
        pos_pl = pos_pl.repeat(num_step, 1)
        orient_pl = orient_pl.repeat(num_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def check_memory(self):
        num_gpus = torch.cuda.device_count()

        for i in range(num_gpus):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # 单位MB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2  # 单位MB
            memory_usage = psutil.virtual_memory().used / 1024**2  # 系统内存

            print(f"GPU {i}:")
            print(
                f"  CUDA Memory Allocated: {memory_allocated:.2f} MB, {memory_allocated / 1024:.2f} GB"
            )
            print(
                f"  CUDA Memory Reserved: {memory_reserved:.2f} MB, {memory_reserved / 1024:.2f} GB"
            )
            print(
                f"  System Memory Usage: {memory_usage:.2f} MB, {memory_usage / 1024:.2f} GB"
            )

    def forward(
        self, data: HeteroData, map_enc: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pos_a = data["agent"]["token_pos"]
        head_a = data["agent"]["token_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        num_agent, num_step, traj_dim = pos_a.shape
        agent_category = data["agent"]["category"]
        agent_token_index = data["agent"]["token_idx"]
        feat_a, agent_token_traj = self.agent_token_embedding(
            data, agent_category, agent_token_index, pos_a, head_vector_a
        )

        agent_valid_mask = data["agent"]["agent_valid_mask"].clone()
        # eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
        # agent_valid_mask[~eval_mask] = False
        mask = agent_valid_mask
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a, head_a, head_vector_a, num_agent, mask
        )

        if isinstance(data, Batch):
            batch_s = torch.cat(
                [data["agent"]["batch"] + data.num_graphs * t for t in range(num_step)],
                dim=0,
            )
            batch_pl = torch.cat(
                [
                    data["pt_token"]["batch"] + data.num_graphs * t
                    for t in range(num_step)
                ],
                dim=0,
            )
        else:
            batch_s = torch.arange(num_step, device=pos_a.device).repeat_interleave(
                data["agent"]["num_nodes"]
            )
            batch_pl = torch.arange(num_step, device=pos_a.device).repeat_interleave(
                data["pt_token"]["num_nodes"]
            )

        mask_s = mask.transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a, head_a, head_vector_a, batch_s, mask_s
        )
        mask[agent_category != 3] = False
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            data,
            num_step,
            agent_category,
            pos_a,
            head_a,
            head_vector_a,
            mask,
            batch_s,
            batch_pl,
        )

        for i in range(self.num_layers):
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = (
                feat_a.reshape(-1, num_step, self.hidden_dim)
                .transpose(0, 1)
                .reshape(-1, self.hidden_dim)
            )
            feat_a = self.pt2a_attn_layers[i](
                (
                    map_enc["x_pt"]
                    .repeat_interleave(repeats=num_step, dim=0)
                    .reshape(-1, num_step, self.hidden_dim)
                    .transpose(0, 1)
                    .reshape(-1, self.hidden_dim),
                    feat_a,
                ),
                r_pl2a,
                edge_index_pl2a,
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

        num_agent, num_step, hidden_dim, traj_num, traj_dim = agent_token_traj.shape
        next_token_prob = self.token_predict_head(feat_a)
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)

        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)
        next_token_eval_mask = mask.clone()
        next_token_eval_mask = (
            next_token_eval_mask
            * next_token_eval_mask.roll(shifts=-1, dims=1)
            * next_token_eval_mask.roll(shifts=1, dims=1)
        )
        next_token_eval_mask[:, -1] = False

        return {
            "x_a": feat_a,
            "next_token_idx": next_token_idx,
            "next_token_prob": next_token_prob,
            "next_token_idx_gt": next_token_index_gt,
            "next_token_eval_mask": next_token_eval_mask,
        }

    def inference(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        ss_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        eval_mask = data["agent"]["valid_mask"][:, self.num_historical_steps - 1]
        pos_a = data["agent"]["token_pos"].clone()
        head_a = data["agent"]["token_heading"].clone()
        num_agent, num_step, traj_dim = pos_a.shape
        pos_a[:, (self.num_historical_steps - 1) // self.shift :] = 0
        head_a[:, (self.num_historical_steps - 1) // self.shift :] = 0
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        agent_valid_mask = data["agent"]["agent_valid_mask"].clone()
        agent_valid_mask[:, (self.num_historical_steps - 1) // self.shift :] = True
        agent_valid_mask[~eval_mask] = False
        agent_token_index = data["agent"]["token_idx"]
        agent_category = data["agent"]["category"]
        (
            feat_a,
            agent_token_traj,
            agent_token_traj_all,
            agent_token_emb,
            categorical_embs,
        ) = self.agent_token_embedding(
            data,
            agent_category,
            agent_token_index,
            pos_a,
            head_vector_a,
            inference=True,
        )

        agent_type = data["agent"]["type"]
        veh_mask = agent_type == 0  # * agent_category==3
        cyc_mask = agent_type == 2  # * agent_category==3
        ped_mask = agent_type == 1  # * agent_category==3
        av_mask = data["agent"]["av_index"]

        self.num_recurrent_steps_val = (
            data["agent"]["position"].shape[1] - self.num_historical_steps
        )
        pred_traj = torch.zeros(
            data["agent"]["num_nodes"],
            self.num_recurrent_steps_val,
            2,
            device=feat_a.device,
        )
        pred_head = torch.zeros(
            data["agent"]["num_nodes"],
            self.num_recurrent_steps_val,
            device=feat_a.device,
        )
        pred_prob = torch.zeros(
            data["agent"]["num_nodes"],
            self.num_recurrent_steps_val // self.shift,
            device=feat_a.device,
        )
        next_token_idx_list = []
        mask = agent_valid_mask.clone()
        feat_a_t_dict = {}
        for t in range(self.num_recurrent_steps_val // self.shift):
            if t == 0:
                inference_mask = mask.clone()
                inference_mask[
                    :, (self.num_historical_steps - 1) // self.shift + t :
                ] = False
            else:
                inference_mask = torch.zeros_like(mask)
                inference_mask[
                    :, (self.num_historical_steps - 1) // self.shift + t - 1
                ] = True
            edge_index_t, r_t = self.build_temporal_edge(
                pos_a, head_a, head_vector_a, num_agent, mask, inference_mask
            )
            if isinstance(data, Batch):
                batch_s = torch.cat(
                    [
                        data["agent"]["batch"] + data.num_graphs * t
                        for t in range(num_step)
                    ],
                    dim=0,
                )
                batch_pl = torch.cat(
                    [
                        data["pt_token"]["batch"] + data.num_graphs * t
                        for t in range(num_step)
                    ],
                    dim=0,
                )
            else:
                batch_s = torch.arange(num_step, device=pos_a.device).repeat_interleave(
                    data["agent"]["num_nodes"]
                )
                batch_pl = torch.arange(
                    num_step, device=pos_a.device
                ).repeat_interleave(data["pt_token"]["num_nodes"])
            # In the inference stage, we only infer the current stage for recurrent
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                data,
                num_step,
                agent_category,
                pos_a,
                head_a,
                head_vector_a,
                inference_mask,
                batch_s,
                batch_pl,
            )
            mask_s = inference_mask.transpose(0, 1).reshape(-1)
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a, head_a, head_vector_a, batch_s, mask_s
            )

            for i in range(self.num_layers):
                if i in feat_a_t_dict:
                    feat_a = feat_a_t_dict[i]
                feat_a = feat_a.reshape(-1, self.hidden_dim)
                feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
                feat_a = (
                    feat_a.reshape(-1, num_step, self.hidden_dim)
                    .transpose(0, 1)
                    .reshape(-1, self.hidden_dim)
                )
                feat_a = self.pt2a_attn_layers[i](
                    (
                        map_enc["x_pt"]
                        .repeat_interleave(repeats=num_step, dim=0)
                        .reshape(-1, num_step, self.hidden_dim)
                        .transpose(0, 1)
                        .reshape(-1, self.hidden_dim),
                        feat_a,
                    ),
                    r_pl2a,
                    edge_index_pl2a,
                )
                feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
                feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

                if i + 1 not in feat_a_t_dict:
                    feat_a_t_dict[i + 1] = feat_a
                else:
                    feat_a_t_dict[i + 1][
                        :, (self.num_historical_steps - 1) // self.shift - 1 + t
                    ] = feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]

            next_token_prob = self.token_predict_head(
                feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]
            )

            next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)

            topk_prob, next_token_idx = torch.topk(
                next_token_prob_softmax, k=self.beam_size, dim=-1
            )

            expanded_index = next_token_idx[..., None, None, None].expand(
                -1, -1, 6, 4, 2
            )
            next_token_traj = torch.gather(agent_token_traj_all, 1, expanded_index)

            theta = head_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = torch.zeros((num_agent, 2, 2), device=theta.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_diff_rel = torch.bmm(
                next_token_traj.view(-1, 4, 2),
                rot_mat[:, None, None, ...]
                .repeat(1, self.beam_size, self.shift + 1, 1, 1)
                .view(-1, 2, 2),
            ).view(num_agent, self.beam_size, self.shift + 1, 4, 2)
            agent_pred_rel = (
                agent_diff_rel
                + pos_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t, :][
                    :, None, None, None, ...
                ]
            )

            sample_index = torch.multinomial(topk_prob, 1).to(agent_pred_rel.device)
            agent_pred_rel = agent_pred_rel.gather(
                dim=1, index=sample_index[..., None, None, None].expand(-1, -1, 6, 4, 2)
            )[:, 0, ...]
            pred_prob[:, t] = topk_prob.gather(dim=-1, index=sample_index)[:, 0]
            pred_traj[:, t * 5 : (t + 1) * 5] = (
                agent_pred_rel[:, 1:, ...].clone().mean(dim=2)
            )
            diff_xy = agent_pred_rel[:, 1:, 0, :] - agent_pred_rel[:, 1:, 3, :]
            pred_head[:, t * 5 : (t + 1) * 5] = torch.arctan2(
                diff_xy[:, :, 1], diff_xy[:, :, 0]
            )

            pos_a[:, (self.num_historical_steps - 1) // self.shift + t] = (
                agent_pred_rel[:, -1, ...].clone().mean(dim=1)
            )
            diff_xy = agent_pred_rel[:, -1, 0, :] - agent_pred_rel[:, -1, 3, :]
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            head_a[:, (self.num_historical_steps - 1) // self.shift + t] = theta
            next_token_idx = next_token_idx.gather(dim=1, index=sample_index)
            next_token_idx = next_token_idx.squeeze(-1)
            next_token_idx_list.append(next_token_idx[:, None])
            agent_token_emb[
                veh_mask, (self.num_historical_steps - 1) // self.shift + t
            ] = self.agent_token_emb_veh[next_token_idx[veh_mask]]
            agent_token_emb[
                ped_mask, (self.num_historical_steps - 1) // self.shift + t
            ] = self.agent_token_emb_ped[next_token_idx[ped_mask]]
            agent_token_emb[
                cyc_mask, (self.num_historical_steps - 1) // self.shift + t
            ] = self.agent_token_emb_cyc[next_token_idx[cyc_mask]]
            motion_vector_a = torch.cat(
                [
                    pos_a.new_zeros(data["agent"]["num_nodes"], 1, self.input_dim),
                    pos_a[:, 1:] - pos_a[:, :-1],
                ],
                dim=1,
            )

            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

            vel = motion_vector_a.clone() / (0.1 * self.shift)
            vel[:, (self.num_historical_steps - 1) // self.shift + 1 + t :] = 0
            motion_vector_a[
                :, (self.num_historical_steps - 1) // self.shift + 1 + t :
            ] = 0
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                    ),
                ],
                dim=-1,
            )

            x_a = self.x_a_emb(
                continuous_inputs=x_a.view(-1, x_a.size(-1)),
                categorical_embs=categorical_embs,
            )
            x_a = x_a.view(-1, num_step, self.hidden_dim)

            feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
            feat_a = self.fusion_emb(feat_a)

        agent_valid_mask[agent_category != 3] = False

        return {
            "pos_a": pos_a[:, (self.num_historical_steps - 1) // self.shift :],
            "head_a": head_a[:, (self.num_historical_steps - 1) // self.shift :],
            "gt": data["agent"]["position"][
                :, self.num_historical_steps :, : self.input_dim
            ].contiguous(),
            "valid_mask": agent_valid_mask[:, self.num_historical_steps :],
            "pred_traj": pred_traj,
            "pred_head": pred_head,
            "next_token_idx": torch.cat(next_token_idx_list, dim=-1),
            "next_token_idx_gt": agent_token_index.roll(shifts=-1, dims=1),
            "next_token_eval_mask": data["agent"]["agent_valid_mask"],
            "pred_prob": pred_prob,
            "vel": vel,
        }

    def save_data_debug(
        self,
        data,
        data_filename="data.pkl",
        save_dir="/home/zxq/ws/SMART/waymax_submission/debug",
    ):
        save_dir = os.path.join(save_dir, f"{data['scenario_id'][0]}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_path = os.path.join(save_dir, data_filename)

        with open(data_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Data saved to {data_path}")
