import math
import logging
import time
import os
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import KDTree, ConvexHull
from typing import List, Optional, Tuple, Union, Any
from torch_geometric.utils import coalesce
from torch_geometric.utils import degree


def check_nan_inf(t, s):
    assert not torch.isinf(t).any(), f"{s} is inf, {t}"
    assert not torch.isnan(t).any(), f"{s} is nan, {t}"


def angle_between_2d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1]
        - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def angle_between_3d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    return torch.atan2(
        torch.cross(ctr_vector, nbr_vector, dim=-1).norm(p=2, dim=-1),
        (ctr_vector * nbr_vector).sum(dim=-1),
    )


def side_to_directed_lineseg(
    query_point: torch.Tensor, start_point: torch.Tensor, end_point: torch.Tensor
) -> str:
    cond = (end_point[0] - start_point[0]) * (query_point[1] - start_point[1]) - (
        end_point[1] - start_point[1]
    ) * (query_point[0] - start_point[0])
    if cond > 0:
        return "LEFT"
    elif cond < 0:
        return "RIGHT"
    else:
        return "CENTER"


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def get_drivable_area_tree(polylines):
    all_points = np.concatenate(polylines, axis=0)
    all_points = all_points[~np.all(all_points == 0, axis=1)][:, :2]
    all_points = np.unique(all_points, axis=0)
    tree = KDTree(all_points)
    return tree, all_points


def get_drivable_area_convex(polylines):
    all_points = np.concatenate(polylines, axis=0)  # (N*num_points, 2)

    mask_valid = ~np.all(all_points == 0, axis=1)
    all_points = all_points[mask_valid]
    all_points = np.unique(all_points, axis=0)

    if all_points.shape[1] > 2:
        all_points = all_points[:, :2]

    if all_points.shape[0] < 3:
        raise ValueError("Not enough valid points to compute ConvexHull")

    hull = ConvexHull(all_points)
    drivable_area = all_points[hull.vertices]

    return drivable_area


def add_edges(
    from_edge_index: torch.Tensor,
    to_edge_index: torch.Tensor,
    from_edge_attr: Optional[torch.Tensor] = None,
    to_edge_attr: Optional[torch.Tensor] = None,
    replace: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    from_edge_index = from_edge_index.to(
        device=to_edge_index.device, dtype=to_edge_index.dtype
    )
    mask = (to_edge_index[0].unsqueeze(-1) == from_edge_index[0].unsqueeze(0)) & (
        to_edge_index[1].unsqueeze(-1) == from_edge_index[1].unsqueeze(0)
    )
    if replace:
        to_mask = mask.any(dim=1)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(
                device=to_edge_attr.device, dtype=to_edge_attr.dtype
            )
            to_edge_attr = torch.cat([to_edge_attr[~to_mask], from_edge_attr], dim=0)
        to_edge_index = torch.cat([to_edge_index[:, ~to_mask], from_edge_index], dim=1)
    else:
        from_mask = mask.any(dim=0)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(
                device=to_edge_attr.device, dtype=to_edge_attr.dtype
            )
            to_edge_attr = torch.cat([to_edge_attr, from_edge_attr[~from_mask]], dim=0)
        to_edge_index = torch.cat(
            [to_edge_index, from_edge_index[:, ~from_mask]], dim=1
        )
    return to_edge_index, to_edge_attr


def merge_edges(
    edge_indices: List[torch.Tensor],
    edge_attrs: Optional[List[torch.Tensor]] = None,
    reduce: str = "add",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    edge_index = torch.cat(edge_indices, dim=1)
    if edge_attrs is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_attr = None
    return coalesce(edge_index=edge_index, edge_attr=edge_attr, reduce=reduce)


def complete_graph(
    num_nodes: Union[int, Tuple[int, int]],
    ptr: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    loop: bool = False,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    if ptr is None:
        if isinstance(num_nodes, int):
            num_src, num_dst = num_nodes, num_nodes
        else:
            num_src, num_dst = num_nodes
        edge_index = torch.cartesian_prod(
            torch.arange(num_src, dtype=torch.long, device=device),
            torch.arange(num_dst, dtype=torch.long, device=device),
        ).t()
    else:
        if isinstance(ptr, torch.Tensor):
            ptr_src, ptr_dst = ptr, ptr
            num_src_batch = num_dst_batch = ptr[1:] - ptr[:-1]
        else:
            ptr_src, ptr_dst = ptr
            num_src_batch = ptr_src[1:] - ptr_src[:-1]
            num_dst_batch = ptr_dst[1:] - ptr_dst[:-1]
        edge_index = torch.cat(
            [
                torch.cartesian_prod(
                    torch.arange(num_src, dtype=torch.long, device=device),
                    torch.arange(num_dst, dtype=torch.long, device=device),
                )
                + p
                for num_src, num_dst, p in zip(
                    num_src_batch, num_dst_batch, torch.stack([ptr_src, ptr_dst], dim=1)
                )
            ],
            dim=0,
        )
        edge_index = edge_index.t()
    if isinstance(num_nodes, int) and not loop:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index.contiguous()


def bipartite_dense_to_sparse(adj: torch.Tensor) -> torch.Tensor:
    index = adj.nonzero(as_tuple=True)
    if len(index) == 3:
        batch_src = index[0] * adj.size(1)
        batch_dst = index[0] * adj.size(2)
        index = (batch_src + index[1], batch_dst + index[2])
    return torch.stack(index, dim=0)


def unbatch(src: torch.Tensor, batch: torch.Tensor, dim: int = 0) -> List[torch.Tensor]:
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif "weight_hr" in name:
                nn.init.xavier_uniform_(param)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)


class Logging:

    def make_log_dir(self, dirname="logs"):
        now_dir = os.path.dirname(__file__)
        path = os.path.join(now_dir, dirname)
        path = os.path.normpath(path)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_log_filename(self):
        filename = "{}.log".format(time.strftime("%Y-%m-%d", time.localtime()))
        filename = os.path.join(self.make_log_dir(), filename)
        filename = os.path.normpath(filename)
        return filename

    def log(self, level="DEBUG", name="simagent"):
        logger = logging.getLogger(name)
        level = getattr(logging, level)
        logger.setLevel(level)
        if not logger.handlers:
            sh = logging.StreamHandler()
            fh = logging.FileHandler(
                filename=self.get_log_filename(), mode="a", encoding="utf-8"
            )
            fmt = logging.Formatter(
                "%(asctime)s-%(levelname)s-%(filename)s-Line:%(lineno)d-Message:%(message)s"
            )
            sh.setFormatter(fmt=fmt)
            fh.setFormatter(fmt=fmt)
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger

    def add_log(self, logger, level="DEBUG"):
        level = getattr(logging, level)
        logger.setLevel(level)
        if not logger.handlers:
            sh = logging.StreamHandler()
            fh = logging.FileHandler(
                filename=self.get_log_filename(), mode="a", encoding="utf-8"
            )
            fmt = logging.Formatter(
                "%(asctime)s-%(levelname)s-%(filename)s-Line:%(lineno)d-Message:%(message)s"
            )
            sh.setFormatter(fmt=fmt)
            fh.setFormatter(fmt=fmt)
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger
