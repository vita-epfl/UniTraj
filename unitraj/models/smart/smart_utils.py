import math
import logging
import time
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import KDTree, ConvexHull
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
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

def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.column_stack((left_front_x, left_front_y))

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.column_stack((right_front_x, right_front_y))

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.column_stack((right_back_x, right_back_y))

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.column_stack((left_back_x, left_back_y))

    polygon_contour = np.concatenate(
        (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1)

    return polygon_contour

def interplating_polyline(polylines, heading, distance=0.5, split_distace=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    dist_along_path_list = [[0]]
    polylines_list = [[polylines[0]]]
    for i in range(1, polylines.shape[0]):
        euclidean_dist = euclidean(polylines[i, :2], polylines[i - 1, :2])
        heading_diff = min(abs(max(heading[i], heading[i - 1]) - min(heading[i], heading[i - 1])),
                           abs(max(heading[i], heading[i - 1]) - min(heading[i], heading[i - 1]) + math.pi))
        if heading_diff > math.pi / 4 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > math.pi / 8 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > 0.1 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif euclidean_dist > 10:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        else:
            dist_along_path_list[-1].append(dist_along_path_list[-1][-1] + euclidean_dist)
            polylines_list[-1].append(polylines[i])
    # plt.plot(polylines[:, 0], polylines[:, 1])
    # plt.savefig('tmp.jpg')
    new_x_list = []
    new_y_list = []
    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = np.array(dist_along_path_list[idx])
        polylines_cur = np.array(polylines_list[idx])
        # Create interpolation functions for x and y coordinates
        fx = interp1d(dist_along_path, polylines_cur[:, 0])
        fy = interp1d(dist_along_path, polylines_cur[:, 1])
        # fyaw = interp1d(dist_along_path, heading)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate([new_dist_along_path, dist_along_path[[-1]]])
        # Use the interpolation functions to generate new x and y coordinates
        new_x = fx(new_dist_along_path)
        new_y = fy(new_dist_along_path)
        # new_yaw = fyaw(new_dist_along_path)
        new_x_list.append(new_x)
        new_y_list.append(new_y)

        # Combine the new x and y coordinates into a single array
        new_polylines = np.vstack((new_x, new_y)).T
        polyline_size = int(split_distace / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (new_polylines.shape[0] - (polyline_size + 1)) % polyline_size
            final_index = (new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(new_polylines[1:, 1] - new_polylines[:-1, 1],
                                  new_polylines[1:, 0] - new_polylines[:-1, 0])
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(dimension=0, size=polyline_size + 1, step=polyline_size)
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size:]
            last_polyline = last_polyline[torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()]
            if multi_polylines is not None:
                multi_polylines = torch.cat([multi_polylines, last_polyline.unsqueeze(0)], dim=0)
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0)
    else:
        multi_polylines_list = None
    return multi_polylines_list

class TokenProcessor:

    def __init__(self, token_size):
        module_dir = os.path.dirname(os.path.dirname(__file__))
        self.agent_token_path = os.path.join(module_dir, f'smart/tokens/cluster_frame_5_{token_size}.pkl')
        self.map_token_traj_path = os.path.join(module_dir, 'smart/tokens/map_traj_token5.pkl')
        self.noise = False
        self.disturb = False
        self.shift = 5
        self.get_trajectory_token()
        self.training = False
        self.current_step = 10

    def preprocess(self, data):
        data = self.tokenize_agent(data)
        data = self.tokenize_map(data)
        # del data['city']
        if 'polygon_is_intersection' in data['map_polygon']:
            del data['map_polygon']['polygon_is_intersection']
        if 'route_type' in data['map_polygon']:
            del data['map_polygon']['route_type']
        return self.convert_all_numpy_to_tensor(data)

    def convert_all_numpy_to_tensor(self, data, dtype=None):
        if isinstance(data, dict):
            return {k: self.convert_all_numpy_to_tensor(v, dtype) for k, v in data.items()}

        elif isinstance(data, list):
            return [self.convert_all_numpy_to_tensor(item, dtype) for item in data]
        
        elif isinstance(data, tuple):
            return tuple(self.convert_all_numpy_to_tensor(item, dtype) for item in data)
        
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
            return tensor.to(dtype) if dtype else tensor

        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return torch.tensor(data)
        
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return torch.tensor(data, dtype=torch.float32)
        
        elif isinstance(data, np.str_):
            return str(data)
        
        else:
            return data

    def get_trajectory_token(self):
        agent_token_data = pickle.load(open(self.agent_token_path, 'rb'))
        map_token_traj = pickle.load(open(self.map_token_traj_path, 'rb'))
        self.trajectory_token = agent_token_data['token']
        self.trajectory_token_all = agent_token_data['token_all']
        self.map_token = {'traj_src': map_token_traj['traj_src'], }
        self.token_last = {}
        for k, v in self.trajectory_token_all.items():
            token_last = torch.from_numpy(v[:, -2:]).to(torch.float)
            diff_xy = token_last[:, 0, 0] - token_last[:, 0, 3]
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(token_last.shape[0], 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = -sin
            rot_mat[:, 1, 0] = sin
            rot_mat[:, 1, 1] = cos
            agent_token = torch.bmm(token_last[:, 1], rot_mat)
            agent_token -= token_last[:, 0].mean(1)[:, None, :]
            self.token_last[k] = agent_token.numpy()

    def clean_heading(self, data):
        heading = data['agent']['heading']
        valid = data['agent']['valid_mask']
        pi = np.array([np.pi])
        n_vehicles, n_frames = heading.shape

        heading_diff_raw = heading[:, :-1] - heading[:, 1:]
        heading_diff = np.remainder(heading_diff_raw + pi, 2 * pi) - pi
        heading_diff[heading_diff > pi] -= 2 * pi
        heading_diff[heading_diff < -pi] += 2 * pi

        valid_pairs = valid[:, :-1] & valid[:, 1:]

        for i in range(n_frames - 1):
            change_needed = (np.abs(heading_diff[:, i:i + 1]) > 1.0) & valid_pairs[:, i:i + 1]

            heading[:, i + 1][change_needed.squeeze()] = heading[:, i][change_needed.squeeze()]

            if i < n_frames - 2:
                heading_diff_raw = heading[:, i + 1] - heading[:, i + 2]
                heading_diff[:, i + 1] = np.remainder(heading_diff_raw + pi, 2 * pi) - pi
                heading_diff[heading_diff[:, i + 1] > pi] -= 2 * pi
                heading_diff[heading_diff[:, i + 1] < -pi] += 2 * pi

    def tokenize_agent(self, data):
        if data['agent']["velocity"].shape[1] == 90:
            print(data['scenario_id'], data['agent']["velocity"].shape)
        interplote_mask = (data['agent']['valid_mask'][:, self.current_step] == False) * (
                data['agent']['position'][:, self.current_step, 0] != 0)
        if data['agent']["velocity"].shape[-1] == 2:
            data['agent']["velocity"] = torch.cat([data['agent']["velocity"],
                                                   torch.zeros(data['agent']["velocity"].shape[0],
                                                               data['agent']["velocity"].shape[1], 1)], dim=-1)
        vel = data['agent']["velocity"][interplote_mask, self.current_step]
        data['agent']['position'][interplote_mask, self.current_step - 1, :3] = data['agent']['position'][
                                                                                interplote_mask, self.current_step,
                                                                                :3] - vel * 0.1
        data['agent']['valid_mask'][interplote_mask, self.current_step - 1:self.current_step + 1] = True
        data['agent']['heading'][interplote_mask, self.current_step - 1] = data['agent']['heading'][
            interplote_mask, self.current_step]
        data['agent']["velocity"][interplote_mask, self.current_step - 1] = data['agent']["velocity"][
            interplote_mask, self.current_step]

        data['agent']['type'] = data['agent']['type'].astype(np.uint8)

        self.clean_heading(data)
        matching_extra_mask = (data['agent']['valid_mask'][:, self.current_step] == True) * (
                data['agent']['valid_mask'][:, self.current_step - 5] == False)

        interplote_mask_first = (data['agent']['valid_mask'][:, 0] == False) * (data['agent']['position'][:, 0, 0] != 0)
        data['agent']['valid_mask'][interplote_mask_first, 0] = True

        agent_pos = data['agent']['position'][:, :, :2]
        valid_mask = data['agent']['valid_mask']

        valid_mask_shift = sliding_window_view(valid_mask, window_shape=self.shift + 1, axis=1)[:, ::self.shift, :]
        token_valid_mask = valid_mask_shift[:, :, 0] * valid_mask_shift[:, :, -1]
        agent_type = data['agent']['type']
        agent_category = data['agent']['category']
        agent_heading = data['agent']['heading']
        vehicle_mask = agent_type == 0
        cyclist_mask = agent_type == 2
        ped_mask = agent_type == 1

        veh_pos = agent_pos[vehicle_mask, :, :]
        veh_valid_mask = valid_mask[vehicle_mask, :]
        cyc_pos = agent_pos[cyclist_mask, :, :]
        cyc_valid_mask = valid_mask[cyclist_mask, :]
        ped_pos = agent_pos[ped_mask, :, :]
        ped_valid_mask = valid_mask[ped_mask, :]

        veh_token_index, veh_token_contour = self.match_token(veh_pos, veh_valid_mask, agent_heading[vehicle_mask],
                                                              'veh', agent_category[vehicle_mask],
                                                              matching_extra_mask[vehicle_mask])
        ped_token_index, ped_token_contour = self.match_token(ped_pos, ped_valid_mask, agent_heading[ped_mask], 'ped',
                                                              agent_category[ped_mask], matching_extra_mask[ped_mask])
        cyc_token_index, cyc_token_contour = self.match_token(cyc_pos, cyc_valid_mask, agent_heading[cyclist_mask],
                                                              'cyc', agent_category[cyclist_mask],
                                                              matching_extra_mask[cyclist_mask])

        token_index = np.zeros((agent_pos.shape[0], veh_token_index.shape[1]), dtype=np.int64)
        token_index[vehicle_mask] = veh_token_index
        token_index[ped_mask] = ped_token_index
        token_index[cyclist_mask] = cyc_token_index

        token_contour = np.zeros((agent_pos.shape[0], veh_token_contour.shape[1],
                                veh_token_contour.shape[2], veh_token_contour.shape[3]), dtype=np.float32)
        token_contour[vehicle_mask] = veh_token_contour
        token_contour[ped_mask] = ped_token_contour
        token_contour[cyclist_mask] = cyc_token_contour

        trajectory_token_veh = self.trajectory_token['veh'].astype(np.float32).copy()
        trajectory_token_ped = self.trajectory_token['ped'].astype(np.float32).copy()
        trajectory_token_cyc = self.trajectory_token['cyc'].astype(np.float32).copy()

        agent_token_traj = np.zeros((agent_pos.shape[0], trajectory_token_veh.shape[0], 4, 2))
        agent_token_traj[vehicle_mask] = trajectory_token_veh
        agent_token_traj[ped_mask] = trajectory_token_ped
        agent_token_traj[cyclist_mask] = trajectory_token_cyc

        if not self.training:
            token_valid_mask[matching_extra_mask, 1] = True

        data['agent']['token_idx'] = token_index
        data['agent']['token_contour'] = token_contour
        token_pos = token_contour.mean(axis=2)
        data['agent']['token_pos'] = token_pos
        diff_xy = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]
        data['agent']['token_heading'] = np.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
        data['agent']['agent_valid_mask'] = token_valid_mask

        zero_padding = np.zeros((data['agent']['num_nodes'], 1, 2))
        velocity_diff = (token_pos[:, 1:] - token_pos[:, :-1]) / (0.1 * self.shift)
        vel = np.concatenate([zero_padding, velocity_diff], axis=1)
        vel_valid_mask = np.concatenate([
            np.zeros((token_valid_mask.shape[0], 1), dtype=bool),
            (token_valid_mask[:, 1:] * np.roll(token_valid_mask, shift=1, axis=1)[:, 1:])
        ], axis=1)
        vel[~vel_valid_mask] = 0
        vel[data['agent']['valid_mask'][:, self.current_step], 1] = data['agent']['velocity'][
                                                                    data['agent']['valid_mask'][:, self.current_step],
                                                                    self.current_step, :2]

        data['agent']['token_velocity'] = vel

        return data

    def match_token(self, pos, valid_mask, heading, category, agent_category, extra_mask):
        agent_token_src = self.trajectory_token[category]
        token_last = self.token_last[category]
        if self.shift <= 2:
            if category == 'veh':
                width = 1.0
                length = 2.4
            elif category == 'cyc':
                width = 0.5
                length = 1.5
            else:
                width = 0.5
                length = 0.5
        else:
            if category == 'veh':
                width = 2.0
                length = 4.8
            elif category == 'cyc':
                width = 1.0
                length = 2.0
            else:
                width = 1.0
                length = 1.0

        prev_heading = heading[:, 0]
        prev_pos = pos[:, 0]
        agent_num, num_step, feat_dim = pos.shape
        token_num, token_contour_dim, feat_dim = agent_token_src.shape
        agent_token_src = np.repeat(agent_token_src.reshape(1, token_num * token_contour_dim, feat_dim), agent_num, axis=0)
        token_last = np.repeat(token_last.reshape(1, token_num * token_contour_dim, feat_dim), extra_mask.sum(), axis=0)
        token_index_list = []
        token_contour_list = []
        prev_token_idx = None

        for i in range(self.shift, pos.shape[1], self.shift):
            theta = prev_heading
            cur_heading = heading[:, i]
            cur_pos = pos[:, i]
            cos, sin = np.cos(theta), np.sin(theta)
            rot_mat = np.zeros((agent_num, 2, 2))
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = np.matmul(agent_token_src, rot_mat).reshape(agent_num,
                                                                            token_num,
                                                                            token_contour_dim,
                                                                            feat_dim)
            agent_token_world += prev_pos[:, None, None, :]

            cur_contour = cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
            dist = np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world) ** 2, axis=-1)).mean(axis=2)
            agent_token_index = np.argmin(dist, axis=-1)
            if prev_token_idx is not None and self.noise:
                same_idx = prev_token_idx == agent_token_index
                same_idx[:] = True
                topk_indices = np.argsort(dist, axis=-1)[:, :5]
                sample_topk = np.random.choice(range(0, topk_indices.shape[1]), topk_indices.shape[0])
                agent_token_index[same_idx] = topk_indices[np.arange(topk_indices.shape[0]), sample_topk][same_idx]

            token_contour_select = agent_token_world[np.arange(agent_num), agent_token_index]

            diff_xy = token_contour_select[:, 0, :] - token_contour_select[:, 3, :]

            prev_heading = heading[:, i].copy()
            prev_heading[valid_mask[:, i - self.shift]] = np.arctan2(diff_xy[:, 1], diff_xy[:, 0])[valid_mask[:, i - self.shift]]

            prev_pos = pos[:, i].copy()
            prev_pos[valid_mask[:, i - self.shift]] = token_contour_select.mean(axis=1)[valid_mask[:, i - self.shift]]
            prev_token_idx = agent_token_index
            token_index_list.append(agent_token_index[:, None])
            token_contour_list.append(token_contour_select[:, None, ...])

        token_index = np.concatenate(token_index_list, axis=1)
        token_contour = np.concatenate(token_contour_list, axis=1)

        # extra matching
        if not self.training:
            theta = heading[extra_mask, self.current_step - 1]
            prev_pos = pos[extra_mask, self.current_step - 1]
            cur_pos = pos[extra_mask, self.current_step]
            cur_heading = heading[extra_mask, self.current_step]
            cos, sin = np.cos(theta), np.sin(theta)
            rot_mat = np.zeros((extra_mask.sum(), 2, 2))
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = np.matmul(token_last, rot_mat).reshape(
                extra_mask.sum(), token_num, token_contour_dim, feat_dim)
            agent_token_world += prev_pos[:, None, None, :]

            cur_contour = cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
            dist = np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world) ** 2, axis=-1)).mean(axis=2)
            agent_token_index = np.argmin(dist, axis=-1)
            token_contour_select = agent_token_world[np.arange(extra_mask.sum()), agent_token_index]

            token_index[extra_mask, 1] = agent_token_index
            token_contour[extra_mask, 1] = token_contour_select

        return token_index, token_contour.astype(np.float32)


    def tokenize_map(self, data):
        data['map_polygon']['type'] = data['map_polygon']['type'].astype(np.uint8)
        data['map_point']['type'] = data['map_point']['type'].astype(np.uint8)
        pt2pl = data[('map_point', 'to', 'map_polygon')]['edge_index']
        pt_type = data['map_point']['type'].astype(np.uint8)
        pt_side = np.zeros_like(pt_type)
        pt_pos = data['map_point']['position'][:, :2]
        data['map_point']['orientation'] = wrap_angle(data['map_point']['orientation'])
        pt_heading = data['map_point']['orientation']
        split_polyline_type = []
        split_polyline_pos = []
        split_polyline_theta = []
        split_polyline_side = []
        pl_idx_list = []
        split_polygon_type = []
        # data['map_point']['type'].unique()

        for i in sorted(np.unique(pt2pl[1])):
            index = pt2pl[0, pt2pl[1] == i]
            polygon_type = data['map_polygon']["type"][i]
            cur_side = pt_side[index]
            cur_type = pt_type[index]
            cur_pos = pt_pos[index]
            cur_heading = pt_heading[index]

            for side_val in np.unique(cur_side):
                for type_val in np.unique(cur_type):
                    if type_val == 13:
                        continue
                    indices = np.where((cur_side == side_val) & (cur_type == type_val))[0]
                    if len(indices) <= 2:
                        continue
                    split_polyline = interplating_polyline(cur_pos[indices], cur_heading[indices])
                    if split_polyline is None:
                        continue
                    new_cur_type = cur_type[indices][0]
                    new_cur_side = cur_side[indices][0]
                    map_polygon_type = polygon_type.repeat(split_polyline.shape[0])
                    new_cur_type = new_cur_type.repeat(split_polyline.shape[0])
                    new_cur_side = new_cur_side.repeat(split_polyline.shape[0])
                    cur_pl_idx = np.array([i])
                    new_cur_pl_idx = cur_pl_idx.repeat(split_polyline.shape[0])
                    split_polyline_pos.append(split_polyline[..., :2])
                    split_polyline_theta.append(split_polyline[..., 2])
                    split_polyline_type.append(new_cur_type)
                    split_polyline_side.append(new_cur_side)
                    pl_idx_list.append(new_cur_pl_idx)
                    split_polygon_type.append(map_polygon_type)

        split_polyline_pos = np.concatenate(split_polyline_pos, axis=0)
        split_polyline_theta = np.concatenate(split_polyline_theta, axis=0)
        split_polyline_type = np.concatenate(split_polyline_type, axis=0)
        split_polyline_side = np.concatenate(split_polyline_side, axis=0)
        split_polygon_type = np.concatenate(split_polygon_type, axis=0)
        pl_idx_list = np.concatenate(pl_idx_list, axis=0)
        vec = split_polyline_pos[:, 1, :] - split_polyline_pos[:, 0, :]
        data['map_save'] = {}
        data['pt_token'] = {}
        data['map_save']['traj_pos'] = split_polyline_pos
        data['map_save']['traj_theta'] = split_polyline_theta[:, 0]  # torch.arctan2(vec[:, 1], vec[:, 0])
        data['map_save']['pl_idx_list'] = pl_idx_list
        data['pt_token']['type'] = split_polyline_type
        data['pt_token']['side'] = split_polyline_side
        data['pt_token']['pl_type'] = split_polygon_type
        data['pt_token']['num_nodes'] = split_polyline_pos.shape[0]
        return data