import math
import os
import random

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Sampler


def is_ddp():
    return "WORLD_SIZE" in os.environ


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z_tensor(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = torch.stack((
            cosa, sina,
            -sina, cosa
        ), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_points_along_z(points, angle):
    """
    Rotate points around the Z-axis using the given angle.

    Args:
        points: ndarray of shape (B, N, 3 + C) - B batches, N points per batch, 3 coordinates (x, y, z) + C extra channels
        angle: ndarray of shape (B,) - angles for each batch in radians

    Returns:
        Rotated points as an ndarray.
    """

    # Checking if the input is 2D or 3D points
    is_2d = points.shape[-1] == 2

    # Cosine and sine of the angles
    cosa = np.cos(angle)
    sina = np.sin(angle)

    if is_2d:
        # Rotation matrix for 2D
        rot_matrix = np.stack((
            cosa, sina,
            -sina, cosa
        ), axis=1).reshape(-1, 2, 2)

        # Apply rotation
        points_rot = np.matmul(points, rot_matrix)
    else:
        # Rotation matrix for 3D
        rot_matrix = np.stack((
            cosa, sina, np.zeros_like(angle),
            -sina, cosa, np.zeros_like(angle),
            np.zeros_like(angle), np.zeros_like(angle), np.ones_like(angle)
        ), axis=1).reshape(-1, 3, 3)

        # Apply rotation to the first 3 dimensions
        points_rot = np.matmul(points[:, :, :3], rot_matrix)

        # Concatenate any additional dimensions back
        if points.shape[-1] > 3:
            points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    return points_rot


def generate_mask(current_index, total_length, interval):
    mask = []
    for i in range(total_length):
        # Check if the position is a multiple of the frequency starting from current_index
        if (i - current_index) % interval == 0:
            mask.append(1)
        else:
            mask.append(0)

    return np.array(mask)


def find_true_segments(mask):
    # Find the indices where `True` changes to `False` and vice versa
    change_points = np.where(np.diff(mask))[0] + 1

    # Add the start and end indices
    indices = np.concatenate(([0], change_points, [len(mask)]))

    # Extract the segments of continuous `True`
    segments = [list(range(indices[i], indices[i + 1])) for i in range(len(indices) - 1) if mask[indices[i]]]

    return segments


def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    assert len(tensor_list[0].shape) in [3, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == 3:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        only_3d_tensor = True
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2, print(cur_tensor.shape)

        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, num_feat1, num_feat2)
        new_tensor[:, :cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0)
        new_mask_tensor[:, :cur_tensor.shape[1]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if only_3d_tensor:
        ret_tensor = ret_tensor.squeeze(dim=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def estimate_kalman_filter(history, prediction_horizon):
    """
    Predict the future position by running the kalman filter.

    :param history: 2d array of shape (length_of_history, 2)
    :param prediction_horizon: how many steps in the future to predict
    :return: the predicted position (x, y)

    Code taken from:
    On Exposing the Challenging Long Tail in Future Prediction of Traffic Actors
    """
    length_history = history.shape[0]
    z_x = history[:, 0]
    z_y = history[:, 1]
    v_x = 0
    v_y = 0
    for index in range(length_history - 1):
        v_x += z_x[index + 1] - z_x[index]
        v_y += z_y[index + 1] - z_y[index]
    v_x = v_x / (length_history - 1)
    v_y = v_y / (length_history - 1)
    x_x = np.zeros(length_history + 1, np.float32)
    x_y = np.zeros(length_history + 1, np.float32)
    P_x = np.zeros(length_history + 1, np.float32)
    P_y = np.zeros(length_history + 1, np.float32)
    P_vx = np.zeros(length_history + 1, np.float32)
    P_vy = np.zeros(length_history + 1, np.float32)

    # we initialize the uncertainty to one (unit gaussian)
    P_x[0] = 1.0
    P_y[0] = 1.0
    P_vx[0] = 1.0
    P_vy[0] = 1.0
    x_x[0] = z_x[0]
    x_y[0] = z_y[0]

    Q = 0.00001
    R = 0.0001
    K_x = np.zeros(length_history + 1, np.float32)
    K_y = np.zeros(length_history + 1, np.float32)
    K_vx = np.zeros(length_history + 1, np.float32)
    K_vy = np.zeros(length_history + 1, np.float32)
    for k in range(length_history - 1):
        x_x[k + 1] = x_x[k] + v_x
        x_y[k + 1] = x_y[k] + v_y
        P_x[k + 1] = P_x[k] + P_vx[k] + Q
        P_y[k + 1] = P_y[k] + P_vy[k] + Q
        P_vx[k + 1] = P_vx[k] + Q
        P_vy[k + 1] = P_vy[k] + Q
        K_x[k + 1] = P_x[k + 1] / (P_x[k + 1] + R)
        K_y[k + 1] = P_y[k + 1] / (P_y[k + 1] + R)
        x_x[k + 1] = x_x[k + 1] + K_x[k + 1] * (z_x[k + 1] - x_x[k + 1])
        x_y[k + 1] = x_y[k + 1] + K_y[k + 1] * (z_y[k + 1] - x_y[k + 1])
        P_x[k + 1] = P_x[k + 1] - K_x[k + 1] * P_x[k + 1]
        P_y[k + 1] = P_y[k + 1] - K_y[k + 1] * P_y[k + 1]
        K_vx[k + 1] = P_vx[k + 1] / (P_vx[k + 1] + R)
        K_vy[k + 1] = P_vy[k + 1] / (P_vy[k + 1] + R)
        P_vx[k + 1] = P_vx[k + 1] - K_vx[k + 1] * P_vx[k + 1]
        P_vy[k + 1] = P_vy[k + 1] - K_vy[k + 1] * P_vy[k + 1]

    k = k + 1
    x_x[k + 1] = x_x[k] + v_x * prediction_horizon
    x_y[k + 1] = x_y[k] + v_y * prediction_horizon
    P_x[k + 1] = P_x[k] + P_vx[k] * prediction_horizon * prediction_horizon + Q
    P_y[k + 1] = P_y[k] + P_vy[k] * prediction_horizon * prediction_horizon + Q
    P_vx[k + 1] = P_vx[k] + Q
    P_vy[k + 1] = P_vy[k] + Q
    return x_x[k + 1], x_y[k + 1]


def calculate_epe(pred, gt):
    diff_x = (gt[0] - pred[0]) * (gt[0] - pred[0])
    diff_y = (gt[1] - pred[1]) * (gt[1] - pred[1])
    epe = math.sqrt(diff_x + diff_y)
    return epe


def count_valid_steps_past(mask):
    reversed_mask = mask[::-1]  # Reverse the mask
    idx_of_first_zero = np.where(reversed_mask == 0)[0]  # Find the index of the first zero
    if len(idx_of_first_zero) == 0:
        return len(mask)  # If no zeros, return the length of the mask
    else:
        return idx_of_first_zero[0]  # Return the index of the first zero


def get_kalman_difficulty(output):
    """
    return the kalman difficulty at 2s, 4s, and 6s
    if the gt future is not valid up to the considered second, the difficulty is set to -1
    """
    for data_sample in output:
        # past trajectory of agent of interest
        past_trajectory = data_sample["obj_trajs"][0, :, :2]  # Time X (x,y)
        past_mask = data_sample["obj_trajs_mask"][0, :]
        valid_past = count_valid_steps_past(past_mask)
        past_trajectory_valid = past_trajectory[-valid_past:, :]  # Time(valid) X (x,y)

        # future gt trajectory of agent of interest
        gt_future = data_sample["obj_trajs_future_state"][0, :, :2]  # Time x (x, y)
        # Get last valid position
        valid_future = int(data_sample["center_gt_final_valid_idx"])

        kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s = -1, -1, -1
        try:
            if valid_future >= 19:
                # Get kalman future prediction at the horizon length, second argument is horizon length
                kalman_2s = estimate_kalman_filter(past_trajectory_valid, 20)  # (x,y)
                gt_future_2s = gt_future[19, :]
                kalman_difficulty_2s = calculate_epe(kalman_2s, gt_future_2s)

                if valid_future >= 39:
                    kalman_4s = estimate_kalman_filter(past_trajectory_valid, 40)  # (x,y)
                    gt_future_4s = gt_future[39, :]
                    kalman_difficulty_4s = calculate_epe(kalman_4s, gt_future_4s)

                    if valid_future >= 59:
                        kalman_6s = estimate_kalman_filter(past_trajectory_valid, 60)  # (x,y)
                        gt_future_6s = gt_future[59, :]
                        kalman_difficulty_6s = calculate_epe(kalman_6s, gt_future_6s)
        except:
            kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s = -1, -1, -1
        data_sample["kalman_difficulty"] = np.array([kalman_difficulty_2s, kalman_difficulty_4s, kalman_difficulty_6s])
    return


class TrajectoryType:
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_RIGHT = 2
    STRAIGHT_LEFT = 3
    RIGHT_U_TURN = 4
    RIGHT_TURN = 5
    LEFT_U_TURN = 6
    LEFT_TURN = 7


def classify_track(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading):
    # The classification strategy is taken from
    # waymo_open_dataset/metrics/motion_metrics_utils.cc#L28

    # Parameters for classification, taken from WOD
    kMaxSpeedForStationary = 2.0  # (m/s)
    kMaxDisplacementForStationary = 5.0  # (m)
    kMaxLateralDisplacementForStraight = 5.0  # (m)
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)

    x_delta = end_point[0] - start_point[0]
    y_delta = end_point[1] - start_point[1]

    final_displacement = np.hypot(x_delta, y_delta)
    heading_diff = end_heading - start_heading
    normalized_delta = np.array([x_delta, y_delta])
    rotation_matrix = np.array([[np.cos(-start_heading), -np.sin(-start_heading)],
                                [np.sin(-start_heading), np.cos(-start_heading)]])
    normalized_delta = np.dot(rotation_matrix, normalized_delta)
    start_speed = np.hypot(start_velocity[0], start_velocity[1])
    end_speed = np.hypot(end_velocity[0], end_velocity[1])
    max_speed = max(start_speed, end_speed)
    dx, dy = normalized_delta

    # Check for different trajectory types based on the computed parameters.
    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return TrajectoryType.STATIONARY
    if np.abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if np.abs(normalized_delta[1]) < kMaxLateralDisplacementForStraight:
            return TrajectoryType.STRAIGHT
        return TrajectoryType.STRAIGHT_RIGHT if dy < 0 else TrajectoryType.STRAIGHT_LEFT
    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return TrajectoryType.RIGHT_U_TURN if normalized_delta[
                                                  0] < kMinLongitudinalDisplacementForUTurn else TrajectoryType.RIGHT_TURN
    if dx < kMinLongitudinalDisplacementForUTurn:
        return TrajectoryType.LEFT_U_TURN
    return TrajectoryType.LEFT_TURN


def interpolate_polyline(polyline, step=0.5):
    # Calculate the cumulative distance along the polyline
    if polyline.shape[0] == 1:
        return polyline
    polyline = polyline[:, :2]
    distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # start with a distance of 0

    # Create the new distance array
    max_distance = distances[-1]
    new_distances = np.arange(0, max_distance, step)

    # Interpolate for x, y, z
    new_polyline = []
    for dim in range(polyline.shape[1]):
        interp_func = interp1d(distances, polyline[:, dim], kind='linear')
        new_polyline.append(interp_func(new_distances))

    new_polyline = np.column_stack(new_polyline)
    # add the third dimension back with zeros
    new_polyline = np.concatenate((new_polyline, np.zeros((new_polyline.shape[0], 1))), axis=1)
    return new_polyline


def get_heading(trajectory):
    # trajectory has shape (Time X (x,y))

    dx_ = np.diff(trajectory[:, 0])
    dy_ = np.diff(trajectory[:, 1])
    heading = np.arctan2(dy_, dx_)

    return heading


def get_trajectory_type(output):
    for data_sample in output:
        # Get last gt position, velocity and heading
        valid_end_point = int(data_sample["center_gt_final_valid_idx"])
        end_point = data_sample["obj_trajs_future_state"][0, valid_end_point, :2]  # (x,y)
        end_velocity = data_sample["obj_trajs_future_state"][0, valid_end_point, 2:]  # (vx, vy)
        # Get last heading, manually approximate it from the series of future position
        end_heading = get_heading(data_sample["obj_trajs_future_state"][0, :valid_end_point + 1, :2])[-1]

        # Get start position, velocity and heading.
        assert data_sample["obj_trajs_mask"][0, -1]  # Assumes that the start point is always valid
        start_point = data_sample["obj_trajs"][0, -1, :2]  # (x,y)
        start_velocity = data_sample["obj_trajs"][0, -1, -4:-2]  # (vx, vy)
        start_heading = 0.  # Initial heading is zero

        # Classify the trajectory
        try:
            trajectory_type = classify_track(start_point, end_point, start_velocity, end_velocity, start_heading,
                                             end_heading)
        except:
            trajectory_type = -1
        data_sample["trajectory_type"] = trajectory_type
    return


class DynamicSampler(Sampler):
    def __init__(self, datasets):
        """
        datasets: Dictionary of datasets.
        epoch_to_datasets: A dict where keys are epoch numbers and values are lists of dataset names to be used in that epoch.
        """
        self.datasets = datasets
        self.config = datasets.config
        all_dataset = self.datasets.dataset_idx.keys()
        self.sample_num = self.config['sample_num']
        self.sample_mode = self.config['sample_mode']

        data_usage_dict = {}
        max_data_num = self.config['max_data_num']
        for k, num in zip(all_dataset, max_data_num):
            data_usage_dict[k] = num
        # self.selected_idx = self.datasets.dataset_idx
        # self.reset()
        self.set_sampling_strategy(data_usage_dict)

    def set_sampling_strategy(self, sampleing_dict):
        all_idx = []
        selected_idx = {}
        for k, v in sampleing_dict.items():
            assert k in self.datasets.dataset_idx.keys()
            data_idx = self.datasets.dataset_idx[k]
            if v <= 1.0:
                data_num = int(len(data_idx) * v)
            else:
                data_num = int(v)
            if data_num == 0:
                continue
            data_num = min(data_num, len(data_idx))
            # randomly select data_idx by data_num
            sampled_data_idx = np.random.choice(data_idx, data_num, replace=False).tolist()
            all_idx.extend(sampled_data_idx)
            selected_idx[k] = sampled_data_idx

        self.idx = all_idx[:self.sample_num]
        self.selected_idx = selected_idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

    def reset(self):
        all_index = []
        for k, v in self.selected_idx.items():
            all_index.extend(v)
        self.idx = all_index

    def set_idx(self, idx):
        self.idx = idx


trajectory_correspondance = {0: "stationary", 1: "straight", 2: "straight_right",
                             3: "straight_left", 4: "right_u_turn", 5: "right_turn",
                             6: "left_u_turn", 7: "left_turn"}
