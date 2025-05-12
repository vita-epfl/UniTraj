import os
import pickle
import torch
import h5py
from collections import defaultdict, Counter
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from torch_geometric.data import Dataset
from unitraj.models.smart.smart_utils import Logging
import numpy as np
from torch_geometric.data import HeteroData, Batch
from torch_geometric.data.storage import NodeStorage
from torch_geometric.loader.dataloader import Collater
from torch_geometric.transforms import BaseTransform
from unitraj.models.smart.datasets.preprocess import TokenProcessor
from .base_dataset import BaseDataset
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario
from unitraj.datasets.common_utils import (
    get_polyline_dir,
    find_true_segments,
    generate_mask,
    get_polyline_mag,
    merge_tuple_key,
    save_item,
    load_item,
)
from unitraj.models.smart.smart_utils import wrap_angle
from unitraj.datasets.types import (
    object_type,
    polyline_type,
    smart_polygon_type,
    smart_traffic_light_state_to_int,
)

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)
polygon_type = defaultdict(lambda: default_value, smart_polygon_type)
traffic_light_state_to_int = defaultdict(
    lambda: default_value, smart_traffic_light_state_to_int
)


class SMARTDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        self.token_processor = TokenProcessor(2048)
        self.target_transform = WaymoTargetBuilder(11, 80)
        self.scene_centric = True
        # self.scene_centric = False
        super().__init__(config, is_validation)
        self.logger = Logging().log(level="DEBUG")

    def process_data_chunk(self, worker_index):
        with open(os.path.join("tmp", "{}.pkl".format(worker_index)), "rb") as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f"{worker_index}.h5")

        with h5py.File(hdf5_path, "w") as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f"{cnt}/{len(data_list)} data processed", flush=True)
                scenario = read_scenario(data_path, mapping, file_name)

                try:
                    output = self.preprocess(scenario)

                    output = self.process(output)

                    output = self.postprocess(output)

                except Exception as e:
                    print("Warning: {} in {}".format(e, file_name))
                    output = None

                if output is None:
                    continue

                for i, record in enumerate(output):
                    grp_name = (
                        dataset_name
                        + "-"
                        + str(worker_index)
                        + "-"
                        + str(cnt)
                        + "-"
                        + str(i)
                    )
                    grp = f.create_group(grp_name)
                    if isinstance(record, HeteroData):
                        record = record.to_dict()
                        global_items = record.pop("_global_store", None)
                        if global_items:
                            record.update(global_items)
                    for key, value in record.items():
                        key = merge_tuple_key(key)
                        save_item(grp, key, value)
                    file_info = {}
                    kalman_difficulty = np.stack(
                        [x["kalman_difficulty"] for x in output]
                    )
                    file_info["kalman_difficulty"] = kalman_difficulty
                    file_info["h5_path"] = hdf5_path
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list

    def preprocess(self, scenario):

        traffic_lights = scenario["dynamic_map_states"]
        tracks = scenario["tracks"]
        map_feat = scenario["map_features"]

        past_length = self.config["past_len"]
        future_length = self.config["future_len"]
        total_steps = past_length + future_length
        starting_fame = self.starting_frame
        ending_fame = starting_fame + total_steps
        trajectory_sample_interval = self.config["trajectory_sample_interval"]
        frequency_mask = generate_mask(
            past_length - 1, total_steps, trajectory_sample_interval
        )

        track_infos = {
            "object_id": [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            "object_type": [],
            "trajs": [],
        }

        for k, v in tracks.items():

            state = v["state"]
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [
                state["position"],
                state["length"],
                state["width"],
                state["height"],
                state["heading"],
                state["velocity"],
                state["valid"],
            ]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)
            # all_state = all_state[::sample_inverval]
            if all_state.shape[0] < ending_fame:
                all_state = np.pad(
                    all_state, ((ending_fame - all_state.shape[0], 0), (0, 0))
                )
            all_state = all_state[starting_fame:ending_fame]

            assert (
                all_state.shape[0] == total_steps
            ), f"Error: {all_state.shape[0]} != {total_steps}"

            track_infos["object_id"].append(k)
            track_infos["object_type"].append(object_type[v["type"]])
            track_infos["trajs"].append(all_state)

        track_infos["trajs"] = np.stack(track_infos["trajs"], axis=0)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos["trajs"][..., -1] *= frequency_mask[np.newaxis]
        scenario["metadata"]["ts"] = scenario["metadata"]["ts"][:total_steps]

        # x,y,z,type
        map_infos = {
            "lane": [],
            "road_line": [],
            "road_edge": [],
            "stop_sign": [],
            "crosswalk": [],
            "speed_bump": [],
        }
        polylines = []
        point_cnt = 0
        point_to_polyline_edge_index = None
        polyline_light_type = None
        polygon_type_ = None
        point_type = None
        point_orientation = None
        point_mag = None
        polyline_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v["type"]]
            # if polyline_type_ == 0:
            # continue

            cur_info = {"id": k}
            cur_info["type"] = v["type"]
            if k in traffic_lights.keys():
                cur_info["light_type"] = traffic_light_state_to_int[
                    traffic_lights[k]["state"]["object_state"][past_length - 1]
                ]  # take light states at 11
            else:
                cur_info["light_type"] = traffic_light_state_to_int[None]
            if polyline_type_ in [1, 2, 3]:
                cur_info["speed_limit_mph"] = v.get("speed_limit_mph", None)
                cur_info["interpolating"] = v.get("interpolating", None)
                cur_info["entry_lanes"] = v.get("entry_lanes", None)
                try:
                    cur_info["left_boundary"] = [
                        {
                            "start_index": x["self_start_index"],
                            "end_index": x["self_end_index"],
                            "feature_id": x["feature_id"],
                            "boundary_type": "UNKNOWN",  # roadline type
                        }
                        for x in v["left_neighbor"]
                    ]
                    cur_info["right_boundary"] = [
                        {
                            "start_index": x["self_start_index"],
                            "end_index": x["self_end_index"],
                            "feature_id": x["feature_id"],
                            "boundary_type": "UNKNOWN",  # roadline type
                        }
                        for x in v["right_neighbor"]
                    ]
                except:
                    cur_info["left_boundary"] = []
                    cur_info["right_boundary"] = []
                polyline = v["polyline"]
                # polyline = interpolate_polyline(polyline)
                map_infos["lane"].append(cur_info)
                if polyline_type_ == 3:
                    cur_info["polygon_type"] = polygon_type["BIKE"]
                else:
                    cur_info["polygon_type"] = polygon_type["VEHICLE"]
            elif polyline_type_ in [0]:
                cur_info["speed_limit_mph"] = v.get("speed_limit_mph", None)
                cur_info["interpolating"] = v.get("interpolating", None)
                cur_info["entry_lanes"] = v.get("entry_lanes", None)
                cur_info["left_boundary"] = []
                cur_info["right_boundary"] = []
                try:
                    polyline = v["polyline"]
                except:
                    polyline = v["polygon"]
                # polyline = interpolate_polyline(polyline)
                cur_info["polygon_type"] = polygon_type["BUS"]  # unknown
                continue
            elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:
                try:
                    polyline = v["polyline"]
                except:
                    polyline = v["polygon"]
                # polyline = interpolate_polyline(polyline)
                map_infos["road_line"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["VEHICLE"]
            elif polyline_type_ in [15, 16]:
                polyline = v["polyline"]
                # polyline = interpolate_polyline(polyline)
                cur_info["type"] = 7
                map_infos["road_line"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["VEHICLE"]
            elif polyline_type_ in [17]:  # stopsign
                cur_info["lane_ids"] = v["lane"]
                cur_info["position"] = v["position"]
                map_infos["stop_sign"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["BUS"]
                polyline = v["position"][np.newaxis]
            elif polyline_type_ in [18]:
                map_infos["crosswalk"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["PEDESTRIAN"]
                polyline = v["polygon"]
            elif polyline_type_ in [19]:  # speed bump
                map_infos["crosswalk"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["BUS"]
                polyline = v["polygon"]
                continue
            if polyline.shape[-1] == 2:
                polyline = np.concatenate(
                    (polyline, np.zeros((polyline.shape[0], 1))), axis=-1
                )
            try:
                cur_polyline_dir = get_polyline_dir(polyline)
                cur_polyline_mag = get_polyline_mag(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = polyline_type_
                light_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = cur_info["light_type"]
                cur_polyline = np.concatenate(
                    (polyline, cur_polyline_dir, light_array, type_array), axis=-1
                )
            except:
                cur_polyline = np.zeros((0, 8), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
            cur_edge_index = np.stack(
                [
                    np.arange(point_cnt, point_cnt + len(cur_polyline)),
                    np.repeat(polyline_cnt, len(cur_polyline)),
                ],
                axis=0,
            )
            if point_to_polyline_edge_index is None:
                point_to_polyline_edge_index = cur_edge_index
            else:
                point_to_polyline_edge_index = np.concatenate(
                    [point_to_polyline_edge_index, cur_edge_index], axis=1
                )

            if polyline_light_type is None:
                polyline_light_type = np.array([cur_info["light_type"]])
            else:
                polyline_light_type = np.concatenate(
                    [polyline_light_type, np.array([cur_info["light_type"]])]
                )

            if point_type is None:
                point_type = np.repeat(polyline_type_, len(cur_polyline))
            else:
                point_type = np.concatenate(
                    [point_type, np.repeat(polyline_type_, len(cur_polyline))], axis=0
                )

            if point_orientation is None:
                point_orientation = np.arctan2(
                    cur_polyline_dir[:, 1], cur_polyline_dir[:, 0]
                )
            else:
                point_orientation = np.concatenate(
                    [
                        point_orientation,
                        np.arctan2(cur_polyline_dir[:, 1], cur_polyline_dir[:, 0]),
                    ],
                    axis=0,
                )

            if point_mag is None:
                point_mag = np.array(cur_polyline_mag)
            else:
                point_mag = np.concatenate(
                    [point_mag, np.array(cur_polyline_mag)], axis=0
                )

            if polygon_type_ is None:
                polygon_type_ = np.array([cur_info["polygon_type"]])
            else:
                # if cur_info['polygon_type']
                polygon_type_ = np.concatenate(
                    [polygon_type_, np.array([cur_info["polygon_type"]])]
                )
            point_cnt += len(cur_polyline)
            polyline_cnt += 1

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 8), dtype=np.float32)
        map_infos["all_polylines"] = polylines  # points (n, 8)
        map_infos["point_to_polyline_edge_index"] = point_to_polyline_edge_index
        map_infos["point_type"] = point_type
        map_infos["point_orientation"] = point_orientation
        map_infos["point_magnitude"] = point_mag
        map_infos["polyline_light_type"] = polyline_light_type
        map_infos["polyline_type"] = polygon_type_

        dynamic_map_infos = {"lane_id": [], "state": [], "stop_point": []}
        for k, v in traffic_lights.items():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in v["state"]["object_state"]:  # (num_observed_signals)
                lane_id.append(str(v["lane"]))
                state.append(cur_signal)
                if type(v["stop_point"]) == list:
                    stop_point.append(v["stop_point"])
                else:
                    stop_point.append(v["stop_point"].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]
            lane_id = lane_id[:total_steps]
            state = state[:total_steps]
            stop_point = stop_point[:total_steps]
            dynamic_map_infos["lane_id"].append(np.array([lane_id]))
            dynamic_map_infos["state"].append(np.array([state]))
            dynamic_map_infos["stop_point"].append(np.array([stop_point]))

        ret = {
            "track_infos": track_infos,
            "dynamic_map_infos": dynamic_map_infos,
            "map_infos": map_infos,
        }
        ret.update(scenario["metadata"])
        ret["timestamps_seconds"] = ret.pop("ts")
        ret["current_time_index"] = self.config["past_len"] - 1
        ret["sdc_track_index"] = track_infos["object_id"].index(ret["sdc_id"])
        if (
            self.config["only_train_on_ego"]
            or ret.get("tracks_to_predict", None) is None
        ):
            tracks_to_predict = {
                "track_index": [ret["sdc_track_index"]],
                "difficulty": [0],
                "object_type": [MetaDriveType.VEHICLE],
            }
        else:
            sample_list = list(
                ret["tracks_to_predict"].keys()
            )  # + ret.get('objects_of_interest', [])
            sample_list = list(set(sample_list))

            tracks_to_predict = {
                "track_index": [
                    track_infos["object_id"].index(id)
                    for id in sample_list
                    if id in track_infos["object_id"]
                ],
                "object_type": [
                    track_infos["object_type"][track_infos["object_id"].index(id)]
                    for id in sample_list
                    if id in track_infos["object_id"]
                ],
            }

        ret["tracks_to_predict"] = tracks_to_predict

        ret["map_center"] = scenario["metadata"].get("map_center", np.zeros(3))[
            np.newaxis
        ]

        ret["track_length"] = total_steps
        return ret

    def process(self, internal_format):

        info = internal_format
        scene_id = info["scenario_id"]

        sdc_track_index = info["sdc_track_index"]
        current_time_index = info["current_time_index"]
        timestamps = np.array(
            info["timestamps_seconds"][: current_time_index + 1], dtype=np.float32
        )

        track_infos = info["track_infos"]

        track_index_to_predict = np.array(info["tracks_to_predict"]["track_index"])
        obj_types = np.array(track_infos["object_type"])
        obj_trajs_full = track_infos["trajs"]  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, : current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1 :]

        if self.scene_centric:
            center_objects = None
        else:
            center_objects, track_index_to_predict = self.get_interested_agents(
                track_index_to_predict=track_index_to_predict,
                obj_trajs_full=obj_trajs_full,
                current_time_index=current_time_index,
                obj_types=obj_types,
                scene_id=scene_id,
            )
        # if center_objects is None: return None
        if center_objects is None:
            sample_num = 1
            (
                obj_trajs_data,
                obj_trajs_mask,
                obj_trajs_pos,
                obj_trajs_last_pos,
                obj_trajs_future_state,
                obj_trajs_future_mask,
                track_index_to_predict_new,
            ) = self.get_agent_data_global(
                obj_trajs_past=obj_trajs_past,
                obj_trajs_future=obj_trajs_future,
                track_index_to_predict=track_index_to_predict,
                sdc_track_index=sdc_track_index,
                timestamps=timestamps,
                obj_types=obj_types,
            )
        else:
            sample_num = center_objects.shape[0]
            (
                obj_trajs_data,
                obj_trajs_mask,
                obj_trajs_pos,
                obj_trajs_last_pos,
                obj_trajs_future_state,
                obj_trajs_future_mask,
                center_gt_trajs,
                center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new,
            ) = self.get_agent_data(
                center_objects=center_objects,
                obj_trajs_past=obj_trajs_past,
                obj_trajs_future=obj_trajs_future,
                track_index_to_predict=track_index_to_predict,
                sdc_track_index=sdc_track_index,
                timestamps=timestamps,
                obj_types=obj_types,
            )
        dynamic_map_infos = info["dynamic_map_infos"]

        ret_dict = {
            "scenario_id": (
                np.array([scene_id] * len(track_index_to_predict_new))
                if center_objects
                else np.array([scene_id])
            ),
            "obj_trajs": obj_trajs_data,
            "obj_trajs_mask": obj_trajs_mask,
            "track_index_to_predict": track_index_to_predict_new,
            "obj_trajs_pos": obj_trajs_pos,
            "obj_trajs_last_pos": obj_trajs_last_pos,
            "map_center": info["map_center"],
            "obj_trajs_future_state": obj_trajs_future_state,
            "obj_trajs_future_mask": obj_trajs_future_mask,
            # 'light_lane_id': np.array(dynamic_map_infos['lane_id']),
            # 'light_state': np.array(dynamic_map_infos['state']),
            # 'light_stop_point': np.array(dynamic_map_infos['stop_point'])
        }
        if center_objects is not None:
            ret_dict.update(
                {
                    "center_objects_world": center_objects,
                    "center_objects_id": np.array(track_infos["object_id"])[
                        track_index_to_predict
                    ],
                    "center_objects_type": np.array(track_infos["object_type"])[
                        track_index_to_predict
                    ],
                    "center_gt_trajs": center_gt_trajs,
                    "center_gt_trajs_mask": center_gt_trajs_mask,
                    "center_gt_final_valid_idx": center_gt_final_valid_idx,
                    "center_gt_trajs_src": obj_trajs_full[track_index_to_predict],
                }
            )

        if info["map_infos"]["all_polylines"].__len__() == 0:
            info["map_infos"]["all_polylines"] = np.zeros((2, 8), dtype=np.float32)
            print(f"Warning: empty HDMap {scene_id}")

        if self.config.manually_split_lane:
            map_polylines_data, map_polylines_mask, map_polylines_center = (
                self.get_manually_split_map_data(
                    center_objects=center_objects, map_infos=info["map_infos"]
                )
            )
            ret_dict["map_polylines_center"] = map_polylines_center
        else:
            if self.scene_centric:
                pass
                # map_polylines_data, map_polylines_mask = self.get_map_data_global(
                #     map_infos=info['map_infos'], light_infos=dynamic_map_infos)
            else:
                map_polylines_data, map_polylines_mask, map_polylines_center = (
                    self.get_map_data(
                        center_objects=center_objects, map_infos=info["map_infos"]
                    )
                )
                ret_dict["map_polylines_center"] = map_polylines_center

        # ret_dict['map_polylines'] = map_polylines_data
        # ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict["map_points_pos"] = info["map_infos"]["all_polylines"].copy()[
            None, ...
        ]
        ret_dict["point_to_polyline_edge_index"] = info["map_infos"][
            "point_to_polyline_edge_index"
        ].copy()[None, ...]
        ret_dict["map_points_type"] = info["map_infos"]["point_type"].copy()[None, ...]
        ret_dict["map_points_orientation"] = info["map_infos"][
            "point_orientation"
        ].copy()[None, ...]
        ret_dict["map_points_magnitude"] = info["map_infos"]["point_magnitude"].copy()[
            None, ...
        ]
        ret_dict["polyline_light_type"] = info["map_infos"][
            "polyline_light_type"
        ].copy()[None, ...]
        ret_dict["polyline_type"] = info["map_infos"]["polyline_type"].copy()[None, ...]

        # masking out unused attributes to Zero
        masked_attributes = self.config["masked_attributes"]
        if "z_axis" in masked_attributes:
            ret_dict["obj_trajs"][..., 2] = 0
            # ret_dict['map_polylines'][..., 2] = 0
            ret_dict["map_points_pos"][..., 2] = 0
        if "size" in masked_attributes:
            ret_dict["obj_trajs"][..., 3:6] = 0
        if "velocity" in masked_attributes:
            ret_dict["obj_trajs"][..., 25:27] = 0
        if "acceleration" in masked_attributes:
            ret_dict["obj_trajs"][..., 27:29] = 0
        if "heading" in masked_attributes:
            ret_dict["obj_trajs"][..., 23:25] = 0

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        ret_dict["map_center"] = ret_dict["map_center"].repeat(sample_num, axis=0)
        ret_dict["dataset_name"] = [info["dataset"]] * sample_num

        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                if k == "track_index_to_predict" and self.scene_centric:
                    ret_dict_i[k] = v
                else:
                    ret_dict_i[k] = v[i]
            ret_list.append(ret_dict_i)

        return ret_list

    def get_agent_data_global(
        self,
        obj_trajs_past,
        obj_trajs_future,
        track_index_to_predict,
        sdc_track_index,
        timestamps,
        obj_types,
    ):
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = obj_trajs_past

        object_onehot_mask = np.zeros((num_objects, num_timestamps, 5))
        object_onehot_mask[obj_types == 1, :, 0] = 1
        object_onehot_mask[obj_types == 2, :, 1] = 1
        object_onehot_mask[obj_types == 3, :, 2] = 1
        object_onehot_mask[track_index_to_predict, :, 3] = 1  # track2pred
        object_onehot_mask[sdc_track_index, :, 4] = 1  # av

        object_time_embedding = np.zeros(
            (num_objects, num_timestamps, num_timestamps + 1)
        )
        for i in range(num_timestamps):
            object_time_embedding[:, i, i] = 1
        object_time_embedding[:, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, 0] = np.sin(obj_trajs[:, :, 6])
        object_heading_embedding[:, :, 1] = np.cos(obj_trajs[:, :, 6])

        vel = obj_trajs[:, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=1)
        acce = (vel - vel_pre) / 0.1
        acce[:, 0, :] = acce[:, 1, :]

        obj_trajs_data = np.concatenate(
            [
                obj_trajs[:, :, 0:6],
                object_onehot_mask,
                object_time_embedding,
                object_heading_embedding,
                obj_trajs[:, :, 7:9],
                acce,
            ],
            axis=-1,
        )

        obj_trajs_mask = obj_trajs[:, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0
        obj_trajs_mask[:, 1:] = obj_trajs_mask[:, 1:] * obj_trajs_mask[:, :-1]
        obj_trajs_mask[:, 0] = False

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future_state = obj_trajs_future[
            :, :, [0, 1, 3, 4, 5, 6, 7, 8]
        ]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[0]
        # valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)
        # valid_past_mask = obj_trajs_past[:, num_timestamps-1, -1].astype(bool)
        valid_past_mask = (obj_trajs_past[:, num_timestamps - 1, :2] != 0).all(axis=-1)

        obj_trajs_mask = obj_trajs_mask[valid_past_mask]
        obj_trajs_data = obj_trajs_data[valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, 0:3]
        num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, k, :][cur_valid_mask]

        max_num_agents = self.config["max_num_agents"]
        num_objs = obj_trajs.shape[0]

        obj_trajs_data = obj_trajs_data[None, ...]
        obj_trajs_mask = obj_trajs_mask[None, ...]
        obj_trajs_pos = obj_trajs_pos[None, ...]
        obj_trajs_last_pos = obj_trajs_last_pos[None, ...]
        obj_trajs_future_state = obj_trajs_future_state[None, ...]
        obj_trajs_future_mask = obj_trajs_future_mask[None, ...]
        # track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)[None, ...]
        track_index_to_predict = np.where(
            np.arange(num_objs)[valid_past_mask] == sdc_track_index
        )[0][0]

        return (
            obj_trajs_data,
            obj_trajs_mask.astype(bool),
            obj_trajs_pos,
            obj_trajs_last_pos,
            obj_trajs_future_state,
            obj_trajs_future_mask,
            track_index_to_predict,
        )

    def get_map_data_global(self, map_infos, light_infos=None):
        map_polylines = np.expand_dims(
            map_infos["all_polylines"].copy(), axis=0
        ).repeat(1, axis=0)

        num_of_src_polylines = self.config["max_num_roads"]
        map_infos["polyline_transformed"] = map_polylines

        all_polylines = map_infos["polyline_transformed"]
        max_points_per_lane = self.config.get("max_points_per_lane", 20)
        line_type = self.config.get("line_type", [])
        # map_range = self.config.get('map_range', None)
        # center_offset = self.config.get('center_offset_of_map', (30.0, 0))
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []

        for k, v in map_infos.items():
            if k == "all_polylines" or k not in line_type:
                continue
            if len(v) == 0:
                continue
            for polyline_dict in v:
                polyline_index = polyline_dict.get("polyline_index", None)
                polyline_segment = all_polylines[
                    :, polyline_index[0] : polyline_index[1]
                ]
                in_range_mask = np.ones(polyline_segment.shape[:2], dtype=bool)

                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(in_range_mask[i]))
                max_segments = max([len(x) for x in segment_index_list])

                segment_list = np.zeros(
                    [num_agents, max_segments, max_points_per_lane, 8], dtype=np.float32
                )
                segment_mask_list = np.zeros(
                    [num_agents, max_segments, max_points_per_lane], dtype=np.int32
                )

                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_index = segment_index_list[i]
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(
                                    0,
                                    segment.shape[0] - 1,
                                    max_points_per_lane,
                                    dtype=int,
                                )
                            ]
                            segment_mask_list[i, num] = 1
                        else:
                            segment_list[i, num, : segment.shape[0]] = segment
                            segment_mask_list[i, num, : segment.shape[0]] = 1

                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
        if len(polyline_list) == 0:
            return np.zeros((num_agents, 0, max_points_per_lane, 8)), np.zeros(
                (num_agents, 0, max_points_per_lane)
            )
        map_polylines = np.concatenate(polyline_list, axis=1)
        map_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        light_types = map_polylines[:, :, :, -2]
        light_types = np.eye(9)[light_types.astype(int)]  # use 9 for light types

        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-2]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate(
            (map_polylines, xy_pos_pre, light_types, map_types), axis=-1
        )
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask

    def get_interested_agents(
        self,
        track_index_to_predict,
        obj_trajs_full,
        current_time_index,
        obj_types,
        scene_id,
    ):
        center_objects_list = []
        track_index_to_predict_selected = []
        selected_type = self.config["object_type"]
        selected_type = [object_type[x] for x in selected_type]
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                print(
                    f"Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}"
                )
                continue
            if obj_types[obj_idx] not in selected_type:
                print(
                    f"Missing obj_types {obj_types[obj_idx]} at time step {current_time_index}, scene_id={scene_id}"
                )
                continue

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        if len(center_objects_list) == 0:
            print(
                f"Warning: no center objects at time step {current_time_index}, scene_id={scene_id}"
            )
            return None, []
        center_objects = np.stack(
            center_objects_list, axis=0
        )  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    def postprocess(self, output):

        # get_kalman_difficulty(output)

        # get_trajectory_type(output)

        return self.smart_convert(output)

    def smart_convert(self, output):
        data = []
        for i in output:
            d = {
                "scenario_id": i["scenario_id"],
                # 'kalman_difficulty': i['kalman_difficulty'],
                # 'trajectory_type': i['trajectory_type'],
            }
            obj_trajs = i["obj_trajs"]
            num_historical_steps = obj_trajs.shape[1]

            agent = {}
            agent["num_nodes"] = i["obj_trajs"].shape[0]
            type_one_hot_mask = obj_trajs[:, :, 6:11]
            agent["av_index"] = i["track_index_to_predict"]
            # agent['av_index'] = i['sdc_track_index']

            valid_mask = np.concatenate(
                (i["obj_trajs_mask"], i["obj_trajs_future_mask"].astype(bool)), axis=-1
            )
            agent["valid_mask"] = valid_mask

            predict_mask = np.zeros_like(valid_mask, dtype=bool)
            predict_mask[:, num_historical_steps:] = True
            predict_mask[~valid_mask] = False
            agent["predict_mask"] = predict_mask

            agent["type"] = np.argmax(i["obj_trajs"][:, 0, 6:11], axis=1)
            # agent['type'] = np.array([self.custom_argmax(row) for row in i['obj_trajs'][:, 0, 6:11]])
            agent["category"] = np.zeros(agent["num_nodes"], dtype=np.uint8)

            agent["position"] = np.concatenate(
                [i["obj_trajs_pos"], i["obj_trajs_future_state"][..., :3]], axis=1
            ).astype(np.float32)

            obj_his_heading_encoding = i["obj_trajs"][..., 23:25]
            obj_his_heading = np.arctan2(
                obj_his_heading_encoding[..., 0], obj_his_heading_encoding[..., 1]
            )
            obj_future_heading = i["obj_trajs_future_state"][..., 5]
            agent["heading"] = np.concatenate(
                [obj_his_heading, obj_future_heading], axis=1
            ).astype(np.float32)

            velo_all = np.concatenate(
                [i["obj_trajs"][..., 25:27], i["obj_trajs_future_state"][..., 3:5]],
                axis=1,
            )
            agent["velocity"] = np.pad(
                velo_all,
                pad_width=((0, 0), (0, 0), (0, 1)),
                mode="constant",
                constant_values=0.0,
            ).astype(np.float32)

            if i.get("center_objects_world"):
                agent["center_objects_world"] = (
                    i["center_objects_world"].reshape(1, -1).astype(np.float32)
                )

            agent["shape"] = np.concatenate(
                [i["obj_trajs"][..., 3:6], i["obj_trajs_future_state"][..., 2:5]],
                axis=1,
            ).astype(np.float32)
            d["agent"] = NodeStorage(agent)

            d["map_polygon"] = {}
            d["map_polygon"]["num_nodes"] = i["polyline_light_type"].shape[0]
            d["map_polygon"]["light_type"] = i["polyline_light_type"].copy()
            d["map_polygon"]["type"] = i["polyline_type"].copy()
            d["map_polygon"] = NodeStorage(d["map_polygon"])

            d["map_point"] = {}
            d["map_point"]["num_nodes"] = i["map_points_pos"].shape[0]
            d["map_point"]["position"] = i["map_points_pos"][:, :2]
            d["map_point"]["orientation"] = i["map_points_orientation"].copy()

            d["map_point"]["magnitude"] = i["map_points_magnitude"].copy()
            d["map_point"]["height"] = np.zeros_like(i["map_points_magnitude"]).astype(
                np.float32
            )
            d["map_point"]["type"] = i["map_points_type"].copy()
            d["map_point"] = NodeStorage(d["map_point"])

            edge_index = i["point_to_polyline_edge_index"].copy()
            d["map_point", "to", "map_polygon"] = NodeStorage(
                {"edge_index": edge_index}
            )
            if i.get("center_objects_type"):
                d["center_objects_type"] = i["center_objects_type"]

            d = self.target_transform(d)

            data.append(self.token_processor.preprocess(d))
        return data

    def collate_fn(self, data_list):
        # if isinstance(data_list[0], List):
        #     data_list = data_list[0]
        batch_size = len(data_list)
        merged_data = HeteroData()

        if self.scene_centric:
            single_keys = ["scenario_id"]
        else:
            single_keys = [
                "scenario_id",
                "kalman_difficulty",
                "trajectory_type",
                "center_objects_type",
            ]
        all_keys = [
            "map_save",
            "pt_token",
            "agent",
            "map_point",
            "map_polygon",
            "map_point_to_map_polygon",
        ]
        sample = data_list[0]
        pt_offset = 0
        pl_offset = 0
        for k in single_keys:
            if k == "scenario_id":
                merged_data[k] = [data[k] for data in data_list]
            elif k == "kalman_difficulty":
                merged_data[k] = torch.from_numpy(
                    np.concatenate([data[k] for data in data_list], axis=0)
                )
            else:
                merged_data[k] = torch.tensor([data[k] for data in data_list])

        for key in all_keys:
            merged_data[key] = {}
            if key != "map_point_to_map_polygon":
                for sub_key in sample[key].keys():
                    if sub_key == "num_nodes":
                        continue
                    elif sub_key == "drivable":
                        merged_data[key][sub_key] = [
                            data[key][sub_key] for data in data_list
                        ]
                    try:
                        merged_data[key][sub_key] = torch.from_numpy(
                            np.concatenate(
                                [data[key][sub_key] for data in data_list], axis=0
                            )
                        )
                    except Exception as e:
                        merged_data[key][sub_key] = torch.tensor(
                            [data[key][sub_key] for data in data_list]
                        )
            else:
                merged_edges = []
                for data in data_list:
                    edge_index = data[key]["edge_index"]
                    edge_index[0] += pt_offset
                    edge_index[1] += pl_offset
                    merged_edges.append(edge_index)
                    pt_offset += edge_index[0].max().item() + 1
                    pl_offset += edge_index[1].max().item() + 1
                merged_data[key]["edge_index"] = torch.from_numpy(
                    np.concatenate(merged_edges, axis=1)
                )
            if key not in ["map_save", "map_point_to_map_polygon"]:
                ptr = torch.tensor([0])
                num_nodes_list = [data[key]["num_nodes"] for data in data_list]
                ptr = torch.cat([ptr, torch.tensor(num_nodes_list, dtype=torch.int64)])
                merged_data[key]["ptr"] = ptr
                merged_data[key]["num_nodes"] = torch.tensor(sum(num_nodes_list))
                merged_data[key]["batch"] = torch.arange(len(ptr)).repeat_interleave(
                    ptr
                )
        return Batch.from_data_list([merged_data])

    def __getitem__(self, idx):
        file_key = self.data_loaded_keys[idx]
        file_info = self.data_loaded[file_key]
        file_path = file_info["h5_path"]

        if file_path not in self.file_cache:
            self.file_cache[file_path] = self._get_file(file_path)

        group = self.file_cache[file_path][file_key]
        record = {k: load_item(group[k]) for k in group.keys()}
        return record

    def custom_argmax(self, row):
        reversed_row = row[::-1]
        max_val = np.max(row)
        if max_val == 0:
            return 0
        else:
            last_idx_reversed = reversed_row.argmax()
            original_idx = len(row) - last_idx_reversed - 1
            return original_idx + 1


class WaymoTargetBuilder(BaseTransform):

    def __init__(
        self, num_historical_steps: int, num_future_steps: int, mode="train"
    ) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.mode = mode
        self.num_features = 3
        self.augment = False
        self.logger = Logging().log(level="DEBUG")

    def score_ego_agent(self, agent):
        av_index = agent["av_index"]
        agent["category"][av_index] = 5
        return agent

    def clip(self, agent, max_num=32):
        av_index = agent["av_index"]
        valid = agent["valid_mask"]
        ego_pos = agent["position"][av_index]
        obstacle_mask = agent["type"] == 3
        distance = torch.norm(
            agent["position"][:, self.num_historical_steps - 1, :2]
            - ego_pos[self.num_historical_steps - 1, :2],
            dim=-1,
        )  # keep the closest 100 vehicles near the ego car
        distance[obstacle_mask] = 10e5
        sort_idx = distance.sort()[1]
        mask = torch.zeros(valid.shape[0])
        mask[sort_idx[:max_num]] = 1
        mask = mask.to(torch.bool)
        mask[av_index] = True
        new_av_index = mask[:av_index].sum()
        agent["num_nodes"] = int(mask.sum())
        agent["av_index"] = int(new_av_index)
        excluded = ["num_nodes", "av_index", "ego"]
        for key, val in agent.items():
            if key in excluded:
                continue
            if key == "id":
                val = list(np.array(val)[mask])
                agent[key] = val
                continue
            if len(val.size()) > 1:
                agent[key] = val[mask, ...]
            else:
                agent[key] = val[mask]
        return agent

    def score_nearby_vehicle(self, agent, max_num=10):
        av_index = agent["av_index"]
        agent["category"] = torch.zeros_like(agent["category"])
        obstacle_mask = agent["type"] == 3
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = torch.norm(
            agent["position"][:, self.num_historical_steps, :2] - pos, dim=-1
        )
        distance[obstacle_mask] = 10e5
        sort_idx = distance.sort()[1]
        nearby_mask = torch.zeros(distance.shape[0])
        nearby_mask[sort_idx[1:max_num]] = 1
        nearby_mask = nearby_mask.bool()
        agent["category"][nearby_mask] = 3
        agent["category"][obstacle_mask] = 0

    def score_trained_vehicle(self, agent, max_num=10, min_distance=0):
        av_index = agent["av_index"]
        agent["category"] = np.zeros_like(agent["category"])
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = np.linalg.norm(
            agent["position"][:, self.num_historical_steps, :2] - pos, axis=-1
        )
        distance_all_time = np.linalg.norm(
            agent["position"][:, :, :2] - agent["position"][av_index, :, :2], axis=-1
        )
        invalid_mask = (
            distance_all_time < 150
        )  # we do not believe the perception out of range of 150 meters
        agent["valid_mask"] = agent["valid_mask"] * invalid_mask
        # we do not predict vehicle  too far away from ego car
        closet_vehicle = distance < 100
        valid = agent["valid_mask"]
        valid_current = valid[:, self.num_historical_steps :]
        valid_counts = valid_current.sum(axis=1)
        counts_vehicle = valid_counts >= 1
        no_backgroud = agent["type"] != 3
        vehicle2pred = closet_vehicle & counts_vehicle & no_backgroud
        if vehicle2pred.sum() > max_num:
            # too many still vehicle so that train the model using the moving vehicle as much as possible
            true_indices = np.nonzero(vehicle2pred)[0]
            selected_indices = np.random.choice(true_indices, max_num, replace=False)
            vehicle2pred[:] = False
            vehicle2pred[selected_indices] = True
        agent["category"][vehicle2pred] = 3

    def rotate_agents(
        self, position, heading, num_nodes, num_historical_steps, num_future_steps
    ):
        origin = position[:, num_historical_steps - 1]
        theta = heading[:, num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_nodes, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        target = origin.new_zeros(num_nodes, num_future_steps, 4)
        target[..., :2] = torch.bmm(
            position[:, num_historical_steps:, :2] - origin[:, :2].unsqueeze(1), rot_mat
        )
        his = origin.new_zeros(num_nodes, num_historical_steps, 4)
        his[..., :2] = torch.bmm(
            position[:, :num_historical_steps, :2] - origin[:, :2].unsqueeze(1), rot_mat
        )
        if position.size(2) == 3:
            target[..., 2] = position[:, num_historical_steps:, 2] - origin[
                :, 2
            ].unsqueeze(-1)
            his[..., 2] = position[:, :num_historical_steps, 2] - origin[
                :, 2
            ].unsqueeze(-1)
            target[..., 3] = wrap_angle(
                heading[:, num_historical_steps:] - theta.unsqueeze(-1)
            )
            his[..., 3] = wrap_angle(
                heading[:, :num_historical_steps] - theta.unsqueeze(-1)
            )
        else:
            target[..., 2] = wrap_angle(
                heading[:, num_historical_steps:] - theta.unsqueeze(-1)
            )
            his[..., 2] = wrap_angle(
                heading[:, :num_historical_steps] - theta.unsqueeze(-1)
            )
        return his, target

    def __call__(self, data) -> HeteroData:
        agent = data["agent"]
        self.score_ego_agent(agent)
        self.score_trained_vehicle(agent, max_num=32)
        return HeteroData(data)
