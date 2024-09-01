import os
import pickle
import shutil
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm

from unitraj.datasets import common_utils
from unitraj.datasets.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type, interpolate_polyline
from unitraj.datasets.types import object_type, polyline_type
from unitraj.utils.visualization import check_loaded_data

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)


class BaseDataset(Dataset):

    def __init__(self, config=None, is_validation=False):
        if is_validation:
            self.data_path = config['val_data_path']
        else:
            self.data_path = config['train_data_path']
        self.is_validation = is_validation
        self.config = config
        self.data_loaded_memory = []
        self.data_chunk_size = 8
        self.load_data()

    def load_data(self):
        self.data_loaded = {}
        if self.is_validation:
            print('Loading validation data...')
        else:
            print('Loading training data...')

        for cnt, data_path in enumerate(self.data_path):
            dataset_name = data_path.split('/')[-1]
            self.cache_path = os.path.join(data_path, f'cache_{self.config.method.model_name}')

            data_usage_this_dataset = self.config['max_data_num'][cnt]
            data_usage_this_dataset = int(data_usage_this_dataset / self.data_chunk_size)
            self.starting_frame = self.config['starting_frame'][cnt]
            if self.config['use_cache'] or is_ddp():
                file_list = self.get_data_list(data_usage_this_dataset)
            else:
                if os.path.exists(self.cache_path) and self.config.get('overwrite_cache', False) is False:
                    print('Warning: cache path {} already exists, skip '.format(self.cache_path))
                    file_list = self.get_data_list(data_usage_this_dataset)
                else:

                    _, summary_list, mapping = read_dataset_summary(data_path)

                    if os.path.exists(self.cache_path):
                        shutil.rmtree(self.cache_path)
                    os.makedirs(self.cache_path, exist_ok=True)
                    process_num = os.cpu_count() - 1
                    print('Using {} processes to load data...'.format(process_num))

                    data_splits = np.array_split(summary_list, process_num)

                    data_splits = [(data_path, mapping, list(data_splits[i]), dataset_name) for i in range(process_num)]
                    # save the data_splits in a tmp directory
                    os.makedirs('tmp', exist_ok=True)
                    for i in range(process_num):
                        with open(os.path.join('tmp', '{}.pkl'.format(i)), 'wb') as f:
                            pickle.dump(data_splits[i], f)

                    # results = self.process_data_chunk(0)
                    with Pool(processes=process_num) as pool:
                        results = pool.map(self.process_data_chunk, list(range(process_num)))

                    # concatenate the results
                    file_list = {}
                    for result in results:
                        file_list.update(result)

                    with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                        pickle.dump(file_list, f)

                    data_list = list(file_list.items())
                    np.random.shuffle(data_list)
                    if not self.is_validation:
                        # randomly sample data_usage number of data
                        file_list = dict(data_list[:data_usage_this_dataset])

            print('Loaded {} samples from {}'.format(len(file_list) * self.data_chunk_size, data_path))
            self.data_loaded.update(file_list)

            if self.config['store_data_in_memory']:
                print('Loading data into memory...')
                for data_path in file_list.keys():
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    self.data_loaded_memory.append(data)
                print('Loaded {} data into memory'.format(len(self.data_loaded_memory)))
        # if not self.is_validation:
        # kalman_list = np.concatenate([x['kalman_difficulty'] for x in self.data_loaded.values()],0)[:,-1]
        # sampled_list, index = self.sample_from_distribution(kalman_list, 100)
        # self.data_loaded = {key: value for i, (key, value) in enumerate(self.data_loaded.items()) if i in index}

        self.data_loaded_keys = list(self.data_loaded.keys())
        print('Data loaded')

    def process_data_chunk(self, worker_index):
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        output_buffer = []
        save_cnt = 0
        for cnt, file_name in enumerate(data_list):
            if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                print(f'{cnt}/{len(data_list)} data processed', flush=True)

            scenario = read_scenario(data_path, mapping, file_name)

            try:
                output = self.preprocess(scenario)

                output = self.process(output)

                output = self.postprocess(output)

            except Exception as e:
                print('Error: {} in {}'.format(e, file_name))
                output = None

            if output is None: continue

            output_buffer += output

            while len(output_buffer) >= self.data_chunk_size:
                save_path = os.path.join(self.cache_path, f'{worker_index}_{save_cnt}.pkl')
                to_save = output_buffer[:self.data_chunk_size]
                output_buffer = output_buffer[self.data_chunk_size:]
                with open(save_path, 'wb') as f:
                    pickle.dump(to_save, f)
                save_cnt += 1
                file_info = {}
                kalman_difficulty = np.stack([x['kalman_difficulty'] for x in to_save])
                file_info['kalman_difficulty'] = kalman_difficulty
                file_info['sample_num'] = len(to_save)
                file_list[save_path] = file_info

        save_path = os.path.join(self.cache_path, f'{worker_index}_{save_cnt}.pkl')
        # if output_buffer is not a list
        if isinstance(output_buffer, dict):
            output_buffer = [output_buffer]
        if len(output_buffer) > 0:
            with open(save_path, 'wb') as f:
                pickle.dump(output_buffer, f)
            file_info = {}
            kalman_difficulty = np.stack([x['kalman_difficulty'] for x in output_buffer])
            file_info['kalman_difficulty'] = kalman_difficulty
            file_info['sample_num'] = len(output_buffer)
            file_list[save_path] = file_info

        return file_list

    def preprocess(self, scenario):

        traffic_lights = scenario['dynamic_map_states']
        tracks = scenario['tracks']
        map_feat = scenario['map_features']

        past_length = self.config['past_len']
        future_length = self.config['future_len']
        total_steps = past_length + future_length
        starting_fame = self.starting_frame
        ending_fame = starting_fame + total_steps
        trajectory_sample_interval = self.config['trajectory_sample_interval']
        frequency_mask = generate_mask(past_length - 1, total_steps, trajectory_sample_interval)

        track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': []
        }

        for k, v in tracks.items():

            state = v['state']
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [state['position'], state['length'], state['width'], state['height'], state['heading'],
                         state['velocity'], state['valid']]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)
            # all_state = all_state[::sample_inverval]
            if all_state.shape[0] < ending_fame:
                all_state = np.pad(all_state, ((ending_fame - all_state.shape[0], 0), (0, 0)))
            all_state = all_state[starting_fame:ending_fame]

            assert all_state.shape[0] == total_steps, f'Error: {all_state.shape[0]} != {total_steps}'

            track_infos['object_id'].append(k)
            track_infos['object_type'].append(object_type[v['type']])
            track_infos['trajs'].append(all_state)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos['trajs'][..., -1] *= frequency_mask[np.newaxis]
        scenario['metadata']['ts'] = scenario['metadata']['ts'][:total_steps]

        # x,y,z,type
        map_infos = {
            'lane': [],
            'road_line': [],
            'road_edge': [],
            'stop_sign': [],
            'crosswalk': [],
            'speed_bump': [],
        }
        polylines = []
        point_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v['type']]
            if polyline_type_ == 0:
                continue

            cur_info = {'id': k}
            cur_info['type'] = v['type']
            if polyline_type_ in [1, 2, 3]:
                cur_info['speed_limit_mph'] = v.get('speed_limit_mph', None)
                cur_info['interpolating'] = v.get('interpolating', None)
                cur_info['entry_lanes'] = v.get('entry_lanes', None)
                try:
                    cur_info['left_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['left_neighbor']
                    ]
                    cur_info['right_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['right_neighbor']
                    ]
                except:
                    cur_info['left_boundary'] = []
                    cur_info['right_boundary'] = []
                polyline = v['polyline']
                polyline = interpolate_polyline(polyline)
                map_infos['lane'].append(cur_info)
            elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:
                try:
                    polyline = v['polyline']
                except:
                    polyline = v['polygon']
                polyline = interpolate_polyline(polyline)
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [15, 16]:
                polyline = v['polyline']
                polyline = interpolate_polyline(polyline)
                cur_info['type'] = 7
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [17]:
                cur_info['lane_ids'] = v['lane']
                cur_info['position'] = v['position']
                map_infos['stop_sign'].append(cur_info)
                polyline = v['position'][np.newaxis]
            elif polyline_type_ in [18]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            elif polyline_type_ in [19]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            if polyline.shape[-1] == 2:
                polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
            try:
                cur_polyline_dir = get_polyline_dir(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = polyline_type_
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
        map_infos['all_polylines'] = polylines

        dynamic_map_infos = {
            'lane_id': [],
            'state': [],
            'stop_point': []
        }
        for k, v in traffic_lights.items():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in v['state']['object_state']:  # (num_observed_signals)
                lane_id.append(str(v['lane']))
                state.append(cur_signal)
                if type(v['stop_point']) == list:
                    stop_point.append(v['stop_point'])
                else:
                    stop_point.append(v['stop_point'].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]
            lane_id = lane_id[:total_steps]
            state = state[:total_steps]
            stop_point = stop_point[:total_steps]
            dynamic_map_infos['lane_id'].append(np.array([lane_id]))
            dynamic_map_infos['state'].append(np.array([state]))
            dynamic_map_infos['stop_point'].append(np.array([stop_point]))

        ret = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }
        ret.update(scenario['metadata'])
        ret['timestamps_seconds'] = ret.pop('ts')
        ret['current_time_index'] = self.config['past_len'] - 1
        ret['sdc_track_index'] = track_infos['object_id'].index(ret['sdc_id'])
        if self.config['only_train_on_ego'] or ret.get('tracks_to_predict', None) is None:
            tracks_to_predict = {
                'track_index': [ret['sdc_track_index']],
                'difficulty': [0],
                'object_type': [MetaDriveType.VEHICLE]
            }
        else:
            sample_list = list(ret['tracks_to_predict'].keys())  # + ret.get('objects_of_interest', [])
            sample_list = list(set(sample_list))

            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],
            }

        ret['tracks_to_predict'] = tracks_to_predict

        ret['map_center'] = scenario['metadata'].get('map_center', np.zeros(3))[np.newaxis]

        ret['track_length'] = total_steps
        return ret

    def process(self, internal_format):

        info = internal_format
        scene_id = info['scenario_id']

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']

        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )
        if center_objects is None: return None

        sample_num = center_objects.shape[0]

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
         obj_trajs_future_mask, center_gt_trajs,
         center_gt_trajs_mask, center_gt_final_valid_idx,
         track_index_to_predict_new) = self.get_agent_data(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types
        )

        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
            'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scene_id}')

        if self.config.manually_split_lane:
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_manually_split_map_data(
                center_objects=center_objects, map_infos=info['map_infos'])
        else:
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_map_data(
                center_objects=center_objects, map_infos=info['map_infos'])

        ret_dict['map_polylines'] = map_polylines_data
        ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict['map_polylines_center'] = map_polylines_center

        # masking out unused attributes to Zero
        masked_attributes = self.config['masked_attributes']
        if 'z_axis' in masked_attributes:
            ret_dict['obj_trajs'][..., 2] = 0
            ret_dict['map_polylines'][..., 2] = 0
        if 'size' in masked_attributes:
            ret_dict['obj_trajs'][..., 3:6] = 0
        if 'velocity' in masked_attributes:
            ret_dict['obj_trajs'][..., 25:27] = 0
        if 'acceleration' in masked_attributes:
            ret_dict['obj_trajs'][..., 27:29] = 0
        if 'heading' in masked_attributes:
            ret_dict['obj_trajs'][..., 23:25] = 0

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        ret_dict['map_center'] = ret_dict['map_center'].repeat(sample_num, axis=0)
        ret_dict['dataset_name'] = [info['dataset']] * sample_num

        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                ret_dict_i[k] = v[i]
            ret_list.append(ret_dict_i)

        return ret_list

    def postprocess(self, output):

        # Add the trajectory difficulty
        get_kalman_difficulty(output)

        # Add the trajectory type (stationary, straight, right turn...)
        get_trajectory_type(output)

        return output

    def collate_fn(self, data_list):
        batch_list = []
        for batch in data_list:
            batch_list += batch

        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            # if val_list is str:
            try:
                input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except:
                input_dict[key] = val_list

        input_dict['center_objects_type'] = input_dict['center_objects_type'].numpy()

        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_size}
        return batch_dict

    def __len__(self):
        return len(self.data_loaded)

    def __getitem__(self, idx):
        if self.config['store_data_in_memory']:
            return self.data_loaded_memory[idx]
        else:
            with open(self.data_loaded_keys[idx], 'rb') as f:
                return pickle.load(f)

    def get_data_list(self, data_usage):
        file_list_path = os.path.join(self.cache_path, 'file_list.pkl')
        if os.path.exists(file_list_path):
            data_loaded = pickle.load(open(file_list_path, 'rb'))
        else:
            raise ValueError('Error: file_list.pkl not found')

        data_list = list(data_loaded.items())
        np.random.shuffle(data_list)

        if not self.is_validation:
            # randomly sample data_usage number of data
            data_loaded = dict(data_list[:data_usage])
        else:
            data_loaded = dict(data_list)
        return data_loaded

    def get_agent_data(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types
    ):

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[np.arange(num_center_objects), track_index_to_predict, :, 3] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1

        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9],
            acce,
        ], axis=-1)

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        max_num_agents = self.config['max_num_agents']
        object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new)

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []
        selected_type = self.config['object_type']
        selected_type = [object_type[x] for x in selected_type]
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                print(f'Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}')
                continue
            if obj_types[obj_idx] not in selected_type:
                continue

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        if len(center_objects_list) == 0:
            print(f'Warning: no center objects at time step {current_time_index}, scene_id={scene_id}')
            return None, []
        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    def transform_trajs_to_center_coords(self, obj_trajs, center_xyz, center_heading, heading_index,
                                         rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = np.tile(obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
                angle=-center_heading
            ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def get_map_data(self, center_objects, map_infos):

        num_center_objects = center_objects.shape[0]

        def transform_to_center_coordinates(neighboring_polylines):
            neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
            neighboring_polylines[:, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 0:2],
                angle=-center_objects[:, 6]
            )
            neighboring_polylines[:, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 3:5],
                angle=-center_objects[:, 6]
            )

            return neighboring_polylines

        polylines = np.expand_dims(map_infos['all_polylines'].copy(), axis=0).repeat(num_center_objects, axis=0)

        map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
        num_of_src_polylines = self.config['max_num_roads']
        map_infos['polyline_transformed'] = map_polylines

        all_polylines = map_infos['polyline_transformed']
        max_points_per_lane = self.config.get('max_points_per_lane', 20)
        line_type = self.config.get('line_type', [])
        map_range = self.config.get('map_range', None)
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []

        for k, v in map_infos.items():
            if k == 'all_polylines' or k not in line_type:
                continue
            if len(v) == 0:
                continue
            for polyline_dict in v:
                polyline_index = polyline_dict.get('polyline_index', None)
                polyline_segment = all_polylines[:, polyline_index[0]:polyline_index[1]]
                polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
                polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
                in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)

                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(in_range_mask[i]))
                max_segments = max([len(x) for x in segment_index_list])

                segment_list = np.zeros([num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32)
                segment_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=np.int32)

                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_index = segment_index_list[i]
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(0, segment.shape[0] - 1, max_points_per_lane, dtype=int)]
                            segment_mask_list[i, num] = 1
                        else:
                            segment_list[i, num, :segment.shape[0]] = segment
                            segment_mask_list[i, num, :segment.shape[0]] = 1

                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
        if len(polyline_list) == 0: return np.zeros((num_agents, 0, max_points_per_lane, 7)), np.zeros(
            (num_agents, 0, max_points_per_lane))
        batch_polylines = np.concatenate(polyline_list, axis=1)
        batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(-1) / np.clip(
            batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None)
        polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
        topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

        # Ensure topk_idxs has the correct shape for indexing
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(batch_polylines_mask, topk_idxs[..., 0], axis=1)

        # pad map_polylines and map_polylines_mask to num_of_src_polylines
        map_polylines = np.pad(map_polylines,
                               ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask,
                                    ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(float)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0,
                                                  a_max=None)  # (num_center_objects, num_polylines, 3)

        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-1]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask, map_polylines_center

    def get_manually_split_map_data(self, center_objects, map_infos):
        """
        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2):, [offset_x, offset_y]
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = center_objects.shape[0]
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))

        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].reshape(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)

            # use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:3]
            xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = np.concatenate((neighboring_polylines, xy_pos_pre), axis=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = map_infos['all_polylines'].copy()
        center_objects = center_objects

        point_dim = polylines.shape[-1]

        point_sampled_interval = self.config['point_sampled_interval']
        vector_break_dist_thresh = self.config['vector_break_dist_thresh']
        num_points_each_polyline = self.config['num_points_each_polyline']

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]),
                                       axis=-1)  # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = \
            (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4],
                            axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        batch_polylines = np.stack(ret_polylines, axis=0)
        batch_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        # collect a number of closest polylines for each center objects
        num_of_src_polylines = self.config['max_num_roads']

        if len(batch_polylines) > num_of_src_polylines:
            # Sum along a specific axis and divide by the minimum clamped sum
            polyline_center = np.sum(batch_polylines[:, :, 0:2], axis=1) / np.clip(
                np.sum(batch_polylines_mask, axis=1)[:, None].astype(float), a_min=1.0, a_max=None)
            # Convert the center_offset to a numpy array and repeat it for each object
            center_offset_rot = np.tile(np.array(center_offset, dtype=np.float32)[None, :], (num_center_objects, 1))

            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot[:, None, :],
                angle=center_objects[:, 6]
            )

            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot[:, 0]

            dist = np.linalg.norm(pos_of_map_centers[:, None, :] - polyline_center[None, :, :], axis=-1)

            # Getting the top-k smallest distances and their indices
            topk_idxs = np.argsort(dist, axis=1)[:, :num_of_src_polylines]
            map_polylines = batch_polylines[
                topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[
                topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)

        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 0)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 0)

            map_polylines = np.pad(map_polylines,
                                   ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
            map_polylines_mask = np.pad(map_polylines_mask,
                                        ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(np.float32)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1)[:, :, np.newaxis].astype(float),
                                                  a_min=1.0, a_max=None)

        map_types = map_polylines[:, :, :, 6]
        xy_pos_pre = map_polylines[:, :, :, 7:]
        map_polylines = map_polylines[:, :, :, :6]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)

        return map_polylines, map_polylines_mask, map_polylines_center

    def sample_from_distribution(self, original_array, m=100):
        distribution = [
            ("-10,0", 0),
            ("0,10", 23.952629169758517),
            ("10,20", 24.611144221251667),
            ("20,30.0", 21.142773679220554),
            ("30,40.0", 15.996653629820514),
            ("40,50.0", 9.446714336574939),
            ("50,60.0", 3.7812939732733786),
            ("60,70", 0.8821063091988663),
            ("70,80.0", 0.1533644322320915),
            ("80,90.0", 0.027777741552241064),
            ("90,100.0", 0.005542507117231198),
        ]

        # Define bins and calculate sample sizes for each bin
        bins = np.array([float(range_.split(',')[1]) for range_, _ in distribution])
        sample_sizes = np.array([round(perc / 100 * m) for _, perc in distribution])

        # Digitize the original array into bins
        bin_indices = np.digitize(original_array, bins)

        # Sample from each bin
        sampled_indices = []
        for i, size in enumerate(sample_sizes):
            # Find indices of original array that fall into current bin
            indices_in_bin = np.where(bin_indices == i)[0]
            # Sample without replacement to avoid duplicates
            sampled_indices_in_bin = np.random.choice(indices_in_bin, size=min(size, len(indices_in_bin)),
                                                      replace=False)
            sampled_indices.extend(sampled_indices_in_bin)

        # Extract the sampled elements and their original indices
        sampled_array = original_array[sampled_indices]
        print('total sample:', len(sampled_indices))
        # Verify distribution (optional, for demonstration)
        for i, (range_, _) in enumerate(distribution):
            print(
                f"Bin {range_}: Expected {distribution[i][1]}%, Actual {len(np.where(bin_indices[sampled_indices] == i)[0]) / len(sampled_indices) * 100}%")

        return sampled_array, sampled_indices


import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def draw_figures(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    train_set = build_dataset(cfg)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0,
                                               collate_fn=train_set.collate_fn)
    # for data in train_loader:
    #     inp = data['input_dict']
    #     plt = check_loaded_data(inp, 0)
    #     plt.show()

    concat_list = [4, 4, 4, 4, 4, 4, 4, 4]
    images = []
    for n, data in tqdm(enumerate(train_loader)):
        for i in range(data['batch_size']):
            plt = check_loaded_data(data['input_dict'], i)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            images.append(img)
        if len(images) >= sum(concat_list):
            break
    final_image = concatenate_varying(images, concat_list)
    final_image.show()

    # kalman_dict = {}
    # # create 10 buckets with length 10 as the key
    # for i in range(10):
    #     kalman_dict[i] = {}
    #
    # data_list = []
    # for data in train_loader:
    #     inp = data['input_dict']
    #     kalman_diff = inp['kalman_difficulty']
    #     for idx,k in enumerate(kalman_diff):
    #         k6 = np.floor(k[2]/10)
    #         if k6 in kalman_dict and len(kalman_dict[k6]) == 0:
    #             kalman_dict[k6]['kalman'] = k[2]
    #             kalman_dict[k6]['data'] = inp
    #             check_loaded_data()
    #


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def split_data(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    train_set = build_dataset(cfg)

    copy_dir = ''
    for data in tqdm(train_set.data_loaded_keys):
        shutil.copy(data, copy_dir)


if __name__ == '__main__':
    from unitraj.datasets import build_dataset
    from unitraj.utils.utils import set_seed
    import io
    from PIL import Image
    from unitraj.utils.visualization import concatenate_varying

    split_data()
    # draw_figures()
