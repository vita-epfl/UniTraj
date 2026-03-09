from .base_dataset import BaseDataset
from collections import defaultdict
from unitraj.datasets.common_utils import save_item, load_item, merge_tuple_key
import numpy as np
import h5py
import pickle
import os
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from unitraj.datasets.base_dataset import read_scenario
from unitraj.datasets.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type, interpolate_polyline
from unitraj.datasets.types import object_type, polyline_type
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import HeteroData

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)

class HPNetDataset(BaseDataset):
    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        # Lane occlusion ratio (used for data augmentation)
        self.lane_occlusion_ratio = self.config.get('lane_occlusion_ratio', 0.0)
    
    def process_data_chunk(self, worker_index):
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f'{worker_index}.h5')

        with h5py.File(hdf5_path, 'w') as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f'{cnt}/{len(data_list)} data processed', flush=True)
                scenario = read_scenario(data_path, mapping, file_name)

                try:
                    output = self.preprocess(scenario)

                    output = self.process(output)

                    output = self.postprocess(output)

                except Exception as e:
                    print('Warning: {} in {}'.format(e, file_name))
                    output = None

                if output is None: continue

                for i, record in enumerate(output):
                    grp_name = dataset_name + '-' + str(worker_index) + '-' + str(cnt) + '-' + str(i)
                    grp = f.create_group(grp_name)
                    
                    # Recursively save all data using save_item (including nested dicts and tuple keys)
                    for key, value in record.items():
                        # Handle tuple keys (edge types)
                        if isinstance(key, tuple):
                            save_key = '__ET__' + '_'.join(key)  # ('lane', 'lane') → '__ET__lane_lane'
                        else:
                            save_key = key
                        
                        save_item(grp, save_key, value)
                    
                    file_info = {}
                    file_info['h5_path'] = hdf5_path
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list
        
    def preprocess(self, scenario):
        "Modified to retain successor lanes and neighbor lane information"
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
                cur_info['exit_lanes'] = v.get('exit_lanes', None)  # Add successor lanes
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
                # Save left_neighbor and right_neighbor
                cur_info['left_neighbor'] = v.get('left_neighbor', [])
                cur_info['right_neighbor'] = v.get('right_neighbor', [])
                polyline = v['polyline']
                if self.config["max_points_per_lane"] > polyline.shape[0] and (self.config["method"]["model_name"] != "forecast" and self.config ["method"]["model_name"]  != "EMP"): #for those models the data should already be interpolated correctly for pretrained checkpoints to work properly
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

        if self.config['only_train_on_ego']:
            tracks_to_predict = {
                'track_index': [ret['sdc_track_index']],
                'difficulty': [0],
                'object_type': [MetaDriveType.VEHICLE]
            }
        elif ret.get('tracks_to_predict', None) is None:
            filtered_tracks = self.trajectory_filter(ret)
            sample_list = list(filtered_tracks.keys())
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],
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
        """Convert to HPNet format graph data, generate a data dict for each track_to_predict"""
        info = internal_format
        scene_id = info['scenario_id']
        
        current_time_index = info['current_time_index']
        track_infos = info['track_infos']
        map_infos = info['map_infos']
        
        # Get the track indices to predict
        track_index_to_predict = info['tracks_to_predict']['track_index']
        if len(track_index_to_predict) == 0:
            return None
        
        # Get data scale control parameters
        max_num_agents = self.config.get('max_num_agents', None)
        map_range = self.config.get('map_range', None)
        max_num_roads = self.config.get('max_num_roads', None)
        max_points_per_lane = self.config.get('max_points_per_lane', 20)
        
        # Construct agent data (global coordinates)
        num_agents = len(track_infos['object_id'])
        num_steps = self.config['past_len'] + self.config['future_len']
        obj_trajs = track_infos['trajs']  # (num_objects, num_timestamp, 10): [x, y, z, l, w, h, heading, vx, vy, valid]
        
        # Select agents individually for each track_to_predict (if max_num_agents is set)
        # Store the agent index mapping for each track
        track_agent_data = {}
        
        for track_idx in track_index_to_predict:
            if max_num_agents is not None and num_agents > max_num_agents:
                # Calculate the distance of all agents to the current prediction target
                center_pos = obj_trajs[track_idx, current_time_index, :2]
                all_pos = obj_trajs[:, current_time_index, :2]
                distances = np.linalg.norm(all_pos - center_pos, axis=-1)
                
                # Select the nearest max_num_agents
                topk_indices = np.argsort(distances)[:max_num_agents]
                selected_agents = topk_indices.tolist()
                
                # Ensure the prediction target itself is included (usually it is closest to itself, but just in case)
                if track_idx not in selected_agents:
                    selected_agents[-1] = track_idx  # Replace the last one
                
                selected_agents = sorted(selected_agents)
            else:
                # No limit, use all agents
                selected_agents = list(range(num_agents))
            
            # Create index mapping: old_idx -> new_idx
            old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_agents)}
            
            # Filter agent data
            selected_obj_trajs = obj_trajs[selected_agents]
            selected_num_agents = len(selected_agents)
            
            # Construct agent node features
            agent_position = np.zeros((selected_num_agents, num_steps, 2), dtype=np.float32)
            agent_heading = np.zeros((selected_num_agents, self.config['past_len']), dtype=np.float32)
            agent_length = np.zeros((selected_num_agents, self.config['past_len']), dtype=np.float32)
            visible_mask = np.zeros((selected_num_agents, num_steps), dtype=bool)
            length_mask = np.zeros((selected_num_agents, self.config['past_len']), dtype=bool)
            
            for i in range(selected_num_agents):
                # Position: x, y from global coordinates
                agent_position[i] = selected_obj_trajs[i, :, :2]  # (T, 2)
                
                # Visible mask from valid flag
                visible_mask[i] = selected_obj_trajs[i, :, -1] > 0  # valid flag
                
                # Compute motion vectors for heading and length (past only)
                for t in range(self.config['past_len']):
                    if t == 0:
                        length_mask[i, t] = True
                    else:
                        # Check if both current and previous timesteps are visible
                        if visible_mask[i, t] and visible_mask[i, t-1]:
                            motion = agent_position[i, t] - agent_position[i, t-1]
                            agent_length[i, t] = np.linalg.norm(motion)
                            if agent_length[i, t] > 1e-6:
                                agent_heading[i, t] = np.arctan2(motion[1], motion[0])
                            else:
                                agent_heading[i, t] = 0.0
                            length_mask[i, t] = False
                        else:
                            length_mask[i, t] = True
                            agent_length[i, t] = 0.0
                            agent_heading[i, t] = 0.0
            
            # Store agent data for the current track
            track_agent_data[track_idx] = {
                'num_agents': selected_num_agents,
                'agent_index': old_to_new_idx[track_idx],  # Position in the new index
                'position': agent_position,
                'heading': agent_heading,
                'length': agent_length,
                'visible_mask': visible_mask,
                'length_mask': length_mask
            }
        
        # Construct lane data (global coordinates)
        lane_list = map_infos.get('lane', [])
        num_lanes = len(lane_list)
        
        if num_lanes == 0:
            return None
        
        # Apply map_range and max_num_roads limits
        if map_range is not None or max_num_roads is not None:
            # Calculate the center position of each prediction target (using current time)
            center_positions = [obj_trajs[track_idx, current_time_index, :2] for track_idx in track_index_to_predict]
            avg_center_pos = np.mean(center_positions, axis=0)
            
            # Calculate the distance from each lane to the center (using lane center point)
            lane_distances = []
            for lane in lane_list:
                start_idx, end_idx = lane['polyline_index']
                polyline = map_infos['all_polylines'][start_idx:end_idx, :2]
                if len(polyline) > 0:
                    lane_center = np.mean(polyline, axis=0)
                    dist = np.linalg.norm(lane_center - avg_center_pos)
                    lane_distances.append(dist)
                else:
                    lane_distances.append(float('inf'))
            
            lane_distances = np.array(lane_distances)
            
            # Apply map_range filter
            if map_range is not None:
                valid_mask = lane_distances <= map_range
                lane_list = [lane for i, lane in enumerate(lane_list) if valid_mask[i]]
                lane_distances = lane_distances[valid_mask]
            
            # Apply max_num_roads limit
            if max_num_roads is not None and len(lane_list) > max_num_roads:
                topk_indices = np.argsort(lane_distances)[:max_num_roads]
                lane_list = [lane_list[i] for i in topk_indices]
            
            num_lanes = len(lane_list)
            if num_lanes == 0:
                return None
        
        # Build mapping from lane_id to index
        lane_id_to_idx = {lane['id']: idx for idx, lane in enumerate(lane_list)}
        
        lane_position = np.zeros((num_lanes, 2), dtype=np.float32)
        lane_heading = np.zeros(num_lanes, dtype=np.float32)
        lane_length = np.zeros(num_lanes, dtype=np.float32)
        lane_is_intersection = np.zeros(num_lanes, dtype=np.uint8)
        lane_turn_direction = np.zeros(num_lanes, dtype=np.uint8)
        lane_traffic_control = np.zeros(num_lanes, dtype=np.uint8)
        
        # Centerline data
        centerline_positions = []
        centerline_headings = []
        centerline_lengths = []
        num_centerlines_per_lane = np.zeros(num_lanes, dtype=np.int64)
        
        # lane topology edges
        lane_adjacent_edges = []
        lane_predecessor_edges = []
        lane_successor_edges = []
        
        for lane_idx, lane in enumerate(lane_list):
            # Get polyline range
            start_idx, end_idx = lane['polyline_index']
            polyline = map_infos['all_polylines'][start_idx:end_idx, :3]  # (N, 3): x, y, z
            
            if len(polyline) < 2:
                continue
            
            # Apply max_points_per_lane limit: use uniform sampling
            if max_points_per_lane is not None and len(polyline) > max_points_per_lane:
                indices = np.linspace(0, len(polyline) - 1, max_points_per_lane, dtype=int)
                polyline = polyline[indices]
            
            # Centerline processing
            num_centerline_nodes = len(polyline) - 1
            num_centerlines_per_lane[lane_idx] = num_centerline_nodes
            
            for i in range(num_centerline_nodes):
                # Centerline node position (midpoint)
                center_pos = (polyline[i, :2] + polyline[i+1, :2]) / 2
                centerline_positions.append(center_pos)
                
                # Centerline direction and length
                vector = polyline[i+1, :2] - polyline[i, :2]
                length = np.linalg.norm(vector)
                centerline_lengths.append(length)
                
                if length > 1e-6:
                    heading = np.arctan2(vector[1], vector[0])
                else:
                    heading = 0.0
                centerline_headings.append(heading)
            
            # Lane node features (using center point)
            center_index = len(polyline) // 2
            lane_position[lane_idx] = polyline[center_index, :2]
            
            if center_index + 1 < len(polyline):
                vector = polyline[center_index + 1, :2] - polyline[center_index, :2]
                length = np.linalg.norm(vector)
                if length > 1e-6:
                    lane_heading[lane_idx] = np.arctan2(vector[1], vector[0])
            
            # Total lane length
            lane_length[lane_idx] = np.array(centerline_lengths[-num_centerline_nodes:]).sum()
            
            # Lane attributes (set according to actual data, default values used here)
            lane_is_intersection[lane_idx] = 0
            lane_turn_direction[lane_idx] = 0
            lane_traffic_control[lane_idx] = 0
            
            # Construct topology edges
            # Adjacent lanes (left/right neighbors)
            left_neighbors = lane.get('left_neighbor', [])
            right_neighbors = lane.get('right_neighbor', [])
            for neighbor in left_neighbors + right_neighbors:
                neighbor_id = neighbor.get('feature_id')
                if neighbor_id and neighbor_id in lane_id_to_idx:
                    neighbor_idx = lane_id_to_idx[neighbor_id]
                    lane_adjacent_edges.append([neighbor_idx, lane_idx])
            
            # Predecessor lanes (entry lanes)
            entry_lanes = lane.get('entry_lanes', [])
            if entry_lanes:
                for entry_id in entry_lanes:
                    if entry_id in lane_id_to_idx:
                        entry_idx = lane_id_to_idx[entry_id]
                        lane_predecessor_edges.append([entry_idx, lane_idx])
            
            # Successor lanes (exit lanes)
            exit_lanes = lane.get('exit_lanes', [])
            if exit_lanes:
                for exit_id in exit_lanes:
                    if exit_id in lane_id_to_idx:
                        exit_idx = lane_id_to_idx[exit_id]
                        lane_successor_edges.append([exit_idx, lane_idx])
        
        # Convert to numpy arrays
        if len(centerline_positions) == 0:
            return None
        
        centerline_position = np.array(centerline_positions, dtype=np.float32)
        centerline_heading = np.array(centerline_headings, dtype=np.float32)
        centerline_length = np.array(centerline_lengths, dtype=np.float32)
        
        # Construct centerline to lane edges
        centerline_to_lane_edges = []
        centerline_offset = 0
        for lane_idx in range(num_lanes):
            num_nodes = num_centerlines_per_lane[lane_idx]
            for i in range(num_nodes):
                centerline_to_lane_edges.append([centerline_offset + i, lane_idx])
            centerline_offset += num_nodes
        
        centerline_to_lane_edge_index = np.array(centerline_to_lane_edges, dtype=np.int64).T if centerline_to_lane_edges else np.zeros((2, 0), dtype=np.int64)
        
        # Lane topology edges
        adjacent_edge_index = np.array(lane_adjacent_edges, dtype=np.int64).T if lane_adjacent_edges else np.zeros((2, 0), dtype=np.int64)
        predecessor_edge_index = np.array(lane_predecessor_edges, dtype=np.int64).T if lane_predecessor_edges else np.zeros((2, 0), dtype=np.int64)
        successor_edge_index = np.array(lane_successor_edges, dtype=np.int64).T if lane_successor_edges else np.zeros((2, 0), dtype=np.int64)
        
        # Create a data dict for each track_to_predict
        data_list = []
        for track_idx in track_index_to_predict:
            data = {
                'agent': {},
                'lane': {},
                'centerline': {},
                ('centerline', 'lane'): {},
                ('lane', 'lane'): {}
            }
            
            # Get the agent data corresponding to this track
            agent_data = track_agent_data[track_idx]
            
            # Agent data
            data['agent']['num_nodes'] = agent_data['num_agents']
            data['agent']['agent_index'] = agent_data['agent_index']
            data['agent']['visible_mask'] = agent_data['visible_mask'].copy()
            data['agent']['position'] = agent_data['position'].copy()
            data['agent']['heading'] = agent_data['heading'].copy()
            data['agent']['length'] = agent_data['length'].copy()
            
            # Lane data
            data['lane']['num_nodes'] = num_lanes
            data['lane']['position'] = lane_position.copy()
            data['lane']['length'] = lane_length.copy()
            data['lane']['heading'] = lane_heading.copy()
            data['lane']['is_intersection'] = lane_is_intersection.copy()
            data['lane']['turn_direction'] = lane_turn_direction.copy()
            data['lane']['traffic_control'] = lane_traffic_control.copy()
            
            # Centerline data
            data['centerline']['num_nodes'] = len(centerline_position)
            data['centerline']['position'] = centerline_position.copy()
            data['centerline']['heading'] = centerline_heading.copy()
            data['centerline']['length'] = centerline_length.copy()
            
            # Edge data
            data['centerline', 'lane']['centerline_to_lane_edge_index'] = centerline_to_lane_edge_index.copy()
            data['lane', 'lane']['adjacent_edge_index'] = adjacent_edge_index.copy()
            data['lane', 'lane']['predecessor_edge_index'] = predecessor_edge_index.copy()
            data['lane', 'lane']['successor_edge_index'] = successor_edge_index.copy()
            
            # Metadata
            data['scenario_id'] = scene_id
            #data['track_index_to_predict'] = track_idx
            
            data_list.append(data)
        
        return data_list
    
    def postprocess(self, output):
        return output

    
    def __getitem__(self, idx):
        file_key = self.data_loaded_keys[idx]
        file_info = self.data_loaded[file_key]
        file_path = file_info['h5_path']
        
        if file_path not in self.file_cache:
            self.file_cache[file_path] = self._get_file(file_path)
        
        group = self.file_cache[file_path][file_key]
        
        # Load all data using load_item helper
        data_dict = {}
        for key in group.keys():
            loaded_value = load_item(group[key])
            
            # Reconstruct tuple keys - look for __ET__ prefix for edge types
            if key.startswith('__ET__'):
                # Edge type: __ET__lane_lane → ('lane', 'lane')
                edge_type_str = key[6:]  # Remove '__ET__' prefix
                parts = edge_type_str.split('_', 1)
                if len(parts) == 2:
                    data_dict[(parts[0], parts[1])] = loaded_value
                else:
                    # Fallback for unexpected format
                    data_dict[key] = loaded_value
            else:
                # Node type or metadata - keep as-is
                data_dict[key] = loaded_value
        
        # Return plain dict
        return data_dict
    
    def _mask_edge_index(self, edge_index, occlusion_index):
        """Filter out edges that contain occluded lane nodes"""
        mask = ~torch.isin(edge_index[0], occlusion_index)
        return edge_index[:, mask]
    
    def _apply_lane_occlusion(self, hetero):
        """Randomly occlude partial lane nodes and their related edges"""
        num_lanes = hetero['lane']['num_nodes']
        
        # Always create visible_mask (maintain batch consistent attributes)
        visible_mask = torch.ones(num_lanes, dtype=torch.bool)
        
        # Apply occlusion only during training
        if self.lane_occlusion_ratio > 0 and not self.is_validation:
            num_occlusions = int(num_lanes * self.lane_occlusion_ratio)
            
            if num_occlusions > 0:
                # Randomly select lanes to occlude
                occlusion_index = torch.randperm(num_lanes)[:num_occlusions]
                visible_mask[occlusion_index] = False
                
                # Filter out edges containing occluded lanes
                edge_types = ['adjacent_edge_index', 'predecessor_edge_index', 'successor_edge_index']
                for edge_type in edge_types:
                    if edge_type in hetero['lane', 'lane']:
                        edge_index = hetero['lane', 'lane'][edge_type]
                        hetero['lane', 'lane'][edge_type] = self._mask_edge_index(edge_index, occlusion_index)
        
        # Always add visible_mask (may contain False during training, all True during validation)
        hetero['lane']['visible_mask'] = visible_mask
        return hetero
    
    def collate_fn(self, data_list):
        """
        Convert dict list to HeteroData Batch
        """
        import torch
        from torch_geometric.data import Batch, HeteroData
        
        hetero_list = []
        
        for data_dict in data_list:
            # Create HeteroData object
            hetero = HeteroData()
            
            # Convert node type data
            for node_type in ['agent', 'lane', 'centerline']:
                if node_type in data_dict:
                    for key, value in data_dict[node_type].items():
                        if isinstance(value, np.ndarray):
                            # Convert to torch tensor
                            if value.dtype == np.float32 or value.dtype == np.float64:
                                hetero[node_type][key] = torch.from_numpy(value).float()
                            elif value.dtype == np.int64 or value.dtype == np.int32:
                                hetero[node_type][key] = torch.from_numpy(value).long()
                            elif value.dtype == bool:
                                hetero[node_type][key] = torch.from_numpy(value)
                            else:
                                hetero[node_type][key] = torch.from_numpy(value)
                        elif isinstance(value, (int, np.integer)):
                            # Integer scalar converted to long tensor
                            hetero[node_type][key] = torch.tensor(value, dtype=torch.long)
                        else:
                            # Other scalar values assigned directly
                            hetero[node_type][key] = value
            
            # Convert edge type data
            for key in data_dict.keys():
                if isinstance(key, tuple):  # Edge type: ('centerline', 'lane') or ('lane', 'lane')
                    for edge_attr, edge_value in data_dict[key].items():
                        if isinstance(edge_value, np.ndarray):
                            # edge_index must be long type
                            if edge_value.dtype == np.int64 or edge_value.dtype == np.int32:
                                hetero[key][edge_attr] = torch.from_numpy(edge_value).long()
                            elif edge_value.dtype == np.float32 or edge_value.dtype == np.float64:
                                hetero[key][edge_attr] = torch.from_numpy(edge_value).float()
                            else:
                                hetero[key][edge_attr] = torch.from_numpy(edge_value)
                        else:
                            hetero[key][edge_attr] = edge_value
            
            # Save metadata (not put into node/edge structure)
            if 'scenario_id' in data_dict:
                hetero.scenario_id = data_dict['scenario_id']
            
            hetero = self._apply_lane_occlusion(hetero)
            
            hetero_list.append(hetero)
        
        # Automatic batching using PyG's Batch
        batch = Batch.from_data_list(hetero_list)
        
        return batch

