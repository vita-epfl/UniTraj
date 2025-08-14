import json

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

import unitraj.datasets.common_utils as common_utils
import unitraj.utils.visualization as visualization


class BaseModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pred_dicts = []

        if config.get('eval_nuscenes', False):
            self.init_nuscenes()

    def init_nuscenes(self):
        if self.config.get('eval_nuscenes', False):
            from nuscenes import NuScenes

            from nuscenes.eval.prediction.config import PredictionConfig

            from nuscenes.prediction import PredictHelper
            nusc = NuScenes(version='v1.0-trainval', dataroot=self.config['nuscenes_dataroot'])

            # Prediction helper and configs:
            self.helper = PredictHelper(nusc)

            with open('models/base_model/nuscenes_config.json', 'r') as f:
                pred_config = json.load(f)
            self.pred_config5 = PredictionConfig.deserialize(pred_config, self.helper)

    def forward(self, batch):
        """
        Forward pass for the model
        :param batch: input batch
        :return: prediction: {
                'predicted_probability': (batch_size,modes)),
                'predicted_trajectory': (batch_size,modes, future_len, 2)
                }
                loss (with gradient)
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status='train')
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status='val')
        return loss

    def predict(self, batch):
        prediction, loss = self.forward(batch)
        return prediction


    def on_validation_epoch_end(self):
        if self.config.get('eval_waymo', False):
            metric_results, result_format_str = self.compute_metrics_waymo(self.pred_dicts)
            print(metric_results)
            print(result_format_str)

        elif self.config.get('eval_nuscenes', False):
            import os
            os.makedirs('submission', exist_ok=True)
            json.dump(self.pred_dicts, open(os.path.join('submission', "evalai_submission.json"), "w"))
            metric_results = self.compute_metrics_nuscenes(self.pred_dicts)
            print('\n', metric_results)
            
        elif self.config.get('eval_argoverse2', False):
            metric_results = self.compute_metrics_av2(self.pred_dicts)
            
        self.pred_dicts = []

    def configure_optimizers(self):
        raise NotImplementedError

    def compute_metrics_nuscenes(self, pred_dicts):
        from nuscenes.eval.prediction.compute_metrics import compute_metrics
        metric_results = compute_metrics(pred_dicts, self.helper, self.pred_config5)
        return metric_results

    def compute_metrics_waymo(self, pred_dicts):
        from unitraj.models.base_model.waymo_eval import waymo_evaluation
        try:
            num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
        except:
            num_modes_for_eval = 6
        metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts,
                                                             num_modes_for_eval=num_modes_for_eval)

        metric_result_str = '\n'
        for key in metric_results:
            metric_results[key] = metric_results[key]
            metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str

        return metric_result_str, metric_results

    def compute_metrics_av2(self, pred_dicts):
        from unitraj.models.base_model.av2_eval import argoverse2_evaluation
        try:
            num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
        except:
            num_modes_for_eval = 6
        metric_results = argoverse2_evaluation(pred_dicts=pred_dicts,
                                               num_modes_for_eval=num_modes_for_eval)
        self.log('val/av2_official_minADE6', metric_results['min_ADE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_minFDE6', metric_results['min_FDE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_brier_minADE', metric_results['brier_min_ADE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_brier_minFDE', metric_results['brier_min_FDE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_miss_rate', metric_results['miss_rate'], prog_bar=True, on_step=False, on_epoch=True)
        
        # metric_result_str = '\n'
        # for key, value in metric_results.items():
        #     metric_result_str += '%s: %.4f\n' % (key, value)
        # metric_result_str += '\n'
        # print(metric_result_str)
        return metric_results
        
    def compute_official_evaluation(self, batch_dict, prediction):
        if self.config.get('eval_waymo', False):

            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]

            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][bs_idx],
                    'pred_trajs': pred_trajs_world[bs_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[bs_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][bs_idx],
                    'object_type': input_dict['center_objects_type'][bs_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][bs_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][bs_idx].cpu().numpy()
                }
                pred_dict_list.append(single_pred_dict)

            assert len(pred_dict_list) == batch_dict['batch_size']

            self.pred_dicts += pred_dict_list

        elif self.config.get('eval_nuscenes', False):
            from nuscenes.eval.prediction.data_classes import Prediction
            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
            # assert num_feat == 7

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]
            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'instance': input_dict['scenario_id'][bs_idx].split('_')[1],
                    'sample': input_dict['scenario_id'][bs_idx].split('_')[2],
                    'prediction': pred_trajs_world[bs_idx, :, 4::5, 0:2].cpu().numpy(),
                    'probabilities': pred_scores[bs_idx, :].cpu().numpy(),
                }

                pred_dict_list.append(
                    Prediction(instance=single_pred_dict["instance"], sample=single_pred_dict["sample"],
                               prediction=single_pred_dict["prediction"],
                               probabilities=single_pred_dict["probabilities"]).serialize())

            self.pred_dicts += pred_dict_list
        
        elif self.config.get('eval_argoverse2', False):

            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]

            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][bs_idx],
                    'pred_trajs': pred_trajs_world[bs_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[bs_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][bs_idx],
                    'object_type': input_dict['center_objects_type'][bs_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][bs_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][bs_idx].cpu().numpy()
                }
                pred_dict_list.append(single_pred_dict)

            assert len(pred_dict_list) == batch_dict['batch_size']

            self.pred_dicts += pred_dict_list

    def log_info(self, batch, batch_idx, prediction, status='train'):
        ## logging
        # Split based on dataset
        inputs = batch['input_dict']
        gt_traj = inputs['center_gt_trajs'].unsqueeze(1)  # .transpose(0, 1).unsqueeze(0)
        gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)
        center_gt_final_valid_idx = inputs['center_gt_final_valid_idx']

        predicted_traj = prediction['predicted_trajectory']
        predicted_prob = prediction['predicted_probability'].detach().cpu().numpy()

        # Calculate ADE losses
        ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
        ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
        ade_losses = ade_losses.cpu().detach().numpy()
        minade = np.min(ade_losses, axis=1)
        # Calculate FDE losses
        bs, modes, future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)

        fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
        minfde = np.min(fde, axis=-1)

        best_fde_idx = np.argmin(fde, axis=-1)
        predicted_prob = predicted_prob[np.arange(bs), best_fde_idx]
        miss_rate = (minfde > 2.0)
        brier_fde = minfde + np.square(1 - predicted_prob)

        loss_dict = {
            'minADE6': minade,
            'minFDE6': minfde,
            'miss_rate': miss_rate.astype(np.float32),
            'brier_fde': brier_fde}

        important_metrics = list(loss_dict.keys())

        new_dict = {}
        dataset_names = inputs['dataset_name']
        unique_dataset_names = np.unique(dataset_names)
        for dataset_name in unique_dataset_names:
            batch_idx_for_this_dataset = np.argwhere([n == str(dataset_name) for n in dataset_names])[:, 0]
            for key in loss_dict.keys():
                new_dict[dataset_name + '/' + key] = loss_dict[key][batch_idx_for_this_dataset]

        # merge new_dict with log_dict
        loss_dict.update(new_dict)
        # loss_dict.update(avg_dict)

        if status == 'val' and self.config.get('eval', False):

            # Split scores based on trajectory type
            new_dict = {}
            trajectory_types = inputs["trajectory_type"].cpu().numpy()
            trajectory_correspondance = {0: "stationary", 1: "straight", 2: "straight_right",
                                         3: "straight_left", 4: "right_u_turn", 5: "right_turn",
                                         6: "left_u_turn", 7: "left_turn"}
            for traj_type in range(8):
                batch_idx_for_traj_type = np.where(trajectory_types == traj_type)[0]
                if len(batch_idx_for_traj_type) > 0:
                    for key in important_metrics:
                        new_dict["traj_type/" + trajectory_correspondance[traj_type] + "_" + key] = loss_dict[key][
                            batch_idx_for_traj_type]
            loss_dict.update(new_dict)

            # Split scores based on kalman_difficulty @6s
            new_dict = {}
            kalman_difficulties = inputs["kalman_difficulty"][:,
                                  -1].cpu().numpy()  # Last is difficulty at 6s (others are 2s and 4s)
            for kalman_bucket, (low, high) in {"easy": [0, 30], "medium": [30, 60], "hard": [60, 9999999]}.items():
                batch_idx_for_kalman_diff = \
                    np.where(np.logical_and(low <= kalman_difficulties, kalman_difficulties < high))[0]
                if len(batch_idx_for_kalman_diff) > 0:
                    for key in important_metrics:
                        new_dict["kalman/" + kalman_bucket + "_" + key] = loss_dict[key][batch_idx_for_kalman_diff]
            loss_dict.update(new_dict)

            new_dict = {}
            agent_types = [1, 2, 3]
            agent_type_dict = {1: "vehicle", 2: "pedestrian", 3: "bicycle"}
            for type in agent_types:
                batch_idx_for_type = np.where(inputs['center_objects_type'] == type)[0]
                if len(batch_idx_for_type) > 0:
                    for key in important_metrics:
                        new_dict["agent_types" + '/' + agent_type_dict[type] + "_" + key] = loss_dict[key][
                            batch_idx_for_type]
            # merge new_dict with log_dict
            loss_dict.update(new_dict)

        # Take mean for each key but store original length before (useful for aggregation)
        size_dict = {key: len(value) for key, value in loss_dict.items()}
        loss_dict = {key: np.mean(value) for key, value in loss_dict.items()}

        for k, v in loss_dict.items():
            self.log(status + "/" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=size_dict[k])

        # if self.local_rank == 0 and status == 'val' and batch_idx == 0:
        #     img = visualization.visualize_prediction(batch, prediction)
        #     wandb.log({"prediction": [wandb.Image(img)]})

        return

    def convert_to_fmae_format(self, input):
        B = len(input['center_objects_id']) #number of interested agents/scenarios
        N_agent = self.config["max_num_agents"]#10 #maximum number of agents per scenari --> get from config file
        N_lane = self.config["max_num_roads"] #80 #maximum number of lane segments per scenario --> get from config file
        h_steps = self.config["past_len"]#50 #num historical timesteps for agent tracks --> get from config file
        f_steps = self.config["future_len"] #60 #num future timesteps for agent tracks --> get from config file
        lane_sampling_pts = self.config["max_points_per_lane"] #TODO:not sure if this is the right value #20 #sampling points per lane segment --> get from config file
        data = {
            "x": torch.rand(B, N_agent, h_steps, 2), #agent tracks as local differences from origin being the last position of the past trajectory
            "x_attr": torch.zeros((B, N_agent, 3), dtype=torch.uint8), #categorical agent attributes
            "x_positions": torch.rand(B, N_agent, h_steps, 2), #agent tracks in scene coordinates
            "x_centers": torch.rand(B, N_agent, 2), #center of agent track
            "x_angles": torch.rand(B, N_agent, h_steps+f_steps), #agent headings
            "x_velocity": torch.rand(B, N_agent, h_steps+f_steps), #velocity of agents as absolute values
            "x_velocity_diff": torch.rand(B, N_agent, h_steps), #velocity changes of agents
            "lane_positions": torch.rand(B, N_lane, lane_sampling_pts, 2), #lane segments in scene coordinates
            "lane_centers": torch.rand(B, N_lane, 2), #center of lane segments
            "lane_angles": torch.rand(B, N_lane), #orientation of lane segments
            "lane_attr": torch.rand(B, N_lane, 3), #categorial lane attributes
            "is_intersections": torch.rand(B, N_lane), # categorical lane attribute
            "y": torch.rand(B, N_agent, f_steps, 2), #agent future tracks as x,y positions -> ground truth in unitraj format
            "x_padding_mask": torch.zeros((B, N_agent, h_steps+f_steps), dtype=torch.bool), #padding mask for agent tracks
            "lane_padding_mask": torch.zeros((B, N_lane, lane_sampling_pts), dtype=torch.bool), #padding mask for lane segment points
            "x_key_padding_mask": torch.zeros((B, N_agent), dtype=torch.bool), #batch padding mask for agent tracks
            "lane_key_padding_mask": torch.zeros((B, N_lane), dtype=torch.bool), #batch padding mask for lane segments
            "num_actors": torch.full((B,), fill_value=N_agent, dtype=torch.int64),
            "num_lanes": torch.full((B,), fill_value=N_lane, dtype=torch.int64),
            "scenario_id": [] * B,
            "track_id": [] * B,
            "origin": torch.rand(B, 2), #scene to global coordinates position
            "theta": torch.rand(B), #scene to global coordinates orientation
        }
        if not self.config["debug"]:
            for key, value in input.items():
                if isinstance(value, torch.Tensor):
                        input[key] = input[key].to("cpu")

        data["origin"] = input["center_objects_world"][..., 0:2].clone()
        data["theta"] = input["center_objects_world"][..., 6].clone()

        data["x"][..., 1:h_steps, 0:2] = torch.where(
            (~input["obj_trajs_mask"][..., :(h_steps - 1)] | ~input["obj_trajs_mask"][..., 1:h_steps]).unsqueeze(-1),
            torch.zeros(B, N_agent, h_steps - 1, 2),
            input["obj_trajs_pos"][..., 1:h_steps, 0:2] - input["obj_trajs_pos"][..., :(h_steps - 1), 0:2],
        )
        data["x"][... , 0, 0:2] = torch.zeros(B, N_agent, 2)

        #data["x_attr"] -> object type | object category | object type combined -> only third attribute is used in model
        data["x_attr"][:, :, 0] = 9 #unknown
        for center_idx, center_obj in enumerate(input["obj_trajs"]):
            for actor_idx, actor in enumerate(center_obj):
                if actor[h_steps - 1][6] == 1:
                    data["x_attr"][center_idx, actor_idx, 0] = 0 #vehicle
                    data["x_attr"][center_idx, actor_idx, 2] = 0 
                elif actor[h_steps - 1][7] == 1:
                    data["x_attr"][center_idx, actor_idx, 0] = 1 #pedestrian
                    data["x_attr"][center_idx, actor_idx, 2] = 1
                elif actor[h_steps - 1][8] == 1:
                    data["x_attr"][center_idx, actor_idx, 0] = 3 #bicycle
                    data["x_attr"][center_idx, actor_idx, 2] = 2
                else:
                    data["x_attr"][center_idx, actor_idx, 0] = 9 #unknown
                    data["x_attr"][center_idx, actor_idx, 2] = 3

                if actor[h_steps - 1][9] == 1:
                    data["x_attr"][center_idx, actor_idx, 1] = 3 #FOCAL_TRACK -> track that is being predicted
                if actor[h_steps - 1][10] == 1:
                    data["x_attr"][center_idx, actor_idx, 1] = 3 #FOCAL_TRACK -> track that is being predicted & is the sdc track
                if actor[h_steps - 1][9] == 0 and actor[0][10] == 0:
                    data["x_attr"][center_idx, actor_idx, 1] = 2 #SCORED_TRACK -> track that is being scored -> TODO are there unscored tracks?
        data["x_positions"] = input["obj_trajs"][..., :h_steps, :2].clone()#torch.matmul(data["x"][..., 0:2] + data["origin"], rotate_mat)[..., :h_steps, :2].clone()
        data["x_centers"] = input["obj_trajs"][..., h_steps - 1, :2].clone() 
        data["x_angles"] = np.arcsin(input["obj_trajs"][..., h_steps+12].float())
        data["x_velocity"] = torch.cat(
            (torch.from_numpy(np.linalg.norm(input["obj_trajs"][..., h_steps+14:h_steps+16], axis=-1)), 
             torch.from_numpy(np.linalg.norm(input["obj_trajs_future_state"][..., 2:], axis=-1))), 
             dim=-1)#np.linalg.norm(input["obj_trajs"][..., 35:37], axis=-1)#torch.cat((np.linalg.norm(input["obj_trajs"][..., 35:37], axis=-1), np.linalg.norm(input["obj_trajs_future_state"][..., 2:], axis=-1)), dim=-1)
        data["lane_positions"] = input["map_polylines"][..., :2].clone()
        data["lane_centers"] = data["lane_positions"][:, :, ((lane_sampling_pts//2)-1):(lane_sampling_pts//2)+1].mean(dim=-2)
        data["lane_angles"] = torch.atan2(
             data["lane_positions"][..., lane_sampling_pts//2, 1] -  data["lane_positions"][..., (lane_sampling_pts//2)-1, 1],
             data["lane_positions"][..., lane_sampling_pts//2, 0] -  data["lane_positions"][..., (lane_sampling_pts//2)-1, 0],
        )

        #data["lane_attr"] -> not used in model
        #data["is_intersections"] -> not used in model

        data["y"] = torch.where(
            (~input["obj_trajs_mask"][..., (h_steps - 1)].unsqueeze(-1) | ~input["obj_trajs_future_mask"][..., :]).unsqueeze(-1),
            torch.zeros(B, N_agent, f_steps, 2),
            input["obj_trajs_future_state"][..., :, 0:2] - input["obj_trajs"][..., (h_steps - 1), 0:2].unsqueeze(-2),
        )#input["obj_trajs_future_state"][..., :2].clone()
        data["x_padding_mask"] = torch.cat((~input["obj_trajs_mask"], ~input["obj_trajs_future_mask"]), dim=-1)
        data["lane_padding_mask"] = ~input["map_polylines_mask"].clone()
        data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
        data["num_actors"] =  (~data["x_key_padding_mask"]).sum(-1)
        data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)
        data["scenario_id"] = input["scenario_id"]

        data["x_padding_mask"] = torch.where(
            torch.from_numpy(np.linalg.norm(data["x_positions"][..., h_steps - 1, :], axis=-1) > 150).unsqueeze(-1),
            torch.ones_like(data["x_padding_mask"]),
            data["x_padding_mask"]
        )
        data["x_padding_mask"] = torch.where(
            (data["x_padding_mask"][..., h_steps - 1]).unsqueeze(-1),
            torch.ones_like(data["x_padding_mask"]),
            data["x_padding_mask"]
        )
        data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)

        data["x_velocity_diff"][..., 1:h_steps] = torch.where(
            (~input["obj_trajs_mask"][..., :(h_steps - 1)] | ~input["obj_trajs_mask"][..., 1:h_steps]),
            torch.zeros(B, N_agent, h_steps-1),
            data["x_velocity"][..., 1:h_steps] - data["x_velocity"][..., :(h_steps - 1)],
        )
        data["x_velocity_diff"][..., 0] = torch.zeros(N_agent)


        #data["track_id"] information not present in input -> only used for av2 submission

        if not self.config["debug"]:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                        data[key] = data[key].cuda()
        return data
    
    def convert_to_qcnet_format(self, input):
        return