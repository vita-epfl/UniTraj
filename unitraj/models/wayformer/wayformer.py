import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from unitraj.models.base_model.base_model import BaseModel
from .wayformer_utils import PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Wayformer(BaseModel):
    '''
    Wayformer Class.
    '''

    def __init__(self, config):
        super(Wayformer, self).__init__(config)
        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.fisher_information = None
        self.map_attr = config['num_map_feature']
        self.k_attr = config['num_agent_feature']
        self.d_k = config['hidden_size']
        self._M = config['max_num_agents']  # num agents without the ego-agent
        self.c = config['num_modes']
        self.T = config['future_len']
        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.num_heads = config['tx_num_heads']
        self.L_dec = config['num_decoder_layers']
        self.tx_hidden_size = config['tx_hidden_size']
        self.use_map_img = config['use_map_image']
        self.use_map_lanes = config['use_map_lanes']
        self.past_T = config['past_len']
        self.max_points_per_lane = config['max_points_per_lane']
        self.max_num_roads = config['max_num_roads']
        self.num_queries_enc = config['num_queries_enc']
        self.num_queries_dec = config['num_queries_dec']

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        self.perceiver_encoder = PerceiverEncoder(self.num_queries_enc, self.d_k,
                                                  num_cross_attention_qk_channels=self.d_k,
                                                  num_cross_attention_v_channels=self.d_k,
                                                  num_self_attention_qk_channels=self.d_k,
                                                  num_self_attention_v_channels=self.d_k)

        output_query_provider = TrainableQueryProvider(
            num_queries=config['num_queries_dec'],
            num_query_channels=self.d_k,
            init_scale=0.1,
        )

        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )

        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )

        self.map_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.max_points_per_lane * self.max_num_roads, self.d_k)), requires_grad=True
        )

        self.perceiver_decoder = PerceiverDecoder(output_query_provider, self.d_k)

        self.prob_predictor = nn.Sequential(init_(nn.Linear(self.d_k, 1)))

        self.output_model = nn.Sequential(init_(nn.Linear(self.d_k, 5 * self.T)))

        self.relu = nn.ReLU(inplace=True)

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.num_queries_dec, 1).view(ego.shape[0] * self.num_queries_dec,
                                                                                   -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def _forward(self, inputs):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        ego_in, agents_in, roads = inputs['ego_in'], inputs['agents_in'], inputs['roads']

        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks_agents, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.relu(self.agents_dynamic_encoder(agents_tensor))
        agents_emb = (agents_emb + self.agents_positional_embedding[:, :,
                                   :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k)
        road_pts_feats = self.relu(self.road_pts_lin(roads[:, :self.max_num_roads, :, :self.map_attr]).view(B, -1,
                                                                                                            self.d_k)) + self.map_positional_embedding
        mixed_input_features = torch.concat([agents_emb, road_pts_feats], dim=1)
        opps_masks_roads = (1.0 - roads[:, :self.max_num_roads, :, -1]).to(torch.bool)
        mixed_input_masks = torch.concat([opps_masks_agents.view(B, -1), opps_masks_roads.view(B, -1)], dim=1)
        # Process through Wazformer's encoder

        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)

        # Wazformer-Ego Decoding

        out_seq = self.perceiver_decoder(context)

        out_dists = self.output_model(out_seq).reshape(B, self.num_queries_dec, self.T, -1)

        # Mode prediction

        mode_probs = F.softmax(self.prob_predictor(out_seq).reshape(B, self.num_queries_dec), dim=1)

        # return  [c, T, B, 5], [B, c]
        output = {}
        output['predicted_probability'] = mode_probs  # #[B, c]
        output['predicted_trajectory'] = out_dists  # [B, c, T, 5] to be able to parallelize code
        output['scene_emb'] = out_seq[:, :self.num_queries_dec].reshape(B, -1)
        if len(np.argwhere(np.isnan(out_dists.detach().cpu().numpy()))) > 1:
            breakpoint()
        return output

    def forward(self, batch):
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                       -2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       agents_mask.shape[
                                                                                                           -1])).squeeze(
            1)
        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)
        roads = torch.cat([inputs['map_polylines'], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)
        model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads
        output = self._forward(model_input)

        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        loss = self.criterion(output, ground_truth, inputs['center_gt_final_valid_idx'])
        output['dataset_name'] = inputs['dataset_name']
        return output, loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)

        prediction['predicted_trajectory'], prediction['predicted_probability'], selected_idxs = self.batch_nms(
            pred_trajs=prediction['predicted_trajectory'], pred_scores=prediction['predicted_probability'],
            dist_thresh=2.5,
            num_ret_modes=self.c
        )
        prediction['predicted_probability'] = F.softmax(prediction['predicted_probability'], dim=1)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, prediction, status='val')
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0002, steps_per_epoch=1, epochs=150,
                                                        pct_start=0.04, div_factor=25.0, final_div_factor=100,
                                                        anneal_strategy='linear')

        return [optimizer], [scheduler]

    def batch_nms(self, pred_trajs, pred_scores, dist_thresh, num_ret_modes=6):
        """

        Args:
            pred_trajs (batch_size, num_modes, num_timestamps, 7)
            pred_scores (batch_size, num_modes):
            dist_thresh (float):
            num_ret_modes (int, optional): Defaults to 6.

        Returns:
            ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
            ret_scores (batch_size, num_ret_modes)
            ret_idxs (batch_size, num_ret_modes)
        """
        batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

        sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
        bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
        sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
        sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
        sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

        dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
        point_cover_mask = (dist < dist_thresh)

        point_val = sorted_pred_scores.clone()  # (batch_size, N)
        point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

        ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
        ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
        ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
        bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

        for k in range(num_ret_modes):
            cur_idx = point_val.argmax(dim=-1)  # (batch_size)
            ret_idxs[:, k] = cur_idx

            new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
            point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
            point_val_selected[bs_idxs, cur_idx] = -1
            point_val += point_val_selected

            ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
            ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

        bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes)

        ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
        return ret_trajs, ret_scores, ret_idxs


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config
        pass

    def forward(self, out, gt, center_gt_final_valid_idx):
        return self.nll_loss_gmm_direct(out['predicted_probability'], out['predicted_trajectory'], gt,
                                        center_gt_final_valid_idx)

    def nll_loss_gmm_direct(self, pred_scores, pred_trajs, gt_trajs, center_gt_final_valid_idx,
                            pre_nearest_mode_idxs=None,
                            timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0),
                            rho_limit=0.5):
        """
        GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
        Written by Shaoshuai Shi

        Args:
            pred_scores (batch_size, num_modes):
            pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs (batch_size, num_timestamps, 3):
            timestamp_loss_weight (num_timestamps):
        """
        if use_square_gmm:
            assert pred_trajs.shape[-1] == 3
        else:
            assert pred_trajs.shape[-1] == 5

        batch_size = pred_trajs.shape[0]

        gt_valid_mask = gt_trajs[..., -1]

        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
        else:
            distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

            nearest_mode_idxs = distance.argmin(dim=-1)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
        res_trajs = gt_trajs[..., :2] - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if use_square_gmm:
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

        loss_cls = (F.cross_entropy(input=pred_scores, target=nearest_mode_idxs, reduction='none'))

        return (reg_loss + loss_cls).mean()
