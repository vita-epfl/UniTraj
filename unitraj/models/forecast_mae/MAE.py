from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from unitraj.models.forecast_mae.layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block

from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.forecast_mae.utils.optim import WarmupCosLR

class MaskedAutoEncoder(BaseModel):

    def __init__(self, config):
        super(MaskedAutoEncoder, self).__init__(config)
        self.config = config
        self.embed_dim = config['embed_dim']
        self.encoder_depth = config['encoder_depth']
        self.decoder_depth= config['decoder_depth']
        self.num_heads = config['num_heads']
        self.mlp_ratio = config['mlp_ratio']
        self.qkv_bias = config['qkv_bias']
        self.drop_path = config['drop_path']
        self.actor_mask_ratio: float = config['actor_mask_ratio']
        self.lane_mask_ratio: float = config['lane_mask_ratio']
        self.past_len: int = config['past_len']
        self.future_len: int = config['future_len']
        self.loss_weight: List[float] = config['loss_weight']
        self.num_agent_feature = config['num_agent_feature']
        self.num_map_feature = config['num_map_feature']
        self.num_fut_feature = config['num_fut_feature']

        self.hist_embed = AgentEmbeddingLayer(4, 32, drop_path_rate=self.drop_path)
        self.future_embed = AgentEmbeddingLayer(3, 32, drop_path_rate=self.drop_path)
        self.lane_embed = LaneEmbeddingLayer(3, self.embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(self.encoder_depth)
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(self.decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(self.embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(1, 1, self.embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, self.embed_dim))

        self.lane_mask_token = nn.Parameter(torch.Tensor(1, 1, self.embed_dim))
        self.future_mask_token = nn.Parameter(torch.Tensor(1, 1, self.embed_dim))
        self.history_mask_token = nn.Parameter(torch.Tensor(1, 1, self.embed_dim))

        self.future_pred = nn.Linear(self.embed_dim, self.future_len * self.num_fut_feature)
        self.history_pred = nn.Linear(self.embed_dim, self.past_len * 2)
        self.lane_pred = nn.Linear(self.embed_dim, 20 * 2)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
        nn.init.normal_(self.future_mask_token, std=0.02)
        nn.init.normal_(self.lane_mask_token, std=0.02)
        nn.init.normal_(self.history_mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def agent_random_masking(
        hist_tokens, fut_tokens, mask_ratio, future_padding_mask, num_actors
    ):
        pred_masks = ~future_padding_mask.all(-1)  # [B, A]
        fut_num_tokens = pred_masks.sum(-1)  # [B]

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int()
        hist_masked_tokens, fut_masked_tokens = [], []
        hist_keep_ids_list, fut_keep_ids_list = [], []
        hist_key_padding_mask, fut_key_padding_mask = [], []

        device = hist_tokens.device
        agent_ids = torch.arange(hist_tokens.shape[1], device=device)
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
            zip(fut_num_tokens, len_keeps, pred_masks)
        ):
            pred_agent_ids = agent_ids[future_pred_mask]
            noise = torch.rand(fut_num_token, device=device)
            ids_shuffle = torch.argsort(noise)
            fut_ids_keep = ids_shuffle[:len_keep]
            fut_ids_keep = pred_agent_ids[fut_ids_keep]
            fut_keep_ids_list.append(fut_ids_keep)

            hist_keep_mask = torch.zeros_like(agent_ids).bool()
            hist_keep_mask[: num_actors[i]] = True
            hist_keep_mask[fut_ids_keep] = False
            hist_ids_keep = agent_ids[hist_keep_mask]
            hist_keep_ids_list.append(hist_ids_keep)

            fut_masked_tokens.append(fut_tokens[i, fut_ids_keep])
            hist_masked_tokens.append(hist_tokens[i, hist_ids_keep])

            fut_key_padding_mask.append(torch.zeros(len_keep, device=device))
            hist_key_padding_mask.append(torch.zeros(len(hist_ids_keep), device=device))

        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        hist_masked_tokens = pad_sequence(hist_masked_tokens, batch_first=True)
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_key_padding_mask = pad_sequence(
            hist_key_padding_mask, batch_first=True, padding_value=True
        )

        return (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        )

    @staticmethod
    def lane_random_masking(x, future_mask_ratio, key_padding_mask):
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        x_masked, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)
            x_masked.append(x[i, ids_keep])
            new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))

        x_masked = pad_sequence(x_masked, batch_first=True)
        new_key_padding_mask = pad_sequence(
            new_key_padding_mask, batch_first=True, padding_value=True
        )

        return x_masked, new_key_padding_mask, ids_keep_list

    def convert(self, batch):
        input_dict = batch['input_dict']
        input_dict['x'] = input_dict['obj_trajs'][..., :2]

        input_dict['x_padding_mask'] = ~(
            torch.cat([input_dict['obj_trajs_mask'], input_dict['obj_trajs_future_mask']], dim=-1).to(torch.bool))
        input_dict['x_key_padding_mask'] = input_dict['x_padding_mask'].all(-1)

        input_dict['lane_padding_mask'] = ~input_dict['map_polylines_mask']
        input_dict['lane_key_padding_mask'] = input_dict['lane_padding_mask'].all(-1)

        input_dict['lane_positions'] = input_dict['map_polylines'][..., :2]
        input_dict['lane_centers'] = (input_dict['lane_positions']*input_dict['map_polylines_mask'].unsqueeze(-1)).mean(dim=2)
        input_dict['lane_angles'] = torch.atan2(
            input_dict['lane_positions'][:, :, 10, 1] - input_dict['lane_positions'][:, :, 9, 1],
            input_dict['lane_positions'][:, :, 10, 0] - input_dict['lane_positions'][:, :, 9, 0],
        )

        input_dict['x_velocity_diff'] = input_dict['obj_trajs'][..., 35:37]
        input_dict['x_centers'] = input_dict['obj_trajs_last_pos'][..., :2]
        input_dict['x_angles'] = torch.norm(input_dict['obj_trajs'][..., -1, 33:35], dim=-1)
        input_dict['y'] = input_dict['obj_trajs_future_state'][...,:2]
        input_dict["num_actors"] = (~input_dict["x_key_padding_mask"]).sum(-1)
        input_dict["num_lanes"] = (~input_dict["lane_key_padding_mask"]).sum(-1)

        input_dict['x_positions'] = input_dict['obj_trajs_pos'][..., :2]

        velocity = input_dict['obj_trajs'][..., 35:37]

        velocity_norm = torch.norm(velocity, dim=-1)

        input_dict['x_velocity'] = velocity_norm

        return batch

    def forward(self, input_batch, batch_idx=None):
        batch = self.convert(input_batch)
        data = batch['input_dict']
        hist_padding_mask = data["x_padding_mask"][:, :, :self.past_len]
        hist_feat = torch.cat(
            [
                data['obj_trajs'][...,:2],
                data['x_velocity'].unsqueeze(-1),
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )
        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat = self.hist_embed(hist_feat.permute(0, 2, 1).contiguous())
        hist_feat = hist_feat.view(B, N, hist_feat.shape[-1])

        future_padding_mask = data["x_padding_mask"][:, :, self.past_len:]
        future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1)
        B, N, L, D = future_feat.shape
        future_feat = future_feat.view(B * N, L, D)
        future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
        future_feat = future_feat.view(B, N, future_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        data['map_polylines'][..., :2] = lane_normalized
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)
        B, M, L, D = lane_feat.shape
        lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"], data["x_angles"], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        lane_feat += lane_type_embed

        hist_feat += pos_embed[:, :N]
        lane_feat += pos_embed[:, -M:]
        future_feat += pos_embed[:, N : N + N]

        (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        ) = self.agent_random_masking(
            hist_feat,
            future_feat,
            self.actor_mask_ratio,
            future_padding_mask,
            data["num_actors"],
        )

        lane_mask_ratio = self.lane_mask_ratio
        (
            lane_masked_tokens,
            lane_key_padding_mask,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            lane_feat, lane_mask_ratio, data["lane_key_padding_mask"]
        )

        x = torch.cat(
            [hist_masked_tokens, fut_masked_tokens, lane_masked_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [hist_key_padding_mask, fut_key_padding_mask, lane_key_padding_mask],
            dim=1,
        )

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # decoding
        x_decoder = self.decoder_embed(x)
        Nh, Nf, Nl = (
            hist_masked_tokens.shape[1],
            fut_masked_tokens.shape[1],
            lane_masked_tokens.shape[1],
        )
        assert x_decoder.shape[1] == Nh + Nf + Nl
        hist_tokens = x_decoder[:, :Nh]
        fut_tokens = x_decoder[:, Nh : Nh + Nf]
        lane_tokens = x_decoder[:, -Nl:]

        decoder_hist_token = self.history_mask_token.repeat(B, N, 1)
        hist_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(hist_keep_ids_list):
            decoder_hist_token[i, idx] = hist_tokens[i, : len(idx)]
            hist_pred_mask[i, idx] = False

        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False

        decoder_lane_token = self.lane_mask_token.repeat(B, M, 1)
        lane_pred_mask = ~data["lane_key_padding_mask"]
        for i, idx in enumerate(lane_ids_keep_list):
            decoder_lane_token[i, idx] = lane_tokens[i, : len(idx)]
            lane_pred_mask[i, idx] = False

        x_decoder = torch.cat(
            [decoder_hist_token, decoder_fut_token, decoder_lane_token], dim=1
        )

        x_decoder = x_decoder + self.decoder_pos_embed(pos_feat)
        decoder_key_padding_mask = torch.cat(
            [
                data["x_key_padding_mask"],
                future_padding_mask.all(-1),
                data["lane_key_padding_mask"],
            ],
            dim=1,
        )

        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        hist_token = x_decoder[:, :N].reshape(-1, self.embed_dim)
        future_token = x_decoder[:, N : 2 * N].reshape(-1, self.embed_dim)
        lane_token = x_decoder[:, -M:]

        # lane pred loss
        lane_pred = self.lane_pred(lane_token).view(B, M, 20, 2)
        lane_reg_mask = ~lane_padding_mask
        lane_pred_loss = (F.mse_loss(
            lane_pred, lane_normalized, reduction='none').mean(-1)*lane_reg_mask).mean(-1).mean(-1)

        # hist pred loss
        x_hat = self.history_pred(hist_token).view(B,-1, self.past_len, 2)
        x = (data["x_positions"] - data["x_centers"].unsqueeze(-2))
        x_reg_mask = ~data["x_padding_mask"][:, :, :self.past_len]
        hist_loss = (F.l1_loss(x_hat, x, reduction='none').mean(-1)*x_reg_mask).mean(-1).mean(-1)

        # future pred loss
        y_hat = self.future_pred(future_token).view(B,-1, self.future_len, self.num_fut_feature)  # B*N, 120
        y = data["y"]
        reg_mask = ~data["x_padding_mask"][:, :, self.past_len:]
        future_loss = (F.l1_loss(y_hat, y, reduction='none').mean(-1)*reg_mask).mean(-1).mean(-1)

        loss = (
            self.loss_weight[0] * future_loss.mean()
            + self.loss_weight[1] * hist_loss.mean()
            + self.loss_weight[2] * lane_pred_loss.mean()
        )

        out = {}
        out['predicted_probability'] = torch.ones((B, 1))
        out['predicted_trajectory'] = y_hat.view(B, N, self.future_len, self.num_fut_feature)[:,:1,:,:2]
        out["hist_loss"] = hist_loss.cpu().detach().numpy()
        out["future_loss"] = future_loss.cpu().detach().numpy()
        out["lane_pred_loss"] = lane_pred_loss.cpu().detach().numpy()

        if not self.training:
            out["x_hat"] = x_hat.view(B, N, self.past_len, 2)
            out["y_hat"] = y_hat.view(1, B, N, self.future_len, self.num_fut_feature)
            out["lane_hat"] = lane_pred.view(B, M, 20, 2)
            out["lane_keep_ids"] = lane_ids_keep_list
            out["hist_keep_ids"] = hist_keep_ids_list
            out["fut_keep_ids"] = fut_keep_ids_list

        return out, loss

    def log_info(self, batch, batch_idx, prediction, status='train'):
        ## logging
        # Split based on dataset
        inputs = batch['input_dict']
        gt_traj = inputs['center_gt_trajs'].unsqueeze(1)#.transpose(0, 1).unsqueeze(0)
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
        bs,modes,future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1,1,1).repeat(1,modes,1).to(torch.int64)

        fde = torch.gather(ade_diff,-1,center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
        minfde = np.min(fde, axis=-1)

        best_fde_idx = np.argmin(fde, axis=-1)
        predicted_prob = predicted_prob[np.arange(bs),best_fde_idx]
        miss_rate = (minfde > 2.0)
        brier_fde = minfde + (1 - predicted_prob)

        loss_dict = {
                    'minADE6': minade,
                    'minFDE6': minfde,
                    'miss_rate': miss_rate.astype(np.float32),
                    'brier_fde': brier_fde,
            'reconstruction_loss_hist':prediction["hist_loss"],
            'reconstruction_loss_future': prediction["future_loss"],
            'reconstruction_loss_lane': prediction["lane_pred_loss"]
                    }


        important_metrics = loss_dict.keys()

        new_dict = {}
        dataset_names = inputs['dataset_name']
        unique_dataset_names = np.unique(dataset_names)
        for dataset_name in unique_dataset_names:
            batch_idx_for_this_dataset = np.argwhere([n == str(dataset_name) for n in dataset_names])[:,0]
            for key in loss_dict.keys():
                new_dict[dataset_name+'/'+key] = loss_dict[key][batch_idx_for_this_dataset]
        loss_dict.update(new_dict)

        if status == 'val' and self.config.get('eval', False):


            # Split scores based on trajectory type
            new_dict = {}
            trajectory_types = inputs["trajectory_type"]
            trajectory_correspondance = {0: "stationary", 1: "straight", 2: "straight_right",
                3: "straight_left", 4: "right_u_turn", 5: "right_turn",
                6: "left_u_turn", 7: "left_turn"}
            for traj_type in range(8):
                batch_idx_for_traj_type = np.where(trajectory_types == traj_type)[0]
                if len(batch_idx_for_traj_type) > 0:
                    for key in important_metrics:
                        new_dict["traj_type/" + trajectory_correspondance[traj_type] + "_" + key] = loss_dict[key][batch_idx_for_traj_type]
            loss_dict.update(new_dict)

            # Split scores based on kalman_difficulty @6s
            new_dict = {}
            kalman_difficulties = inputs["kalman_difficulty"][:, -1] # Last is difficulty at 6s (others are 2s and 4s)
            for kalman_bucket, (low, high) in {"easy": [0, 30], "medium": [30, 60], "hard": [60, 9999999]}.items():
                batch_idx_for_kalman_diff = np.where(np.logical_and(low <= kalman_difficulties, kalman_difficulties < high))[0]
                if len(batch_idx_for_kalman_diff) > 0:
                    for key in important_metrics:
                        new_dict["kalman/" + kalman_bucket + "_" + key] = loss_dict[key][batch_idx_for_kalman_diff]
            loss_dict.update(new_dict)

            new_dict = {}
            agent_types = [1,2,3]
            agent_type_dict = {1: "vehicle", 2: "pedestrian", 3: "bicycle"}
            for type in agent_types:
                batch_idx_for_type = np.where(inputs['center_objects_type'] == type)[0]
                if len(batch_idx_for_type) > 0:
                    for key in important_metrics:
                        new_dict["agent_types"+'/'+agent_type_dict[type]+"_"+key] = loss_dict[key][batch_idx_for_type]
            # merge new_dict with log_dict
            loss_dict.update(new_dict)


        # Take mean for each key but store original length before (useful for aggregation)
        size_dict = {key: len(value) for key, value in loss_dict.items()}
        loss_dict = {key: np.mean(value) for key, value in loss_dict.items()}


        for k, v in loss_dict.items():
            self.log(status+"/" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=size_dict[k])
        return

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay']
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.config['learning_rate'],
            min_lr=self.config['min_learning_rate'],
            warmup_epochs=self.config['warmup_epochs'],
            epochs=self.config['max_epochs'],
        )
        return [optimizer], [scheduler]
