import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.forecast_mae.layers.agent_embedding import AgentEmbeddingLayer
from unitraj.models.forecast_mae.layers.lane_embedding import LaneEmbeddingLayer
from unitraj.models.forecast_mae.layers.transformer_blocks import Block
from unitraj.models.forecast_mae.layers.multimodal_decoder import MultimodalDecoder

from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.forecast_mae.utils.optim import WarmupCosLR


class ForecastMAE(BaseModel):
    def __init__(self, config):
        super(ForecastMAE, self).__init__(config)
        self.config = config
        self.embed_dim = config['embed_dim']
        self.encoder_depth = config['encoder_depth']
        self.num_heads = config['num_heads']
        self.mlp_ratio = config['mlp_ratio']
        self.qkv_bias = config['qkv_bias']
        self.drop_path = config['drop_path']
        self.future_len: int = config['future_len']
        self.past_len: int = config['past_len']
        self.num_agent_feature = config['num_agent_feature']
        self.num_map_feature = config['num_map_feature']

        self.strict_loading = False  # Allow partial loading

        #changed 4 by 40 in_chans
        self.hist_embed = AgentEmbeddingLayer(
            4, self.embed_dim // 4, drop_path_rate=self.drop_path
        )
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

        self.actor_type_embed = nn.Parameter(torch.Tensor(1,1, self.embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, self.embed_dim))

        self.decoder = MultimodalDecoder(self.embed_dim, self.future_len)
        self.dense_predictor = nn.Sequential(
            nn.Linear(self.embed_dim, 256), nn.ReLU(), nn.Linear(256, self.future_len * 2)
        )
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def create_padding_mask_keys(self,padding_masks):
        padding_mask_keys = {}
        for i, mask in enumerate(padding_masks):
            key = f"padding_mask_{i}"  # Generate a unique key for each mask
            padding_mask_keys[key] = mask
        return padding_mask_keys

    def convert(self, batch):
        input_dict = batch['input_dict']
        input_dict['x'] = input_dict['obj_trajs'][..., :2]

        input_dict['x_padding_mask'] = ~(torch.cat([input_dict['obj_trajs_mask'], input_dict['obj_trajs_future_mask']], dim=-1).to(torch.bool))
        input_dict['x_key_padding_mask'] = input_dict['x_padding_mask'].all(-1)

        input_dict['lane_padding_mask'] = ~input_dict['map_polylines_mask']
        input_dict['lane_key_padding_mask'] = input_dict['lane_padding_mask'].all(-1)

        input_dict['lane_positions'] = input_dict['map_polylines'][..., :2]
        input_dict['lane_centers'] = input_dict['lane_positions'][:, :, 9:11].mean(dim=2)
        input_dict['lane_angles'] = torch.atan2(
            input_dict['lane_positions'][:, :, 10, 1] - input_dict['lane_positions'][:, :, 9, 1],
            input_dict['lane_positions'][:, :, 10, 0] - input_dict['lane_positions'][:, :, 9, 0],
        )

        input_dict['x_velocity_diff'] = input_dict['obj_trajs'][..., 35:37]
        input_dict['x_centers'] = input_dict['obj_trajs_last_pos'][..., :2]
        input_dict['y'] = input_dict['obj_trajs_future_state'][..., :2]

        velocity = input_dict['obj_trajs'][..., 35:37]

        velocity_norm = torch.norm(velocity, dim=-1)

        input_dict['x_velocity'] = velocity_norm

        input_dict['x_angles'] = torch.norm(input_dict['obj_trajs'][...,-1, 33:35], dim=-1)

        return batch





    def forward(self, input_batch, batch_idx=None):
        batch = self.convert(input_batch)
        input_dict = batch['input_dict']
        hist_padding_mask = input_dict["x_padding_mask"][...,:self.past_len]
        hist_key_padding_mask = input_dict["x_key_padding_mask"]


        hist_feat = torch.cat(
            [
                input_dict['obj_trajs'][...,:2],
                input_dict['x_velocity'].unsqueeze(-1),
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )


        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_padding_mask = input_dict["lane_padding_mask"]
        lane_normalized = input_dict["lane_positions"] - input_dict["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([input_dict["x_centers"], input_dict["lane_centers"]], dim=1)
        angles = torch.cat([input_dict["x_angles"], input_dict["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        #actor_type_embed = self.actor_type_embed[batch["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += self.actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [input_dict["x_key_padding_mask"], input_dict["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        x_agent = x_encoder[:, 0]
        y_hat, pi = self.decoder(x_agent)

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, self.future_len, 2)

        output = {}
        output['predicted_probability'] = pi
        output['predicted_trajectory'] = y_hat

        output['others_predicted_trajectory'] = y_hat_others

        loss = self.cal_loss(output, input_dict)

        return output, loss


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
            optim_groups, lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"]
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.config["learning_rate"],
            min_lr=self.config["min_learning_rate"],
            warmup_epochs=self.config["warmup_epochs"],
            epochs=self.config["max_epochs"],
        )
        return [optimizer], [scheduler]

    def cal_loss(self, out, data):
        y_hat, pi, y_hat_others = out["predicted_trajectory"], out["predicted_probability"], out['others_predicted_trajectory']

        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.past_len:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        loss = agent_reg_loss + agent_cls_loss + others_reg_loss

        return loss
