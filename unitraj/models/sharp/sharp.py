from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block, InteractionModule
from .layers.custom_transformer_blocks import Block as CustomBlock
from .layers.multimodal_decoder_attn import MultimodalDecoder            
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Sharp_I(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps=60,
        use_transformer_decoder=False,
        num_decoder_layers=6,
        dm="av2",
        k=6
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.dm = dm

        # AV1/AV2: 10 Hz, NUS 2 Hz
        self.frame_rate = 0.1 if self.dm != "nus" else 0.5

        # Agent encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.h_proj = nn.Linear(5 if self.dm != "av1" and self.dm != "nus" else 4, embed_dim)
        self.h_embed = nn.ModuleList(
            CustomBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                cross_attn=False
            )
            for i in range(encoder_depth)
        )

        # Lane encoder
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        # Positional embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scene encoder
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Type embeddings for agents and lanes
        if self.dm != "nus":
            self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        else:
            self.actor_type_embed = nn.Parameter(torch.Tensor(7, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.k = k

        # Decoder
        self.decoder = MultimodalDecoder(use_target_context=False, future_steps=self.future_steps, k=k)
        # Auxiliary decoder
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, self.future_steps * 2)
        )

        self.initialize_weights()
        return


    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)
        return
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        return
    

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        state_dict = {
            k[len('model.') :]: v for k, v in ckpt.items() if k.startswith('model.') 
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)


    def forward(self, data):
        hist_valid_mask = data['x_valid_mask']
        hist_key_valid_mask = data['x_key_valid_mask']
        # AV2: use positions and velocity as features; AV1 & NUS: use positions as features
        if self.dm != "av1" and self.dm != "nus":
            hist_feat = torch.cat(
                [
                    data['x_positions_diff'],
                    data['x_velocity_diff'][..., None],
                    hist_valid_mask[..., None],
                ],
                dim=-1,
            )
        else:
            hist_feat = torch.cat(
                [
                    data['x_positions_diff'],
                    hist_valid_mask[..., None],
                ],
                dim=-1,
            )

        ##################
        # AGENT ENCODING #
        ##################

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid_mask = (hist_key_valid_mask).view(B * N)
        actor_feat = hist_feat[hist_feat_key_valid_mask]

        num_hist_ts = hist_feat.shape[1]
        ts = torch.arange(num_hist_ts).view(1, -1, 1).repeat(actor_feat.shape[0], 1, 1).to(actor_feat.device).float()
        actor_feat = torch.cat([actor_feat, ts], dim=-1)

        actor_feat = self.h_proj( actor_feat )
        kpm = (~hist_valid_mask).view(B*N, -1)[hist_feat_key_valid_mask]
        for blk in self.h_embed:
            actor_feat = blk(actor_feat, key_padding_mask=kpm)
        actor_feat = torch.max(actor_feat, axis=1).values


        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )

        actor_feat_tmp[hist_feat_key_valid_mask] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        #################
        # LANE ENCODING #
        #################
   
        lane_valid_mask = data['lane_valid_mask']
        lane_normalized = data['lane_positions'] - data['lane_centers'].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        ###################################################
        # POSITIONAL EMBEDDINGS AND TOKEN TYPE EMBEDDINGS #
        ###################################################

        x_centers = torch.cat([data['x_centers'], data['lane_centers']], dim=1)
        angles = torch.cat([data['x_angles'][:, :, -1], data['lane_angles']], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data['x_attr'][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        ###########################
        # CONSTRUCT SCENE CONTEXT #
        ###########################

        actor_key_valid_mask = data['x_key_valid_mask']
        lane_key_valid_mask = data['lane_key_valid_mask']

        x_encoder_all = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask_all = torch.cat(
            [actor_key_valid_mask, lane_key_valid_mask], dim=1
        )
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask = torch.cat(
            [actor_key_valid_mask, lane_key_valid_mask], dim=1
        )
        x_type_mask = torch.cat([actor_feat.new_ones(*actor_feat.shape[:2]),
                                lane_feat.new_zeros(*lane_feat.shape[:2])], dim=1).bool()
        
        ###########################
        # TARGET-CENTRIC FEATURES #
        ###########################

        if "memory_dict" in data and data["memory_dict"] is not None and self.use_target_context:
            cos, sin = data["theta"].cos(), data["theta"].sin()
            rot_mat = data["theta"].new_zeros(B, 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = -sin
            rot_mat[:, 1, 0] = sin
            rot_mat[:, 1, 1] = cos

            # project previous predictions to current coordinate frame
            memory_new_y_hat = data["memory_dict"]["glo_y_hat"].float()
            ori_idx = ((data["timestamp"] - data["memory_dict"]["timestamp"]) / self.frame_rate).long() - 1
            ori_idx[ori_idx < 0] = 0
            memory_traj_ori = torch.gather(memory_new_y_hat, 2, ori_idx.reshape(
                B, 1, -1, 1).repeat(1, memory_new_y_hat.size(1), 1, memory_new_y_hat.size(-1)))
            memory_new_y_hat = torch.bmm(
                (memory_new_y_hat - memory_traj_ori).reshape(B, -1, 2), rot_mat
            ).reshape(B, memory_new_y_hat.size(1), -1, 2).to(torch.float32)
            
            # get projected endpoints and compute positional embeddings w.r.t. endpoints
            data["memory_new_y_hat"] = memory_new_y_hat
            target_offset = 59 if self.dm == "av2" else 29 if self.dm == "av1" else 11 if self.dm == "nus" else -1
            target_pos = memory_new_y_hat[:, :, target_offset].detach()
            target_angle = torch.atan2(memory_new_y_hat[:, :, target_offset, 1]-memory_new_y_hat[:, :, target_offset-4, 1], memory_new_y_hat[:, :, target_offset, 0]-memory_new_y_hat[:, :, target_offset-4, 0]).detach()
            x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
            x_centers = x_centers.unsqueeze(1).repeat(1, self.k, 1, 1) - target_pos.unsqueeze(2)
            angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
            angles = angles.unsqueeze(1).repeat(1, self.k, 1) - target_angle.unsqueeze(2)
            x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            pos_feat = torch.cat([x_centers, x_angles], dim=-1)
            target_pos_embed = self.target_pos_embed(pos_feat)

            # use current scene context to get target-centric context feature set - use max_distance to select only tokens around endpoint
            max_distance = 30
            target_encoder = x_encoder_all.unsqueeze(1).expand_as(target_pos_embed) + target_pos_embed
            target_mask = (~key_valid_mask_all.unsqueeze(1).expand(B, self.k, key_valid_mask_all.shape[1])) | (torch.norm(x_centers, dim=-1) > max_distance)
            target_mask = target_mask.view(B*self.k, -1)
            target_mask[:, 0] = False
            target_encoder[:, 0] = 0

            # efficiently compute target-centric features
            target_encoder = target_encoder.view(B*self.k, -1, self.embed_dim)
            container = torch.zeros_like(target_encoder).view(-1, self.embed_dim)
            target_valid = ~target_mask
            
            batch_indices, valid_indices = target_valid.nonzero(as_tuple=True)
            compressed_target_encoder = target_encoder[batch_indices, valid_indices]  # [M_total, D]
            compressed_target_mask = target_mask[batch_indices, valid_indices]  # [M_total]

            M_per_batch = target_valid.sum(dim=1).tolist()
            compressed_target_encoder = torch.split(compressed_target_encoder, M_per_batch)
            compressed_target_mask = torch.split(compressed_target_mask, M_per_batch)
            compressed_target_encoder = pad_sequence(compressed_target_encoder, batch_first=True)
            compressed_target_mask = pad_sequence(compressed_target_mask, batch_first=True, padding_value=True)

            for blk in self.target_blocks:
                compressed_target_encoder = blk(compressed_target_encoder, key_padding_mask=compressed_target_mask)
            compressed_target_encoder = self.target_norm(compressed_target_encoder) 

            # add embedding current coordinate system root -> endpoint positions
            target_center_embed = self.target_center_embed(target_pos) 
            compressed_target_encoder = compressed_target_encoder.view(B, self.k, -1, self.embed_dim) + target_center_embed.unsqueeze(2)

            container[target_valid.view(-1)] = compressed_target_encoder.view(-1, self.embed_dim)[~compressed_target_mask.view(-1)]
            target_encoder = container.view(B, self.k, -1, self.embed_dim)

        # add positional embedding to scene encoding after computation of the target-centric features is done
        x_encoder = x_encoder + pos_embed

        #################
        # DUAL TRAINING #
        #################
        # if dual training: do prediction using only the current scene context without streaming
        if self.dual:
            x_curr = x_encoder.clone()
            
            for blk in self.blocks:
                x_curr = blk(x_curr, key_padding_mask=~key_valid_mask)
            x_curr = self.norm(x_curr)
            x_agent = x_curr[:, 0]
            aux = [None, None, data]
            y_hat_single, pi_single, __ = self.decoder(x_agent, x_curr, (~key_valid_mask), N, aux=aux)
        else:
            y_hat_single = None
            pi_single = None

        ####################################
        # INSTANCE-AWARE CONTEXT STREAMING #
        ####################################
        if isinstance(self, Sharp):
            ids_query = torch.zeros((x_encoder.shape[0], x_encoder.shape[1]), dtype=torch.long, device=x_encoder.device)
            
            # read memory for streaming processing
            if 'memory_dict' in data and data['memory_dict'] is not None:
                rel_pos = data['origin'] - data['memory_dict']['origin']
                rel_ang = (data['theta'] - data['memory_dict']['theta'] + torch.pi) % (2 * torch.pi) - torch.pi
                rel_ts = data['timestamp'] - data['memory_dict']['timestamp']
                memory_pose = torch.cat([
                    rel_ts.unsqueeze(-1), rel_ang.unsqueeze(-1), rel_pos
                ], dim=-1).float().to(x_encoder.device)
                memory_x_encoder = data['memory_dict']['x_encoder']
                memory_valid_mask = data['memory_dict']['x_mask']
                memory_type_mask = data['memory_dict']['x_type_mask']
                memory_ids = data['memory_dict']['ids_query']
                memory_cache_ids = data['memory_dict']['cache_ids']
            else:
                memory_pose = x_encoder.new_zeros(x_encoder.size(0), self.pose_dim)
                memory_x_encoder = x_encoder
                memory_valid_mask = key_valid_mask
                memory_type_mask = x_type_mask
                memory_ids = ids_query
                memory_cache_ids = torch.cat([data['agent_ids'], data['lane_ids']], dim=1)
            cur_pose = torch.zeros_like(memory_pose)

            if self.use_stream_encoder:
                # scene interaction
                new_x_encoder = x_encoder
                C = x_encoder.size(-1)

                # interaction-aware: get token ids (agent ids and lane ids) and construct same instance masks
                if self.biased_interaction:
                    cache_ids = torch.cat([data['agent_ids'], data['lane_ids']], dim=1)
                    mask = (cache_ids.unsqueeze(2) == memory_cache_ids.unsqueeze(1))
                    mask = mask[x_type_mask].reshape(B, -1, memory_ids.shape[-1])
                    # agent to agent+lanes with mask
                    new_actor_feat = self.scene_interact(new_x_encoder[x_type_mask].reshape(B, -1, C), memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask, mask=mask)           
                else:
                    # agent to agent+lanes
                    new_actor_feat = self.scene_interact(new_x_encoder[x_type_mask].reshape(B, -1, C), memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask)
                # lane to lane
                new_lane_feat = self.scene_interact(new_x_encoder[~x_type_mask].reshape(B, -1, C), memory_x_encoder[~memory_type_mask].reshape(B, -1, C), cur_pose, memory_pose, key_padding_mask=~memory_valid_mask[~memory_type_mask].reshape(B, -1))
                new_x_encoder = torch.cat([new_actor_feat, new_lane_feat], dim=1)
                x_encoder = new_x_encoder * key_valid_mask.unsqueeze(-1) + x_encoder * ~key_valid_mask.unsqueeze(-1)

        ##################
        # SCENE ENCODING #
        ##################
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)
        x_encoder = self.norm(x_encoder)

        ###########
        # DECODER #
        ###########
        x_agent = x_encoder[:, 0]
        if "memory_dict" in data and data["memory_dict"] is not None and self.use_target_context:
           aux = [target_encoder, target_mask, data, compressed_target_encoder, compressed_target_mask]
        else:
           aux = [None, None, data]
        y_hat, pi, aux_dec_ret = self.decoder(x_agent, x_encoder, (~key_valid_mask), N, aux=aux)
        x_mode = aux_dec_ret[0]

        ######################
        # AUXILIARY DECODING #
        ######################
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), self.future_steps, 2) 

        ####################
        # TRAJECTORY RELAY #
        ####################
        cos, sin = data['theta'].cos(), data['theta'].sin()
        rot_mat = data['theta'].new_zeros(B, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos

        if isinstance(self, Sharp) and self.use_stream_decoder:
            # traj interaction
            if 'memory_dict' in data and data['memory_dict'] is not None:
                memory_y_hat = data['memory_dict']['glo_y_hat']
                memory_x_mode = data['memory_dict']['x_mode']
                ori_idx = ((data['timestamp'] - data['memory_dict']['timestamp']) / self.frame_rate).long() - 1
                ori_idx[ori_idx < 0] = 0
                memory_traj_ori = torch.gather(memory_y_hat, 2, ori_idx.reshape(
                    B, 1, -1, 1).repeat(1, memory_y_hat.size(1), 1, memory_y_hat.size(-1)))
                memory_y_hat = torch.bmm((memory_y_hat - memory_traj_ori).reshape(B, -1, 2), rot_mat
                                        ).reshape(B, memory_y_hat.size(1), -1, 2)
                if self.window > 0:
                    traj_embed = self.traj_embed_cur(y_hat[:, :, :60].detach().reshape(B, y_hat.size(1), -1))
                    memory_traj = memory_y_hat[:, :, self.window:self.window+60]
                    memory_traj_embed = self.traj_embed_mem(memory_traj.reshape(B, memory_y_hat.size(1), -1))
                    x_mode = self.traj_interact(x_mode, memory_x_mode, cur_pose, memory_pose,
                                                        cur_pos_embed=traj_embed,
                                                        memory_pos_embed=memory_traj_embed)
                    y_hat_diff = self.stream_loc(x_mode).reshape(B, y_hat.size(1), -1, 2)
                    y_hat = y_hat + y_hat_diff
                else:
                    traj_embed = self.traj_embed(y_hat.detach()[..., :2].reshape(B, y_hat.size(1), -1))
                    memory_traj_embed = self.traj_embed(memory_y_hat.reshape(B, memory_y_hat.size(1), -1))
                    x_mode = self.traj_interact(x_mode, memory_x_mode, cur_pose, memory_pose,
                                                        cur_pos_embed=traj_embed,
                                                        memory_pos_embed=memory_traj_embed)
                    y_hat_diff = self.stream_loc(x_mode).reshape(B, y_hat.size(1), -1, 2)
                    y_hat[..., :2] = y_hat[..., :2] + y_hat_diff

        ###########################################
        # PREPARE OUTPUT AND MEMORY FOR STREAMING #
        ###########################################
        ret_dict = {
            'y_hat': y_hat,
            'pi': pi,
            'y_hat_others': y_hat_others,
            'y_hat_single': y_hat_single,
            'pi_single': pi_single
        }

        glo_y_hat = torch.bmm(y_hat.detach()[..., :2].reshape(B, -1, 2), torch.inverse(rot_mat))
        glo_y_hat = glo_y_hat.reshape(B, y_hat.size(1), -1, 2)

        # store to memory
        if isinstance(self, Sharp):
            memory_dict = {
                'x_encoder': x_encoder,
                'x_mode': x_mode,
                'glo_y_hat': glo_y_hat,
                'x_mask': key_valid_mask,
                'x_type_mask': x_type_mask,
                'origin': data['origin'],
                'theta': data['theta'],
                'timestamp': data['timestamp'],
                'rot_mat': rot_mat,
                'ids_query': ids_query,
                'cache_ids': torch.cat([data['agent_ids'], data['lane_ids']], dim=1)
            }
            ret_dict['memory_dict'] = memory_dict

        return ret_dict

class Sharp(Sharp_I):
    def __init__(self, 
                 use_stream_encoder=True,
                 use_stream_decoder=True,
                 use_target_context=True,
                 dual=False,
                 biased_interaction=False,
                 ma=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_stream_encoder = use_stream_encoder
        self.use_stream_decoder = use_stream_decoder
        self.use_target_context = use_target_context
        self.embed_dim = kwargs['embed_dim']
        self.pose_dim = 4

        self.decoder = MultimodalDecoder(use_target_context=self.use_target_context, future_steps=kwargs['future_steps'], k=kwargs['k'], ma=ma)

        self.dual = dual
        self.biased_interaction = biased_interaction

        # Instance-aware context streamer
        if self.use_stream_encoder:
            self.scene_interact = InteractionModule(
                dim=kwargs['embed_dim'],
                pose_dim=self.pose_dim,
                num_heads=kwargs['num_heads'],
                mlp_ratio=kwargs['mlp_ratio'],
                qkv_bias=kwargs['qkv_bias'],
                with_mask=self.biased_interaction
            )

        # Trajectory relay
        if self.use_stream_decoder:
            self.traj_interact = InteractionModule(
                dim=kwargs['embed_dim'],
                pose_dim=self.pose_dim,
                num_heads=kwargs['num_heads'],
                mlp_ratio=kwargs['mlp_ratio'],
                qkv_bias=kwargs['qkv_bias'],
            )
            self.window = 0
            self.stream_loc = nn.Sequential(
                nn.Linear(kwargs['embed_dim'], 256),
                nn.ReLU(),
                nn.Linear(256, kwargs['embed_dim']),
                nn.ReLU(),
                nn.Linear(kwargs['embed_dim'], kwargs['future_steps']*2),
            )
            if self.window > 0:
                self.traj_embed_mem = nn.Sequential(
                    nn.Linear(kwargs['future_steps']*2-20*2, kwargs['embed_dim']),
                    nn.GELU(),
                    nn.Linear(kwargs['embed_dim'], kwargs['embed_dim']),
                )
                self.traj_embed_cur = nn.Sequential(
                    nn.Linear(kwargs['future_steps']*2-20*2, kwargs['embed_dim']),
                    nn.GELU(),
                    nn.Linear(kwargs['embed_dim'], kwargs['embed_dim']),
                )
            else:
                self.traj_embed = nn.Sequential(
                    nn.Linear(kwargs['future_steps'] * 2, kwargs['embed_dim']),
                    nn.GELU(),
                    nn.Linear(kwargs['embed_dim'], kwargs['embed_dim']),
                )

        # Target-centric context encoder
        if self.use_target_context:
            self.target_pos_embed = nn.Sequential(
                nn.Linear(4, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

            self.target_center_embed = nn.Sequential(
                nn.Linear(2, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

            tb_depth = 2
            dpr_ = [x.item() for x in torch.linspace(0, 0.2, tb_depth)]
            self.target_blocks = nn.ModuleList(
                Block(
                    dim=self.embed_dim,
                    num_heads=kwargs['num_heads'],
                    mlp_ratio=kwargs['mlp_ratio'],
                    qkv_bias=kwargs['qkv_bias'],
                    drop_path=dpr_[i],
                )
                for i in range(tb_depth)
            )
            self.target_norm = nn.LayerNorm(self.embed_dim)

        # Preload single-agent checkpoint when finetuning on multi-agent data
        if ma:
            self.load_from_checkpoint("PLEASE_FILL_IN_YOUR_OWN_CHECKPOINT_PATH_HERE")
            grad = []
            for name, param in self.named_parameters():
                print("grad", name)
