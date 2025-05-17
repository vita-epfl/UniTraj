from os.path import exists as ospexists
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block


class EMP(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        decoder="",
        historical_steps: int = 50,
        future_steps: int = 60,
    ) -> None:
        super().__init__()

        self.history_steps = historical_steps
        self.future_steps = future_steps
            
        self.embed_dim = embed_dim
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]

        if decoder == "mlp":
            print("RUN EMP-M")
            from .layers.multimodal_decoder_emp import MultimodalDecoder
        elif decoder == "detr":
            print("RUN EMP-D")
            from .layers.multimodal_decoder_emp_attn import MultimodalDecoder
        else:
            assert False, "Unknown Decoder Type in Config (must be <mlp> or <detr>, but is <{}>)".format(decoder)
    

        self.h_proj = nn.Linear(5, embed_dim)
        self.h_embed = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                cross_attn=False
            )
            for i in range(encoder_depth)
        )

        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                cross_attn=False
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        
        k = 6
        self.decoder = MultimodalDecoder(embed_dim, self.future_steps, k=k)
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, self.future_steps * 2)
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


    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        #print( "\nLOAD {}\n".format(ckpt_path) )
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.") and (not k.startswith("net.decoder.") or "last" in ckpt_path or "epoch" in ckpt_path)
        }
        
        return self.load_state_dict(state_dict=state_dict, strict=False)


    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :self.history_steps]
        
        hist_key_padding_mask = data["x_key_padding_mask"]

        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                (~hist_padding_mask[..., None]).float(),
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)

        ####################
        # AGENT ENCODING

        actor_feat = hist_feat[~hist_feat_key_padding]
        
        ts = torch.arange(self.history_steps).view(1, -1, 1).repeat(actor_feat.shape[0], 1, 1).to(actor_feat.device).float()
        actor_feat = torch.cat([actor_feat, ts], dim=-1)

        actor_feat = self.h_proj( actor_feat )
        kpm = hist_padding_mask.view(B*N, -1)[~hist_feat_key_padding]
        for blk in self.h_embed:
            actor_feat = blk(actor_feat, key_padding_mask=kpm)
        actor_feat = torch.max(actor_feat, axis=1).values
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )

        #actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        mask_indices = torch.nonzero(~hist_feat_key_padding, as_tuple=True)
        actor_feat_tmp.scatter_(0, mask_indices[0].unsqueeze(1).expand(-1, self.embed_dim), actor_feat)

        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        ####################
        # LANE ENCODING

        lane_padding_mask = data["lane_padding_mask"]

        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, (~lane_padding_mask[..., None]).float()], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        ####################
        # POS ENCODING

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, self.history_steps-1], data["lane_angles"]], dim=1)    

        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)      
        pos_embed = self.pos_embed(pos_feat)

        ####################
        # SCENE ENCODING

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat([data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)           

        x_encoder = x_encoder + pos_embed

        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        ####################
        # DECODING        

        x_agent = x_encoder[:, 0] 
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, self.future_steps, 2)

        y_hat, pi = self.decoder(x_agent, x_encoder, key_padding_mask, N)
        
        y_hat_eps = y_hat[:, :, -1]

        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_others": y_hat_others,
            "y_hat_eps": y_hat_eps,
            "x_agent": x_agent
        }
