from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.transformer_blocks import Block


class GlobalConsistencyModule(nn.Module):
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
        use_stream_encoder=True,
        use_stream_decoder=True,
        use_target_context=True,
        k=6
    ) -> None:
        super().__init__()

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        b1_d = 3
        dpr1 = [x.item() for x in torch.linspace(0, drop_path, b1_d)]
        self.blocks1 = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr1[i],
            )
            for i in range(b1_d)
        )

        b2_d = 3
        dpr2 = [x.item() for x in torch.linspace(0, drop_path, b2_d)]      
        self.blocks2 = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr2[i],
            )
            for i in range(b2_d)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, future_steps * 2),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.future_steps = future_steps

        self.actor_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        self.initialize_weights()
        return


    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)

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
        ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        state_dict = {
            k[len('net.') :]: v for k, v in ckpt.items() if k.startswith('net.')
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        # Positional embedding w.r.t. scene global coordinate system
        B, N, K, D = data["x_modes"].shape
        x_centers = data["origins"] - data["origins"][:, 0].unsqueeze(1)
        angles = (data["thetas"] - data["thetas"][:, 0].unsqueeze(1) + torch.pi) % (2 * torch.pi) - torch.pi
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        # Get all agents which are potentially standing still
        base_pred = data["y_hats"].clone()
        length = torch.norm(base_pred[:, 1:, :, -1], dim=-1)
        mask = (length < 1.0).any(dim=-1)

        # Type tokens: moving or standing still
        type_tokens = self.actor_type_embed[0].unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)
        type_tokens = torch.cat( [type_tokens, self.actor_type_embed[mask.int()+1]], dim=1 )

        # Get scene-wide embeddings with positional embeddings
        x_encoder = data["x_modes"] + (pos_embed + type_tokens).unsqueeze(-2)
        x_key_valid_mask = data["x_key_valid_mask"].unsqueeze(-1).repeat(1, 1, K)
        
        # Self-attention across all modes per agent
        x_encoder = x_encoder.view(B*N, K, D)
        container = torch.zeros_like(x_encoder)
        x_encoder = x_encoder[x_key_valid_mask.view(B*N, K)]
        for blk in self.blocks1:
            x_encoder = blk(x_encoder)
        container[x_key_valid_mask.view(B*N, K)] = x_encoder
        x_encoder = container

        # Self-attention across all agent per predicted world
        x_encoder = x_encoder.view(B, N, K, D).permute(0, 2, 1, 3).contiguous().view(B*K, N, D) 
        x_key_valid_mask = x_key_valid_mask.view(B, N, K).permute(0, 2, 1).contiguous().view(B*K, N) 

        for blk in self.blocks2:
            x_encoder = blk(x_encoder, key_padding_mask=~x_key_valid_mask)    
        x_encoder = self.norm(x_encoder)

        # Decoding
        x_encoder = x_encoder.view(B, K, N, D)

        # Trajectory Predictions
        y_hat = self.loc(x_encoder).view(B, 6, -1, self.future_steps, 2)

        # Score (done using only moving agents) 
        new_pred = y_hat.permute(0, 2, 1, 3, 4).detach()
        length = torch.norm(new_pred[:, 1:, :, -1], dim=-1)
        mask = (length < 1.0).any(dim=-1)
        mask = ~torch.cat( [torch.zeros_like(mask)[:, :1], mask], dim=1 ).unsqueeze(1).unsqueeze(-1).repeat(1, 6, 1, 1)
        mask = mask & data["x_key_valid_mask"].unsqueeze(1).unsqueeze(-1).repeat(1, K, 1, 1)

        masked_src = x_encoder * mask        
        masked_src = masked_src.sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        pi = self.pi(masked_src).view(B, 6)

        ret_dict = {
            'y_hat': y_hat,
            'pi': pi,
            'orig_y_hat': data["y_hats"],
            'orig_pi': data["pis"],
            'x_mode': x_encoder,
        }
        return ret_dict