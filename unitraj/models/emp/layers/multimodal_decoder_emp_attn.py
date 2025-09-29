import os

import numpy as np
import torch
import torch.nn as nn

from .transformer_blocks import Block

class MultimodalDecoder(nn.Module):
    def __init__(self, embed_dim, future_steps, k=6) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.k = k

        self.attn_depth = 3
        dpr = [x.item() for x in torch.linspace(0, 0.2, self.attn_depth)]
        self.lane_blks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=8,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=dpr[i],
                cross_attn=True,
                kdim=embed_dim,
                vdim=embed_dim
            )
            for i in range(self.attn_depth)
        )
        self.agent_blks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=8,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=dpr[i],
                cross_attn=True,
                kdim=embed_dim,
                vdim=embed_dim
            )
            for i in range(self.attn_depth)
        )

        self.pi = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, 1),
        )

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, future_steps * 2),
        )

        self.pi_norm = nn.Softmax(dim=-1)

        self.mode_embed = nn.Embedding(self.k, embedding_dim=embed_dim)  

        self.initialize_weights()
        return


    def initialize_weights(self):
        nn.init.normal_(self.mode_embed.weight, std=0.02)
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


    def forward(self, x, x_encoder, key_padding_mask, N):
        B = x.shape[0]

        kv = x_encoder

        kv_agent = kv[:, 0].unsqueeze(1)
        mask_agent = key_padding_mask[:, 0].unsqueeze(1)
        kv_lane = kv[:, N:]
        mask_lane = key_padding_mask[:, N:]

        intention_query = self.mode_embed.weight.view(1, self.k, self.embed_dim).repeat(B, 1, 1)
        for ali in range(self.attn_depth):
            intention_query = self.agent_blks[ali](intention_query, k=kv_agent, v=kv_agent, key_padding_mask=mask_agent)
            intention_query = self.lane_blks[ali](intention_query, k=kv_lane, v=kv_lane, key_padding_mask=mask_lane)
        intention_query.reshape(B, -1, self.embed_dim)
        
        loc = self.loc(intention_query).view(B, self.k, self.future_steps, 2)
        pi = self.pi(intention_query).squeeze(2)
        return loc, pi
