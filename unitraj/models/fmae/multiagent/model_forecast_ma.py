import torch
import torch.nn as nn

from ..layers.agent_embedding import AgentEmbeddingLayer
from ..layers.lane_embedding import LaneEmbeddingLayer
from ..layers.transformer_blocks import Block
from torch_scatter import scatter_mean


class MLPDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class ModelForecastMultiAgent(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
        num_modes: int = 6,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.use_cls_token = use_cls_token

        # encoder
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
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

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.traj_decoder = MLPDecoder(
            embed_dim + self.num_modes, self.future_steps * 2
        )
        self.prob_decoder = nn.Linear(embed_dim + self.num_modes, 1)

        self.register_buffer("one_hot_mask", torch.eye(self.num_modes))

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.cls_pos, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :50]
        hist_key_padding_mask = data["x_key_padding_mask"]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
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

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        # pos embed
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, 49], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        # type embed
        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        if self.use_cls_token:
            x_encoder = torch.cat([x_encoder, self.cls_token.repeat(B, 1, 1)], dim=1)
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros(B, 1, device=key_padding_mask.device)],
                dim=1,
            )
            pos_embed = torch.cat([pos_embed, self.cls_pos.repeat(B, 1, 1)], dim=1)

        x_encoder = x_encoder + pos_embed
        for i, blk in enumerate(self.blocks):
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        if self.use_cls_token:
            cls_token = x_encoder[:, -1]
        else:
            # global avg pooling
            cls_token = scatter_mean(x_encoder, key_padding_mask.long(), dim=1)[:, 0]

        K = self.num_modes
        x_actors = x_encoder[:, :N].unsqueeze(1).repeat(1, K, 1, 1)
        cls_token = cls_token.unsqueeze(1).repeat(1, K, 1)
        one_hot_mask = self.one_hot_mask

        x_actors = torch.cat(
            [x_actors, one_hot_mask.view(1, K, 1, K).repeat(B, 1, N, 1)], dim=-1
        )
        cls_token = torch.cat(
            [cls_token, one_hot_mask.view(1, K, K).repeat(B, 1, 1)], dim=-1
        )

        y_hat = self.traj_decoder(x_actors).view(B, K, N, self.future_steps, 2)
        pi = self.prob_decoder(cls_token).view(B, K)

        return {"y_hat": y_hat, "pi": pi}
