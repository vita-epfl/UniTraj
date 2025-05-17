import torch
import torch.nn as nn


class LaneEmbeddingLayer(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(feat_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.encoder_channel, 1),
        )

    def forward(self, x):
        bs, n, _ = x.shape

        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # B 512 n
        feature = self.second_conv(feature)  # B c n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B c
        return feature_global
