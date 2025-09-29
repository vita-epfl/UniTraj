import torch
import torch.nn as nn


class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, k=6) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.k = k

        #self.multimodal_proj = nn.Linear(embed_dim, self.k * embed_dim)
        self.mode_embed = nn.Embedding(self.k, embedding_dim=embed_dim) 

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

        nn.init.orthogonal_(self.mode_embed.weight)
        return


    def forward(self, x, __1, __2, __3):
        B = x.shape[0]

        mode_embeds = self.mode_embed.weight.view(1, self.k, self.embed_dim).repeat(B, 1, 1)
        x = x.unsqueeze(1).repeat(1, self.k, 1) + mode_embeds

        loc = self.loc(x).view(-1, self.k, self.future_steps, 2)
        pi = self.pi(x).squeeze(-1)

        return loc, pi
