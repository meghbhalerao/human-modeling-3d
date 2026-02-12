import torch
import torch.nn as nn
import math

class DiTBlock(nn.Module):
    def __init__(self, latent_dim, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim)
        )

    def forward(self, x, t_emb):
        # We add timestep embedding to the hidden state
        x = x + t_emb.unsqueeze(1)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MotionDiT(nn.Module):
    def __init__(self, input_dim, latent_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(latent_dim, input_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, latent_dim)) # Max seq len 512

    def forward(self, x, t):
        # x: [Batch, SeqLen, InputDim], t: [Batch]
        t_emb = self.t_embed(self.timestep_embedding(t, 512)) # 512 is latent_dim
        
        h = self.input_proj(x) + self.pos_embed[:, :x.shape[1], :]
        for block in self.blocks:
            h = block(h, t_emb)
        
        return self.output_proj(h)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
