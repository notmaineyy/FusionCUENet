import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseTemporalEncoder(nn.Module):
    """
    Multi-person pose encoder with velocity/acceleration features and temporal fusion.
    Handles (B, T, P, J, 4) -> (B, T, D)
    """
    def __init__(self, num_joints, in_dim=3, embed_dim=256, temporal_layers=3):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # Per-joint MLP
        self.joint_proj = nn.Sequential(
            nn.Linear(in_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Temporal conv layers with residual
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=8),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ) for _ in range(temporal_layers)
        ])

        # Transformer for global temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Learnable person pooling
        self.person_pool = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        """
        x: (B, T, P, J, 4)
        """
        coords = x[..., :3]
        conf = x[..., 3]  # (B, T, P, J)

        B, T, P, J, _ = coords.shape

        # Compute velocity + acceleration
        vel = coords[:, 1:] - coords[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]

        vel = F.pad(vel, (0, 0, 0, 0, 0, 0, 1, 0))
        acc = F.pad(acc, (0, 0, 0, 0, 0, 0, 2, 0))

        feats = torch.cat([coords, vel, acc], dim=-1)  # (B, T, P, J, 9)
        feats = feats.view(B * T * P * J, -1)
        feats = self.joint_proj(feats).view(B, T, P, J, self.embed_dim)

        # Confidence-weighted joint pooling per person
        conf = conf.unsqueeze(-1)
        conf = conf / (conf.sum(dim=3, keepdim=True) + 1e-8)
        person_feats = (feats * conf).sum(dim=3)  # (B, T, P, D)

        # Average across persons + learnable bias
        pooled_person = person_feats.mean(dim=2) + self.person_pool

        # Temporal modeling: conv residual
        x = pooled_person.permute(0, 2, 1)  # (B, D, T)
        for conv in self.temporal_convs:
            x = x + conv(x)  # residual
        x = x.permute(0, 2, 1)  # (B, T, D)

        # Transformer
        x = self.temporal_transformer(x)
        return x  # (B, T, D)
