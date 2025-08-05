class PoseAttentionEncoder(nn.Module):
    def __init__(self, num_joints, in_dim=3, embed_dim=256):
        super().__init__()
        self.num_joints = num_joints
        self.in_dim = in_dim  # x, y, z
        self.embed_dim = embed_dim

        self.joint_proj = nn.Linear(in_dim, embed_dim)
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: (B, T, P, J, 4) -> (x, y, z, confidence)
        Returns: (B, T, embed_dim)
        """
        coords = x[..., :3]         # (B, T, P, J, 3)
        conf = x[..., 3]            # (B, T, P, J)

        B, T, P, J, _ = coords.shape
        x = coords.view(B * T * P * J, self.in_dim)
        x = self.joint_proj(x)                      # (B*T*P*J, embed_dim)
        x = x.view(B, T, P, J, self.embed_dim)      # (B, T, P, J, D)

        # Attention weights from confidence
        conf = conf.unsqueeze(-1)                   # (B, T, P, J, 1)
        attn_scores = self.att_mlp(x)               # (B, T, P, J, 1)
        attn_weights = torch.softmax(attn_scores * conf, dim=3)  # (B, T, P, J, 1)

        # Apply attention to joint features
        pose_repr = (x * attn_weights).sum(dim=3)   # (B, T, P, D)

        return pose_repr  # (B, T, P, D)
