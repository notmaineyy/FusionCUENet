import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    """Fuse RGB and Pose features with cross-attention + residual gate."""
    def __init__(self, dim_rgb=400, dim_pose=400, hidden=512):
        super().__init__()
        self.query_rgb = nn.Linear(dim_rgb, hidden)
        self.key_pose = nn.Linear(dim_pose, hidden)
        self.value_pose = nn.Linear(dim_pose, hidden)
        self.proj = nn.Linear(hidden, dim_rgb)
        self.norm = nn.LayerNorm(dim_rgb)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, rgb, pose):
        q = self.query_rgb(rgb)
        k = self.key_pose(pose)
        v = self.value_pose(pose)

        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5), dim=-1)
        out = torch.bmm(attn, v)
        fused = self.norm(rgb + self.gate * self.proj(out))
        return fused


class TextVisualFusion(nn.Module):
    """Fuse visual and text embeddings via multihead attention with scaling."""
    def __init__(self, dim_visual=400, dim_text=400, hidden=512):
        super().__init__()
        self.visual_proj = nn.Linear(dim_visual, hidden)
        self.text_proj = nn.Linear(dim_text, hidden)
        self.text_weight = nn.Parameter(torch.ones(1))
        self.text_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim_visual)

    def forward(self, visual_feats, text_feats):
        visual_proj = self.visual_proj(visual_feats)
        text_proj = self.text_norm(self.text_proj(text_feats))
        weighted_text = text_proj * self.text_weight.clamp(0.1, 10.0)

        weighted_text = weighted_text.unsqueeze(1)
        attn_output, _ = self.attention(
            query=visual_proj, key=weighted_text, value=weighted_text, need_weights=False
        )
        fused = self.norm(visual_proj + attn_output)
        return visual_feats + self.out_proj(fused)
