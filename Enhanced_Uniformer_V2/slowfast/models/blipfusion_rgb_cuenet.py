import torch
import torch.nn as nn
import torch.nn.functional as F
from .pose_temporal_encoder import PoseTemporalEncoder
from .text_encoder import TextEncoder
from .fusion_blocks import CrossModalFusion, TextVisualFusion
from .build import MODEL_REGISTRY
import slowfast.models.uniformerv2_model as model
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

class AttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)

@MODEL_REGISTRY.register()
class BlipFusionRGBCUENet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.use_pose = cfg.MODEL.USE_POSE
        self.use_text = cfg.MODEL.USE_TEXT

        backbone = cfg.UNIFORMERV2.BACKBONE
        self.backbone = model.__dict__[backbone](...)

        # Pose branch
        if self.use_pose:
            self.pose_encoder = PoseTemporalEncoder(num_joints=cfg.DATA.NUM_JOINTS)
            self.pose_projection = nn.Linear(256, 400)
            self.cross_fusion = CrossModalFusion()

        # Text branch
        if self.use_text:
            self.text_encoder = TextEncoder(freeze=cfg.MODEL.FREEZE_TEXT_ENCODER)
            self.text_projection = nn.Sequential(
                nn.Linear(768, 400),
                nn.GELU(),
                nn.LayerNorm(400),
                nn.Dropout(0.3)
            )
            self.text_visual_fusion = TextVisualFusion()

        # Classification head
        self.attn_pool = AttentionPooling(400)
        self.classifier = nn.Sequential(
            nn.Linear(400, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, cfg.MODEL.NUM_CLASSES)
        )

    def forward(self, video_tensor, pose_tensor=None, input_ids=None, attention_mask=None):
        B = video_tensor.shape[0]
        rgb_feats = self.backbone(video_tensor)

        if rgb_feats.ndim == 2:
            rgb_feats = rgb_feats.unsqueeze(1)

        fused_feats = rgb_feats

        # Pose fusion
        if self.use_pose and pose_tensor is not None:
            pose_feats = self.pose_projection(self.pose_encoder(pose_tensor))
            fused_feats = self.cross_fusion(fused_feats, pose_feats)

        # Text fusion
        if self.use_text and input_ids is not None:
            text_feats = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feats = self.text_projection(text_feats)
            fused_feats = self.text_visual_fusion(fused_feats, text_feats)

        pooled = self.attn_pool(fused_feats)
        return self.classifier(pooled)
