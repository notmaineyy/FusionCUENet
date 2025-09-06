# models/fusion_cuenet.py

import torch
import torch.nn as nn
from .uniformerv2 import Uniformerv2  # assume this is your base RGB model
import slowfast.models.uniformerv2_model as model
from .build import MODEL_REGISTRY
import torch.nn.functional as F

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

# ===== Pose Temporal Encoder with Motion Dynamics =====
class PoseTemporalEncoder(nn.Module):
    def __init__(self, num_joints, in_dim=3, embed_dim=256, temporal_layers=2):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # Joint embedding
        self.joint_proj = nn.Sequential(
            nn.Linear(in_dim * 3, 128),  # xyz + vel + acc
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Temporal conv layers
        self.temporal_convs = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(temporal_layers)]
        )

        # Transformer for long-term temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=512, dropout=0.2, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Person attention
        self.person_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: (B, T, P, J, 4) -> (x, y, z, confidence)
        Returns: (B, T, D)
        """
        original_shape = x.shape
        
        # Handle different input shapes
        if x.dim() == 6:  # (B, 1, T, P, J, 3) - squeeze out the extra dimension
            if x.shape[1] == 1:
                x = x.squeeze(1)  # -> (B, T, P, J, 3)
            else:
                raise ValueError(f"Unexpected 6D pose tensor shape: {x.shape}")
            
        coords = x[..., :3]  # (B, T, P, J, 3)
        conf = x[..., 3]     # (B, T, P, J)
        B, T, P, J, _ = coords.shape

        # Compute velocity & acceleration
        vel = coords[:, 1:] - coords[:, :-1]       # (B, T-1, P, J, 3)
        acc = vel[:, 1:] - vel[:, :-1]             # (B, T-2, P, J, 3)

        # Pad to align dims with coords
        vel = F.pad(vel, (0, 0, 0, 0, 0, 0, 1, 0))
        acc = F.pad(acc, (0, 0, 0, 0, 0, 0, 2, 0))

        feats = torch.cat([coords, vel, acc], dim=-1)  # (B, T, P, J, 9)

        # Project joints
        x = feats.reshape(B * T * P * J, -1)
        x = self.joint_proj(x)
        x = x.view(B, T, P, J, self.embed_dim)

        # Confidence weighting
        conf = conf.unsqueeze(-1)
        conf = conf / (conf.sum(dim=3, keepdim=True) + 1e-8)
        x = (x * conf).sum(dim=3)  # (B, T, P, D)

        # Temporal convs (over T)
        x = x.mean(dim=2)  # average across persons for temporal conv
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.temporal_convs(x)  # (B, D, T)
        x = x.permute(0, 2, 1)  # (B, T, D)

        # Transformer
        x = self.temporal_transformer(x)  # (B, T, D)

        return x


# ===== Cross-Modal Fusion =====
class CrossModalFusion(nn.Module):
    def __init__(self, dim_rgb=400, dim_pose=256, hidden=512):
        super().__init__()
        self.query_rgb = nn.Linear(dim_rgb, hidden)
        self.key_pose = nn.Linear(dim_pose, hidden)
        self.value_pose = nn.Linear(dim_pose, hidden)
        self.proj = nn.Linear(hidden, dim_rgb)

    def forward(self, rgb, pose):
        """
        rgb: (B, T, D_rgb)
        pose: (B, T, D_pose)
        """
        q = self.query_rgb(rgb)       # (B, T, H)
        k = self.key_pose(pose)       # (B, T, H)
        v = self.value_pose(pose)     # (B, T, H)

        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5), dim=-1)
        out = torch.bmm(attn, v)  # (B, T, H)
        return rgb + self.proj(out)


# ===== Attention Pooling =====
class AttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        weights = F.softmax(self.attn(x), dim=1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)         # (B, D)
        return pooled


@MODEL_REGISTRY.register()
class ImprovedFusionCUENet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        super().__init__()

        self.cfg = cfg

        use_checkpoint = cfg.MODEL.USE_CHECKPOINT
        checkpoint_num = cfg.MODEL.CHECKPOINT_NUM
        num_classes = cfg.MODEL.NUM_CLASSES 
        t_size = cfg.DATA.NUM_FRAMES

        backbone = cfg.UNIFORMERV2.BACKBONE
        n_layers = cfg.UNIFORMERV2.N_LAYERS
        n_dim = cfg.UNIFORMERV2.N_DIM
        n_head = cfg.UNIFORMERV2.N_HEAD
        mlp_factor = cfg.UNIFORMERV2.MLP_FACTOR
        backbone_drop_path_rate = cfg.UNIFORMERV2.BACKBONE_DROP_PATH_RATE
        drop_path_rate = cfg.UNIFORMERV2.DROP_PATH_RATE
        mlp_dropout = cfg.UNIFORMERV2.MLP_DROPOUT
        cls_dropout = cfg.UNIFORMERV2.CLS_DROPOUT
        return_list = cfg.UNIFORMERV2.RETURN_LIST

        temporal_downsample = cfg.UNIFORMERV2.TEMPORAL_DOWNSAMPLE
        dw_reduction = cfg.UNIFORMERV2.DW_REDUCTION
        no_lmhra = cfg.UNIFORMERV2.NO_LMHRA
        double_lmhra = cfg.UNIFORMERV2.DOUBLE_LMHRA

        frozen = cfg.UNIFORMERV2.FROZEN

        # pre-trained from CLIP
        self.use_pose = True

        # === Setup RGB Encoder ===
        self.backbone = model.__dict__[backbone](
            use_checkpoint=use_checkpoint,
            checkpoint_num=checkpoint_num,
            t_size=t_size,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate, 
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list, 
            n_layers=n_layers, 
            n_dim=n_dim, 
            n_head=n_head, 
            mlp_factor=mlp_factor, 
            drop_path_rate=drop_path_rate, 
            mlp_dropout=mlp_dropout, 
            cls_dropout=cls_dropout, 
            num_classes=400,
            frozen=frozen,
        )

        if cfg.UNIFORMERV2.PRETRAIN != '':
            logger.info(f'Loading pretrained model from {cfg.UNIFORMERV2.PRETRAIN}')
            state_dict = torch.load(cfg.UNIFORMERV2.PRETRAIN, map_location='cpu')['model_state']

            if cfg.UNIFORMERV2.DELETE_SPECIAL_HEAD or \
               state_dict['backbone.transformer.proj.2.weight'].shape[0] != cfg.MODEL.NUM_CLASSES:
                logger.info('Deleting classification head from state dict')
                del state_dict['backbone.transformer.proj.2.weight']
                del state_dict['backbone.transformer.proj.2.bias']

            self.backbone.load_state_dict(state_dict, strict=False)

        # === Pose Encoder ===
        # Pose branch
        if self.use_pose:
            self.pose_encoder = PoseTemporalEncoder(
                num_joints=cfg.DATA.NUM_JOINTS,
                in_dim=3,
                embed_dim=256,
                temporal_layers=2
            )

            self.pose_projection = nn.Linear(256, 400)
            self.cross_fusion = CrossModalFusion(400, 400, hidden=512)

        # Classifier with attention pooling
        self.attn_pool = AttentionPooling(400, hidden=128)
        self.classifier = nn.Sequential(
            nn.Linear(400, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, cfg.MODEL.NUM_CLASSES)
        )

    def forward(self, video_tensor, pose_tensor=None):
        B = video_tensor.shape[0]
        V = 1
        if video_tensor.dim() == 6:
            V = video_tensor.shape[1]
            video_tensor = video_tensor.view(B * V, *video_tensor.shape[2:])

        # RGB branch
        rgb_feats = self.backbone(video_tensor)  # (B*V, T, D) or (B*V, D)

        if rgb_feats.ndim == 2:
            rgb_feats = rgb_feats.unsqueeze(1)  # (B, 1, D)

        # Pose branch
        if self.use_pose and pose_tensor is not None:
            if V > 1:
                B_orig, V_orig, T, P, J, D = pose_tensor.shape
                pose_tensor = pose_tensor.view(B_orig * V_orig, T, P, J, D)

            pose_feats = self.pose_encoder(pose_tensor)   # (B*V, T, D)
            pose_feats = self.pose_projection(pose_feats) # match RGB dim

            fused_feats = self.cross_fusion(rgb_feats, pose_feats)  # (B*V, T, D)
        else:
            fused_feats = rgb_feats

        # Attention pooling
        pooled_feats = self.attn_pool(fused_feats)  # (B*V, D)

        # Classify
        out = self.classifier(pooled_feats)

        if V > 1:
            out = out.view(B, V, -1).mean(dim=1)

        return out