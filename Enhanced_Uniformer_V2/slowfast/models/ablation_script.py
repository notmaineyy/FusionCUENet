import torch
import torch.nn as nn
import torch.nn.functional as F
from .uniformerv2 import Uniformerv2
import slowfast.models.uniformerv2_model as model
from .build import MODEL_REGISTRY
from transformers import DistilBertModel
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

# ======================================================================
# Pose Temporal Encoder (Enhanced for YOLO + MediaPipe pipeline)
# ======================================================================
class PoseTemporalEncoder(nn.Module):
    def __init__(self, num_joints, in_dim=3, embed_dim=256, temporal_layers=3):
        """
        Encodes multi-person poses over time with velocity/acceleration features
        and temporal transformer + conv layers for spatiotemporal fusion.
        """
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # Per-joint MLP (with velocity + acceleration)
        self.joint_proj = nn.Sequential(
            nn.Linear(in_dim * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Temporal conv layers for local motion smoothing
        self.temporal_convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=8),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ) for _ in range(temporal_layers)
        ])

        # Transformer for global temporal context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        """
        x: (B, T, P, J, 4) with last dim = (x, y, z, confidence)
        """
        original_shape = x.shape
        
        # Handle different input shapes
        if x.dim() == 6:  # (B, 1, T, P, J, 3) - squeeze out the extra dimension
            if x.shape[1] == 1:
                x = x.squeeze(1)  # -> (B, T, P, J, 3)
            else:
                raise ValueError(f"Unexpected 6D pose tensor shape: {x.shape}")
        coords = x[..., :3]
        conf = x[..., 3]

        B, T, P, J, _ = coords.shape

        # Compute velocity and acceleration (temporal derivatives)
        vel = coords[:, 1:] - coords[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]

        # Pad to match original time dimension
        vel = F.pad(vel, (0, 0, 0, 0, 0, 0, 1, 0))
        acc = F.pad(acc, (0, 0, 0, 0, 0, 0, 2, 0))

        # Concatenate features: [coords, vel, acc]
        feats = torch.cat([coords, vel, acc], dim=-1)  # (B, T, P, J, 9)
        feats = feats.view(B * T * P * J, -1)
        feats = self.joint_proj(feats).view(B, T, P, J, self.embed_dim)

        # Confidence-weighted joint pooling
        conf = conf.unsqueeze(-1)
        conf = conf / (conf.sum(dim=3, keepdim=True) + 1e-8)
        x = (feats * conf).sum(dim=3).mean(dim=2)  # (B, T, D)

        # Temporal modeling: conv -> transformer
        x = x.permute(0, 2, 1)      # (B, D, T)
        x = self.temporal_convs(x)  # (B, D, T)
        x = x.permute(0, 2, 1)      # (B, T, D)
        x = self.temporal_transformer(x)

        return x  # (B, T, D)


# ======================================================================
# Text Encoder (DistilBERT, frozen or trainable)
# ======================================================================
class TextEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(
            '/vol/bitbucket/sna21/distilbert-base-uncased'
        )
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # CLS token (B, 768)


# ======================================================================
# Cross-Modal Fusion: RGB ↔ Pose
# ======================================================================
class CrossModalFusion(nn.Module):
    def __init__(self, dim_rgb=400, dim_pose=400, hidden=512):
        super().__init__()
        self.query_rgb = nn.Linear(dim_rgb, hidden)
        self.key_pose = nn.Linear(dim_pose, hidden)
        self.value_pose = nn.Linear(dim_pose, hidden)
        self.proj = nn.Linear(hidden, dim_rgb)
        self.norm = nn.LayerNorm(dim_rgb)

    def forward(self, rgb, pose):
        # rgb: (B,T,dim_rgb), pose: (B,T,dim_pose)
        q = self.query_rgb(rgb)
        k = self.key_pose(pose)
        v = self.value_pose(pose)

        attn = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5), dim=-1
        )
        out = torch.bmm(attn, v)
        return self.norm(rgb + self.proj(out))


# ======================================================================
# Text-Visual Fusion: RGB ↔ Text
# ======================================================================
class TextVisualFusion(nn.Module):
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

        if self.training:
            logger.info(f"[Fusion] Learnable text weight: {self.text_weight.item():.4f}")

        weighted_text = weighted_text.unsqueeze(1)  # (B, 1, hidden)
        attn_output, _ = self.attention(
            query=visual_proj, key=weighted_text, value=weighted_text, need_weights=False
        )
        fused = self.norm(visual_proj + attn_output)
        return visual_feats + self.out_proj(fused)


# ======================================================================
# Attention Pooling
# ======================================================================
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


# ======================================================================
# Main Fusion CUENet Model with Ablation Support
# ======================================================================
@MODEL_REGISTRY.register()
class BlipFusionRGBCUENet_v1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Ablation flags from config
        self.use_rgb = getattr(cfg.MODEL, 'USE_RGB', True)
        self.use_pose = getattr(cfg.MODEL, 'USE_POSE', True) 
        self.use_text = getattr(cfg.MODEL, 'USE_TEXT', True)
        
        # Log ablation configuration
        logger.info(f"Ablation Study Configuration:")
        logger.info(f"  RGB: {'ENABLED' if self.use_rgb else 'DISABLED'}")
        logger.info(f"  Pose: {'ENABLED' if self.use_pose else 'DISABLED'}")
        logger.info(f"  Text: {'ENABLED' if self.use_text else 'DISABLED'}")

        # Validate at least one modality is enabled
        if not any([self.use_rgb, self.use_pose, self.use_text]):
            raise ValueError("At least one modality (RGB, Pose, Text) must be enabled!")

        # Feature dimensions
        self.feature_dim = 400
        
        # RGB backbone (only initialize if needed)
        if self.use_rgb:
            backbone = cfg.UNIFORMERV2.BACKBONE
            self.backbone = model.__dict__[backbone](
                use_checkpoint=cfg.MODEL.USE_CHECKPOINT,
                checkpoint_num=cfg.MODEL.CHECKPOINT_NUM,
                t_size=cfg.DATA.NUM_FRAMES,
                dw_reduction=cfg.UNIFORMERV2.DW_REDUCTION,
                backbone_drop_path_rate=cfg.UNIFORMERV2.BACKBONE_DROP_PATH_RATE,
                temporal_downsample=cfg.UNIFORMERV2.TEMPORAL_DOWNSAMPLE,
                no_lmhra=cfg.UNIFORMERV2.NO_LMHRA,
                double_lmhra=cfg.UNIFORMERV2.DOUBLE_LMHRA,
                return_list=cfg.UNIFORMERV2.RETURN_LIST,
                n_layers=cfg.UNIFORMERV2.N_LAYERS,
                n_dim=cfg.UNIFORMERV2.N_DIM,
                n_head=cfg.UNIFORMERV2.N_HEAD,
                mlp_factor=cfg.UNIFORMERV2.MLP_FACTOR,
                drop_path_rate=cfg.UNIFORMERV2.DROP_PATH_RATE,
                mlp_dropout=cfg.UNIFORMERV2.MLP_DROPOUT,
                cls_dropout=cfg.UNIFORMERV2.CLS_DROPOUT,
                num_classes=self.feature_dim,
                frozen=cfg.UNIFORMERV2.FROZEN,
            )

            # Load pretrained weights for RGB backbone
            if cfg.UNIFORMERV2.PRETRAIN:
                logger.info(f'Loading pretrained RGB model from {cfg.UNIFORMERV2.PRETRAIN}')
                state_dict = torch.load(cfg.UNIFORMERV2.PRETRAIN, map_location='cpu')['model_state']
                if 'backbone.transformer.proj.2.weight' in state_dict and \
                    state_dict['backbone.transformer.proj.2.weight'].shape[0] != cfg.MODEL.NUM_CLASSES:
                    del state_dict['backbone.transformer.proj.2.weight']
                    del state_dict['backbone.transformer.proj.2.bias']
                self.backbone.load_state_dict(state_dict, strict=False)

        # Pose branch (only initialize if needed)
        if self.use_pose:
            self.pose_encoder = PoseTemporalEncoder(
                num_joints=cfg.DATA.NUM_JOINTS, 
                embed_dim=256
            )
            self.pose_projection = nn.Sequential(
                nn.Linear(256, self.feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Cross-modal fusion (only if both RGB and Pose are enabled)
            if self.use_rgb:
                self.cross_fusion = CrossModalFusion(
                    dim_rgb=self.feature_dim,
                    dim_pose=self.feature_dim
                )

        # Text branch (only initialize if needed)
        if self.use_text:
            self.text_encoder = TextEncoder(freeze=cfg.MODEL.FREEZE_TEXT_ENCODER)
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Text-visual fusion (only if visual modalities are enabled)
            if self.use_rgb or self.use_pose:
                self.text_visual_fusion = TextVisualFusion(
                    dim_visual=self.feature_dim,
                    dim_text=self.feature_dim
                )

        # Feature combination layer for multi-modal scenarios
        modality_count = sum([self.use_rgb, self.use_pose, self.use_text])
        if modality_count > 1:
            self.feature_combiner = nn.Sequential(
                nn.Linear(self.feature_dim * modality_count, self.feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.LayerNorm(self.feature_dim)
            )

        # Classification head
        self.attn_pool = AttentionPooling(self.feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, cfg.MODEL.NUM_CLASSES)
        )

    def forward(self, video_tensor=None, pose_tensor=None, input_ids=None, attention_mask=None):
        """
        Forward pass supporting different modality combinations based on config
        """
        # Determine batch size from available inputs
        if video_tensor is not None:
            B = video_tensor.shape[0]
        elif pose_tensor is not None:
            B = pose_tensor.shape[0]
        elif input_ids is not None:
            B = input_ids.shape[0]
        else:
            raise ValueError("At least one input modality must be provided!")

        # Handle multi-view inputs
        V = 1
        if video_tensor is not None and video_tensor.dim() == 6:  # Multi-view
            V = video_tensor.shape[1]
            video_tensor = video_tensor.view(B * V, *video_tensor.shape[2:])
            
        if pose_tensor is not None:
            if pose_tensor.dim() == 4:  # (T, P, J, D) case
                pose_tensor = pose_tensor.unsqueeze(0).unsqueeze(0)  # Add B and V dims
            elif pose_tensor.dim() == 5:  # (B, T, P, J, D)
                pose_tensor = pose_tensor.unsqueeze(1)  # Add V dim
            if V > 1:
                pose_tensor = pose_tensor.view(B * V, *pose_tensor.shape[2:])
                
        if input_ids is not None and V > 1:
            input_ids = input_ids.repeat_interleave(V, dim=0)
            attention_mask = attention_mask.repeat_interleave(V, dim=0)

        # Feature extraction per modality
        features_list = []
        
        # RGB features
        rgb_feats = None
        if self.use_rgb:
            if video_tensor is None:
                raise ValueError("RGB is enabled but video_tensor is None!")
            rgb_feats = self.backbone(video_tensor)
            if rgb_feats.ndim == 2:
                rgb_feats = rgb_feats.unsqueeze(1)  # Add temporal dim
            features_list.append(rgb_feats.mean(dim=1))  # Global temporal pooling

        # Pose features  
        pose_feats = None
        if self.use_pose:
            if pose_tensor is None:
                raise ValueError("Pose is enabled but pose_tensor is None!")
            pose_encoded = self.pose_encoder(pose_tensor)
            pose_feats = self.pose_projection(pose_encoded)
            features_list.append(pose_feats.mean(dim=1))  # Global temporal pooling

        # Text features
        text_feats = None
        if self.use_text:
            if input_ids is None or attention_mask is None:
                raise ValueError("Text is enabled but input_ids/attention_mask is None!")
            with torch.set_grad_enabled(not self.cfg.MODEL.FREEZE_TEXT_ENCODER):
                text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_feats = self.text_projection(text_encoded)
            features_list.append(text_feats)

        # Feature fusion strategy
        if len(features_list) == 1:
            # Single modality - use as is
            final_feats = features_list[0]
            if final_feats.ndim == 3:  # Has temporal dimension
                final_feats = final_feats.mean(dim=1)
                
        elif len(features_list) == 2:
            # Two modalities - use dedicated fusion modules
            if self.use_rgb and self.use_pose:
                # RGB-Pose fusion
                fused = self.cross_fusion(rgb_feats, pose_feats)
                final_feats = fused.mean(dim=1)
                
            elif (self.use_rgb or self.use_pose) and self.use_text:
                # Visual-Text fusion
                visual_feats = rgb_feats if self.use_rgb else pose_feats
                fused = self.text_visual_fusion(visual_feats, text_feats)
                final_feats = fused.mean(dim=1)
                
            else:
                # Fallback: simple concatenation
                final_feats = torch.cat(features_list, dim=-1)
                final_feats = self.feature_combiner(final_feats)
                
        else:
            # Three modalities - hierarchical fusion
            # First fuse RGB+Pose if both available
            if self.use_rgb and self.use_pose:
                visual_fused = self.cross_fusion(rgb_feats, pose_feats)
            else:
                visual_fused = rgb_feats if self.use_rgb else pose_feats
                
            # Then fuse with text
            if self.use_text:
                final_fused = self.text_visual_fusion(visual_fused, text_feats)
                final_feats = final_fused.mean(dim=1)
            else:
                final_feats = visual_fused.mean(dim=1)

        # Classification
        if final_feats.ndim == 3:  # Still has temporal dim
            pooled = self.attn_pool(final_feats)
        else:
            pooled = final_feats
            
        out = self.classifier(pooled)

        # Handle multi-view aggregation
        if V > 1:
            out = out.view(B, V, -1).mean(dim=1)

        return out