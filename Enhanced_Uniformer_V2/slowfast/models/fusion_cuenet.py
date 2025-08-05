# models/fusion_cuenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .uniformerv2 import Uniformerv2
import slowfast.models.uniformerv2_model as model
from .build import MODEL_REGISTRY
from transformers import DistilBertModel

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

# ===== Pose Temporal Encoder =====
class PoseTemporalEncoder(nn.Module):
    def __init__(self, num_joints, in_dim=3, embed_dim=256, temporal_layers=2):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # Joint embedding
        self.joint_proj = nn.Sequential(
            nn.Linear(in_dim * 3, 128),
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

        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=512, dropout=0.2, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        coords = x[..., :3]
        conf = x[..., 3]
        B, T, P, J, _ = coords.shape

        vel = coords[:, 1:] - coords[:, :-1]  # (B,T-1,P,J,3)
        acc = vel[:, 1:] - vel[:, :-1]        # (B,T-2,P,J,3)

        # Pad properly at start (keep time axis aligned)
        vel = F.pad(vel, (0, 0, 0, 0, 0, 0, 1, 0))   # (B,T,P,J,3)
        acc = F.pad(acc, (0, 0, 0, 0, 0, 0, 2, 0))   # (B,T,P,J,3)

        feats = torch.cat([coords, vel, acc], dim=-1)  # (B,T,P,J,9)
        x = feats.reshape(B * T * P * J, -1)
        x = self.joint_proj(x).view(B, T, P, J, self.embed_dim)

        conf = conf.unsqueeze(-1)
        conf = conf / (conf.sum(dim=3, keepdim=True) + 1e-8)
        x = (x * conf).sum(dim=3).mean(dim=2)  # (B,T,D)

        x = x.permute(0, 2, 1)
        x = self.temporal_convs(x).permute(0, 2, 1)
        return self.temporal_transformer(x)

# ===== Text Encoder =====
class TextEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('/vol/bitbucket/sna21/distilbert-base-uncased')
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        return outputs.last_hidden_state[:, 0]  # CLS token

# ===== Cross-Modal Fusion =====
class CrossModalFusion(nn.Module):
    def __init__(self, dim_rgb=400, dim_pose=256, hidden=512):
        super().__init__()
        self.query_rgb = nn.Linear(dim_rgb, hidden)
        self.key_pose = nn.Linear(dim_pose, hidden)
        self.value_pose = nn.Linear(dim_pose, hidden)
        self.proj = nn.Linear(hidden, dim_rgb)

    def forward(self, rgb, pose):
        # rgb: (B,T,dim_rgb), pose: (B,T,dim_pose)
        q = self.query_rgb(rgb)
        k = self.key_pose(pose)
        v = self.value_pose(pose)

        # Correct attention computation
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5), dim=-1)
        out = torch.bmm(attn, v)
        return rgb + self.proj(out)

# ===== Text-Visual Fusion =====
class TextVisualFusion(nn.Module):
    def __init__(self, dim_visual=400, dim_text=400, hidden=512):
        super().__init__()
        self.visual_proj = nn.Linear(dim_visual, hidden)
        self.text_proj = nn.Linear(dim_text, hidden)
        self.attention = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim_visual)

    def forward(self, visual_feats, text_feats):
        """
        visual_feats: (B, T, D_visual)
        text_feats: (B, D_text)
        Returns: (B, T, D_visual)
        """
        # Project visual features
        visual_proj = self.visual_proj(visual_feats)  # (B, T, hidden)
        
        # Project text features and add sequence dimension
        text_proj = self.text_proj(text_feats).unsqueeze(1)  # (B, 1, hidden)
        
        # Cross-attention
        attn_output, _ = self.attention(
            query=visual_proj,
            key=text_proj,
            value=text_proj,
            need_weights=False
        )
        
        # Residual connection
        fused = self.norm(visual_proj + attn_output)
        return visual_feats + self.out_proj(fused)
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
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)

@MODEL_REGISTRY.register()
class FusionCUENet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Modality flags
        self.use_pose = False
        self.use_text = True

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
        self.use_pose = False

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

        # Load pretrained weights if available
        if cfg.UNIFORMERV2.PRETRAIN:
            logger.info(f'Loading pretrained model from {cfg.UNIFORMERV2.PRETRAIN}')
            state_dict = torch.load(cfg.UNIFORMERV2.PRETRAIN, map_location='cpu')['model_state']
            # Handle head mismatch
            if state_dict['backbone.transformer.proj.2.weight'].shape[0] != num_classes:
                del state_dict['backbone.transformer.proj.2.weight']
                del state_dict['backbone.transformer.proj.2.bias']
            self.backbone.load_state_dict(state_dict, strict=False)

        # === Pose Encoder ===
        if self.use_pose:
            self.pose_encoder = PoseTemporalEncoder(
                num_joints=cfg.DATA.NUM_JOINTS,
                embed_dim=256
            )
            self.pose_projection = nn.Linear(256, 400)
            self.cross_fusion = CrossModalFusion()
            

        # === Text Encoder ===
        if self.use_text:
            self.text_encoder = TextEncoder(freeze=cfg.MODEL.FREEZE_TEXT_ENCODER)
            self.text_projection = nn.Sequential(
                nn.Linear(768, 400),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.text_visual_fusion = TextVisualFusion()

        # === Classification Head ===
        self.attn_pool = AttentionPooling(400)
        self.classifier = nn.Sequential(
            nn.Linear(400, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, video_tensor, input_ids=None, attention_mask=None, pose_tensor=None):
        B = video_tensor.shape[0]
        V = 1
        if video_tensor.dim() == 6:  # Multi-view: (B, V, C, T, H, W)
            V = video_tensor.shape[1]
            video_tensor = video_tensor.view(B * V, *video_tensor.shape[2:])
            if pose_tensor is not None:
                pose_tensor = pose_tensor.view(B * V, *pose_tensor.shape[2:])
            if input_ids is not None:
                input_ids = input_ids.repeat_interleave(V, dim=0)
                attention_mask = attention_mask.repeat_interleave(V, dim=0)

        # RGB backbone
        rgb_feats = self.backbone(video_tensor)
        if rgb_feats.ndim == 2:
            rgb_feats = rgb_feats.unsqueeze(1)  # Add time dim if missing

        # Pose fusion (optional)
        if self.use_pose and pose_tensor is not None:
            pose_feats = self.pose_projection(self.pose_encoder(pose_tensor))
            fused_feats = self.cross_fusion(rgb_feats, pose_feats)
        else:
            fused_feats = rgb_feats

        # Text encoding + fusion
        # Text encoding + fusion
        if self.use_text and input_ids is not None and attention_mask is not None:
            with torch.set_grad_enabled(not self.cfg.MODEL.FREEZE_TEXT_ENCODER):
                text_feats = self.text_encoder(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )  # (B*V, 768)
                text_feats = self.text_projection(text_feats)  # (B*V, 400)
            fused_feats = self.text_visual_fusion(fused_feats, text_feats)

        # Pooling and classification
        pooled = self.attn_pool(fused_feats)
        out = self.classifier(pooled)

        if V > 1:
            out = out.view(B, V, -1).mean(dim=1)

        return out
