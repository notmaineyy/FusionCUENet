import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .uniformerv2 import Uniformerv2
import slowfast.models.uniformerv2_model as model_zoo
from .build import MODEL_REGISTRY
from transformers import DistilBertModel
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

# ======================================================================
# 1. LoRA Implementation
# ======================================================================
class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA Matrices
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        orig_out = self.original_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return orig_out + lora_out

def inject_lora(model, r=8):
    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, module in model.named_modules():
        # Inject into Query and Value projections in Attention blocks
        if isinstance(module, nn.Linear) and any(k in name for k in ['q_lin', 'v_lin', 'q_proj', 'v_proj']):
            lora_layer = LoRALinear(module, r=r)
            
            # Helper to set nested attribute
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
                
            setattr(parent, child_name, lora_layer)
            
    params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LoRA Injected. Trainable Params: {params_before} -> {params_after}")

# ======================================================================
# 2. Soft Prompting Text Encoder (Fixed Transformer Call)
# ======================================================================
class SoftPromptTextEncoder(nn.Module):
    def __init__(self, n_ctx=16, ctx_init=None, freeze_bert=True):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('/vol/bitbucket/sna21/distilbert-base-uncased')
        
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
                
        hidden_dim = self.model.config.dim
        self.n_ctx = n_ctx
        self.ctx_vectors = nn.Parameter(torch.empty(n_ctx, hidden_dim))
        
        if ctx_init:
            nn.init.normal_(self.ctx_vectors, std=0.02)
        else:
            nn.init.normal_(self.ctx_vectors, std=0.02)
            
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        B = input_ids.shape[0]
        
        # 1. Get static embeddings
        static_embeds = self.model.embeddings(input_ids) # (B, L, D)
        
        # 2. Add Soft Prompts
        prompts = self.ctx_vectors.unsqueeze(0).expand(B, -1, -1) # (B, n_ctx, D)
        combined_embeds = torch.cat([prompts, static_embeds], dim=1) # (B, n_ctx + L, D)
        
        # 3. Extend Mask
        prompt_mask = torch.ones(B, self.n_ctx).to(input_ids.device)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # 4. Forward (Standard Interface)
        out = self.model(inputs_embeds=combined_embeds, attention_mask=combined_mask)
        last_hidden = out.last_hidden_state
        
        # 5. Return feature corresponding to original [CLS] position (after prompts)
        cls_token = last_hidden[:, self.n_ctx, :]
        return cls_token


# Simple / standard text encoder (no soft prompts). Returns CLS embedding.
class BasicTextEncoder(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        # Reuse the same DistilBert weights used for soft prompting
        self.model = DistilBertModel.from_pretrained('/vol/bitbucket/sna21/distilbert-base-uncased')
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT does not provide pooled_output by default; use first token ([CLS])
        cls_token = out.last_hidden_state[:, 0, :]
        return cls_token

# ======================================================================
# 3. Pose Temporal Encoder (Fixed Dynamic Padding)
# ======================================================================
class PoseTemporalEncoder(nn.Module):
    def __init__(self, num_joints, in_dim=3, embed_dim=256, temporal_layers=3, use_velocity=True):
        """
        Encodes multi-person poses over time with velocity/acceleration features
        and temporal transformer + conv layers for spatiotemporal fusion.
        """
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.use_velocity = use_velocity

        # Per-joint MLP (with velocity + acceleration)
        self.joint_proj = nn.Sequential(
            nn.Linear(in_dim * (1 + 2 * self.use_velocity), 256),
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
        if x.dim() == 4:
            # (B, T, J, 4) → add person dimension
            x = x.unsqueeze(2)  # (B, T, 1, J, 4)

        elif x.dim() == 5:
            pass  # already correct

        else:
            raise ValueError(f"Unsupported pose tensor shape: {x.shape}")

        coords = x[..., :3]
        conf   = x[..., 3]

        B, T, P, J, _ = coords.shape

        # Compute velocity and acceleration (temporal derivatives)
        if self.use_velocity:
            vel = coords[:, 1:] - coords[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]

            # Pad to match original time dimension
            vel = F.pad(vel, (0, 0, 0, 0, 0, 0, 1, 0))
            acc = F.pad(acc, (0, 0, 0, 0, 0, 0, 2, 0))

            # Concatenate features: [coords, vel, acc]
            feats = torch.cat([coords, vel, acc], dim=-1)  # (B, T, P, J, 9)
        else:
            feats = coords  # (B, T, P, J, 3)
        feats = feats.view(B * T * P * J, -1)
        feats = self.joint_proj(feats).view(B, T, P, J, self.embed_dim)
        conf = conf.unsqueeze(-1)
        conf = conf / (conf.sum(dim=3, keepdim=True) + 1e-8)
        x = (feats * conf).sum(dim=3).mean(dim=2)  # (B, T, D)sum(dim=3).mean(dim=2)  # (B, T, D)

        # Temporal modeling: conv -> transformer
        x = x.permute(0, 2, 1)      # (B, D, T)
        x = self.temporal_convs(x)  # (B, D, T)
        x = x.permute(0, 2, 1)      # (B, T, D)
        x = self.temporal_transformer(x)

        return x  # (B, T, D)

class CrossModalFusion(nn.Module):
    def __init__(self, dim_rgb=400, dim_pose=400, hidden=512):
        super().__init__()
        self.query_rgb = nn.Linear(dim_rgb, hidden)
        self.key_pose = nn.Linear(dim_pose, hidden)
        self.value_pose = nn.Linear(dim_pose, hidden)
        self.proj = nn.Linear(hidden, dim_rgb)
        self.norm = nn.LayerNorm(dim_rgb)
    def forward(self, rgb, pose):
        q, k, v = self.query_rgb(rgb), self.key_pose(pose), self.value_pose(pose)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.size(-1)**0.5), dim=-1)
        return self.norm(rgb + self.proj(torch.bmm(attn, v)))

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
        v = self.visual_proj(visual_feats)
        t = self.text_norm(self.text_proj(text_feats)) * self.text_weight.clamp(0.1, 10.0)
        t = t.unsqueeze(1)
        attn, _ = self.attention(v, t, t, need_weights=False)
        return visual_feats + self.out_proj(self.norm(v + attn))

class AttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)

class ModalityWeighting(nn.Module):
    def __init__(self, use_rgb=True, use_pose=True, use_text=True):
        super().__init__()
        self.use_rgb = use_rgb
        self.use_pose = use_pose
        self.use_text = use_text
        weights = []
        if self.use_rgb:
            weights.append(nn.Parameter(torch.tensor(1.0)))
        if self.use_pose:
            weights.append(nn.Parameter(torch.tensor(1.0)))
        if self.use_text:
            weights.append(nn.Parameter(torch.tensor(1.0)))
        self.modality_weights = nn.Parameter(torch.stack(weights))

    def forward(self, features_list):
        norm_weights = torch.softmax(self.modality_weights, dim=0)
        weighted_feats = []
        for w, f in zip(norm_weights, features_list):
            weighted_feats.append(w * f)
        fused_feats = torch.stack(weighted_feats, dim=0).sum(dim=0)
        return fused_feats, norm_weights

# ======================================================================
# Main Model (Fixed Fusion Logic)
# ======================================================================
@MODEL_REGISTRY.register()
class FusionCUENet_eccv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_rgb = getattr(cfg.MODEL, 'USE_RGB', True)
        self.use_pose = getattr(cfg.MODEL, 'USE_POSE', True)
        self.use_text = getattr(cfg.MODEL, 'USE_TEXT', True)
        self.feature_dim = 400
        # Read common feature flags from config early so we can log them
        self.lora_enabled = getattr(cfg.MODEL, 'ENABLE_LORA', False)
        self.soft_prompting_enabled = getattr(cfg.MODEL, 'SOFT_PROMPT', True)
        self.freeze_text_encoder = getattr(cfg.MODEL, 'FREEZE_TEXT_ENCODER', True)

        logger.info(
            f"Model init flags - USE_RGB={self.use_rgb}, USE_POSE={self.use_pose}, USE_TEXT={self.use_text}, "
            f"ENABLE_LORA={self.lora_enabled}, SOFT_PROMPT={self.soft_prompting_enabled}, "
            f"FREEZE_TEXT_ENCODER={self.freeze_text_encoder}"
        )

        # 1. Visual Stream (RGB)
        if self.use_rgb:
            backbone_name = cfg.UNIFORMERV2.BACKBONE
            self.backbone = getattr(model_zoo, backbone_name)(
                num_classes=self.feature_dim,
                t_size=cfg.DATA.NUM_FRAMES,
                frozen=False
            )
            # Freeze and Inject LoRA
            for param in self.backbone.parameters():
                param.requires_grad = False
            if self.lora_enabled:
                inject_lora(self.backbone, r=8)
            else:
                logger.info("LoRA not enabled for backbone (cfg.MODEL.ENABLE_LORA=False)")

        # 2. Kinematic Stream (Pose)
        if self.use_pose:
            self.pose_encoder = PoseTemporalEncoder(num_joints=cfg.DATA.NUM_JOINTS, embed_dim=256)
            self.pose_projection = nn.Sequential(nn.Linear(256, self.feature_dim), nn.ReLU(), nn.Dropout(0.3))
            if self.use_rgb: self.cross_fusion = CrossModalFusion(dim_rgb=self.feature_dim, dim_pose=self.feature_dim)

        # 3. Semantic Stream (Text)
        if self.use_text:
            # Use the stored flags read earlier
            use_soft = bool(self.soft_prompting_enabled)
            freeze_text = bool(self.freeze_text_encoder)
            if use_soft:
                n_ctx = getattr(cfg.MODEL, 'SOFT_PROMPT_N_CTX', 16)
                self.text_encoder = SoftPromptTextEncoder(n_ctx=n_ctx, freeze_bert=freeze_text)
            else:
                # Use a standard text encoder (no soft prompts)
                self.text_encoder = BasicTextEncoder(freeze_bert=freeze_text)

            # Projection from text embedding dim (DistilBERT: 768) -> feature dim
            self.text_projection = nn.Sequential(nn.Linear(768, self.feature_dim), nn.ReLU(), nn.Dropout(0.3))
            if self.use_rgb or self.use_pose:
                self.text_visual_fusion = TextVisualFusion(dim_visual=self.feature_dim, dim_text=self.feature_dim)

        self.modality_weighting = ModalityWeighting(self.use_rgb, self.use_pose, self.use_text)
        self.attn_pool = AttentionPooling(self.feature_dim)
        
        self.classifier_proj = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.LayerNorm(512)
        )
        self.classifier_head = nn.Linear(512, cfg.MODEL.NUM_CLASSES)

    def forward(self, video_tensor=None, pose_tensor=None, input_ids=None, attention_mask=None, return_features=False):
        # 1. Dimension Handling
        def unflatten_views(x, B):
            if x.shape[0] == B:
                return x
            V = x.shape[0] // B
            return x.view(B, V, -1).mean(dim=1)

        B = video_tensor.shape[0] if video_tensor is not None else input_ids.shape[0]
        
        # Flatten Multi-view batch if needed
        if video_tensor is not None and video_tensor.dim() == 6:
            video_tensor = video_tensor.view(B * video_tensor.shape[1], *video_tensor.shape[2:])
            if pose_tensor is not None: 
                pose_tensor = pose_tensor.view(B * pose_tensor.shape[1], *pose_tensor.shape[2:])
            if input_ids is not None: 
                input_ids = input_ids.repeat_interleave(video_tensor.shape[0] // input_ids.shape[0], dim=0)
                attention_mask = attention_mask.repeat_interleave(video_tensor.shape[0] // attention_mask.shape[0], dim=0)

        features_list = []
        
        # 2. Extract RGB
        if self.use_rgb:
            rgb = self.backbone(video_tensor) # (B, 400)
            if rgb.ndim == 2: 
                rgb = rgb.unsqueeze(1)
            rgb = rgb.mean(dim=1)           # (B*V, 400)
            rgb = unflatten_views(rgb, B)   # (B, 400)
            features_list.append(rgb)


        # 3. Extract Pose
        if self.use_pose:
            pose = self.pose_encoder(pose_tensor) # (B, T, 256)
            pose = self.pose_projection(pose)     # (B, T, 400)
            pose = pose.mean(dim=1)         # (B*V, 400)
            pose = unflatten_views(pose, B) # (B, 400)
            features_list.append(pose)


        # 4. Extract Text
        if self.use_text:
            with torch.set_grad_enabled(not self.cfg.MODEL.FREEZE_TEXT_ENCODER):
                text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text = self.text_projection(text_encoded)
            #text = self.text_projection(self.text_encoder(input_ids, attention_mask))
            features_list.append(text)

        # 5. Fuse
        final_feats, modality_wts = self.modality_weighting(features_list)
        if self.training:
            logger.info(f"Learned modality weights: {modality_wts.detach().cpu().numpy()}")

        pooled = self.attn_pool(final_feats.unsqueeze(1) if final_feats.ndim == 2 else final_feats)
        features = self.classifier_proj(pooled)
        out = self.classifier_head(features)
        
        if return_features:
            return out, features
        return out