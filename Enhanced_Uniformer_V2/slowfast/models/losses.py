#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """Tversky Loss optimized for violence detection (class 0)"""
    def __init__(self, alpha=0.75, beta=0.25, smooth=1e-5, reduction='mean'):
        """
        Args:
            alpha: Controls penalty for false negatives (violence recall)
            beta: Controls penalty for false positives
            smooth: Prevents division by zero
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Handle both single and dual logit outputs
        if inputs.dim() > 1 and inputs.shape[1] == 2:
            # Dual-logit output: use class 0 probability (violent)
            probs = torch.softmax(inputs, dim=1)
            prob_violent = probs[:, 0]
        else:
            # Single-logit output: sigmoid for violence probability
            prob_violent = torch.sigmoid(inputs).squeeze()
        
        # Convert labels to violence probability (1 = violent)
        target_violent = 1 - targets.float()  # Original: 0=violent, 1=nonviolent
        
        tp = (prob_violent * target_violent).sum()
        fp = (prob_violent * (1 - target_violent)).sum()
        fn = ((1 - prob_violent) * target_violent).sum()
        
        numerator = tp + self.smooth
        denominator = numerator + self.alpha * fn + self.beta * fp + self.smooth
        loss = 1 - (numerator / denominator)
        
        if self.reduction == 'sum':
            return loss * inputs.size(0)
        return loss

class BalancedTverskyLoss(nn.Module):
    """Class-balanced Tversky Loss with violence focus"""
    def __init__(self, alpha_dict={0: 0.75, 1: 0.65}, beta=0.25, smooth=1e-5, reduction='mean'):
        """
        Args:
            alpha_dict: Class-specific alpha parameters
            beta: Shared beta parameter
            smooth: Prevents division by zero
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha_dict = alpha_dict
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Handle both single and dual logit outputs
        if inputs.dim() > 1 and inputs.shape[1] == 2:
            probs = torch.softmax(inputs, dim=1)
        else:
            # Convert to dual probabilities if single logit output
            prob_violent = torch.sigmoid(inputs).squeeze()
            probs = torch.stack([prob_violent, 1 - prob_violent], dim=1)
        
        total_loss = 0.0
        batch_size = inputs.size(0)
        
        for class_idx, alpha in self.alpha_dict.items():
            # Get probability for current class
            p = probs[:, class_idx]
            
            # Create target vector for this class
            class_mask = (targets == class_idx).float()
            
            tp = (p * class_mask).sum()
            fp = (p * (1 - class_mask)).sum()
            fn = ((1 - p) * class_mask).sum()
            
            numerator = tp + self.smooth
            denominator = numerator + alpha * fn + self.beta * fp + self.smooth
            class_loss = 1 - (numerator / denominator)
            
            # Weight by class proportion in batch
            class_weight = class_mask.mean() if class_mask.sum() > 0 else 0
            total_loss += class_weight * class_loss
        
        if self.reduction == 'sum':
            return total_loss * batch_size
        return total_loss
    
class CE_TLoss(nn.Module):
    """Combined Cross Entropy and Tversky Loss"""
    def __init__(self, alpha_dict={0: 0.75, 1: 0.65}, beta=0.25, smooth=1e-5, reduction='mean'):
        """
        Args:
            alpha: Controls penalty for false negatives (violence recall)
            beta: Controls penalty for false positives
            smooth: Prevents division by zero
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.tversky_loss = BalancedTverskyLoss(alpha_dict, beta, smooth, reduction)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, inputs, targets):
        t_loss = self.tversky_loss(inputs, targets)
        ce_loss = self.cross_entropy_loss(inputs, targets)
        return t_loss + ce_loss

class CE_FocalLoss(nn.Module):
    """Combined Cross Entropy and Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma, reduction)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        ce_loss = self.cross_entropy_loss(inputs, targets)
        return focal_loss + ce_loss
    
class Focal_TverskyLoss(nn.Module):
    """Combined Focal Loss and Tversky Loss"""
    def __init__(self, alpha=0.25, gamma=2.0, alpha_dict={0: 0.75, 1: 0.65}, beta=0.25, smooth=1e-5, reduction='mean'):
        """
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            alpha_dict: Class-specific Tversky loss alpha parameters
            beta: Shared Tversky loss beta parameter
            smooth: Prevents division by zero
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma, reduction)
        self.tversky_loss = BalancedTverskyLoss(alpha_dict, beta, smooth, reduction)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        t_loss = self.tversky_loss(inputs, targets)
        return focal_loss + t_loss
    

# Add to losses.py
class FocalLossBCE(nn.Module):
    """Binary-compatible Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError
# Add to losses.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N] with class indices
        log_probs = F.log_softmax(inputs, dim=1)  # [N, C]
        probs = torch.exp(log_probs)              # [N, C]
        
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()  # [N, C]
        
        focal_weight = (1 - probs) ** self.gamma
        loss = -self.alpha * focal_weight * log_probs * targets_one_hot
        loss = loss.sum(dim=1)  # sum over classes

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "focal_loss": FocalLoss,
    "focal_loss_bce": FocalLossBCE,
    "balanced_tversky_loss": BalancedTverskyLoss,
    "tversky_loss": TverskyLoss,
    "cross_entropy_tversky_loss": CE_TLoss,
    "cross_entropy_focal_loss": CE_FocalLoss,
    "focal_tversky_loss": Focal_TverskyLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
