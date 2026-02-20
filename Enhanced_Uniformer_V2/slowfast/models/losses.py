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

class dynamic_CE_TLoss(nn.Module):
    def __init__(self, num_classes=2, init_alpha=0.5, init_lambda=0.5, beta=0.25, smooth=1e-5, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=reduction)

        # Learnable λ for loss weighting (sigmoid to [0,1])
        self.lambda_logit = nn.Parameter(torch.tensor(float(init_lambda)).logit())

        # Learnable α for each class (sigmoid to [0,1])
        self.alpha_logits = nn.Parameter(torch.full((num_classes,), float(init_alpha)).logit())

        self.num_classes = num_classes

    def forward(self, inputs, targets):
        """
        inputs: (B, C)
        targets: (B,)
        """
        ce_loss = self.cross_entropy_loss(inputs, targets)

        # Convert logits to usable [0, 1] values
        lambda_val = torch.sigmoid(self.lambda_logit)
        alpha_vals = torch.sigmoid(self.alpha_logits)  # (C,)
        targets = targets.to(alpha_vals.device)        # ✅ Fix device mismatch
        alpha = alpha_vals[targets]                    # (B,)


        # Prepare per-sample alpha
        if inputs.dim() == 2:
            # Classification shape: (B, C), targets: (B,)
            alpha = alpha_vals[targets]  # (B,)
        else:
            raise ValueError("Unsupported input shape for CE_TLoss")

        # Compute Tversky Loss manually with dynamic alpha
        probs = torch.softmax(inputs, dim=1)
        true_1_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float().to(inputs.device)

        # TP, FP, FN
        dims = list(range(1, true_1_hot.dim()))
        TP = (probs * true_1_hot).sum(dim=dims)
        FP = (probs * (1 - true_1_hot)).sum(dim=dims)
        FN = ((1 - probs) * true_1_hot).sum(dim=dims)

        # alpha: penalty for FN (higher → more recall emphasis)
        alpha = alpha.to(inputs.device)
        T_loss = (TP + self.smooth) / (TP + alpha * FN + self.beta * FP + self.smooth)
        tversky_loss = 1 - T_loss.mean()

        # Final combined loss
        total_loss = lambda_val * ce_loss + (1 - lambda_val) * tversky_loss
        return total_loss


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

class CRDLoss(nn.Module):
    def __init__(self, opt):
        # 1. Embeddings (Projectors)
        # Maps student/teacher features to a common dimension (e.g., 128)
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)

        # 2. Memory Bank
        # Stores past features to provide a large number of negative samples
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)

        # 3. NCE Loss Criteria
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        # Project features
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)

        # Get dot products with memory bank (positives and negatives)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)

        # Calculate NCE loss
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        return s_loss + t_loss
    
class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj
    
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
    "dynamic_ce_tloss": dynamic_CE_TLoss,
    "crd_loss": CRDLoss,
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
