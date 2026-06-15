import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for dense prediction.
    Focal_Loss = -alpha * (1 - pt)^gamma * log(pt)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super().__init__()
        self.alpha         = alpha
        self.gamma         = gamma
        self.balance_index = balance_index
        self.smooth        = smooth
        self.size_average  = size_average

        if self.smooth is not None and not (0 <= self.smooth <= 1.0):
            raise ValueError("smooth must be in [0, 1]")

    def forward(self, logit, target):
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)  # [N, C, d1*d2]
            logit = logit.permute(0, 2, 1).contiguous().view(-1, num_class)

        target = torch.squeeze(target, 1).view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1) * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError("Unsupported alpha type")

        alpha = alpha.to(logit.device)

        idx = target.cpu().long()
        one_hot = torch.FloatTensor(target.size(0), num_class).zero_().scatter_(1, idx, 1)
        one_hot = one_hot.to(logit.device)

        if self.smooth:
            one_hot = torch.clamp(one_hot, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt     = (one_hot * logit).sum(1) + self.smooth
        logpt  = pt.log()
        alpha  = alpha[idx].squeeze()
        loss   = -alpha * torch.pow(1 - pt, self.gamma) * logpt

        return loss.mean() if self.size_average else loss.sum()


class BinaryDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def forward(self, pred, target):
        N       = target.size(0)
        smooth  = 1.0
        pred_f  = pred.view(N, -1)
        tgt_f   = target.view(N, -1)
        inter   = (pred_f * tgt_f).sum(1)
        dice    = (2 * inter + smooth) / (pred_f.sum(1) + tgt_f.sum(1) + smooth)
        return 1 - dice.mean()
