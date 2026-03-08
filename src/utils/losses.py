"""
Loss functions for Re-ID.
Includes Triplet Loss with semi-hard mining.
"""
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        inputs = inputs.float()  # ensure FP32 for distance computation (safe under autocast)
        # Compute pairwise distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative (vectorized — no Python loop)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # Hardest positive: max distance among same-identity pairs
        dist_ap = (dist * mask.float()).max(dim=1)[0]
        # Hardest negative: min distance among different-identity pairs (mask out positives with large value)
        dist_an = (dist + mask.float() * 1e9).min(dim=1)[0]

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)