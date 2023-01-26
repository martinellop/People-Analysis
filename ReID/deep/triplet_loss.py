import torch
import torch.nn.functional as F
from torch import nn
from enum import Enum

from common.distances import L2_distance, Cosine_distance

class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.
    """

    class Distance_mode(Enum):
        EUCLIDEAN_L2 = 1
        COSINE = 2


    def __init__(self, margin=0.3, distance:Distance_mode=Distance_mode.EUCLIDEAN_L2):
        """
        Args:
            margin (float): margin for triplet.
            distance: which type of distance has to be used by loss function.
        """
        super(TripletLoss, self).__init__()
        self.distance = distance
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        -------------- 
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)

        """
        # Compute pairwise distance, replace by the official when merged
        if self.distance == TripletLoss.Distance_mode.EUCLIDEAN_L2:
            dist = L2_distance(inputs, inputs)

        elif self.distance == TripletLoss.Distance_mode.COSINE:
            dist = Cosine_distance(inputs, inputs)

        # For each anchor, find the hardest positive and negative
        n = targets.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        hardest_dist_ap, ap_idx = torch.max((dist * mask),dim=0)            # hardest positive examples
        max = torch.max(dist).item()
        hardest_dist_an, an_idx = torch.min((dist + (mask*max)), dim=0)     # hardest negative examples

        # Compute ranking hinge loss
        y = torch.ones_like(hardest_dist_an)
        loss = self.ranking_loss(hardest_dist_an, hardest_dist_ap, y)
        return loss