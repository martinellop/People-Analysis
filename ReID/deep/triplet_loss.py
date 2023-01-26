import torch
import torch.nn.functional as F
from torch import nn

from common.distances import L2_distance, Cosine_distance
# Prende in ingresso un batch di immagini e i corrispettivi label

class TripletLoss(nn.Module):
    # for a better loss consider this code: https://github.com/michuanhaohao/reid-strong-baseline/blob/master/layers/triplet_loss.py --pit
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='euclidean'):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """

        Args:
        -------------- 
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)

        """
        n = inputs.size(0)

        # Seems that the sometimes the distance calculation method taken from this repo do wrong calculations.
        # while our methods are ok.

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = L2_distance(inputs, inputs)

        elif self.distance == 'cosine':
            dist = Cosine_distance(inputs, inputs)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        hardest_dist_ap, ap_idx = torch.max((dist * mask),dim=0)            # hardest positive examples
        max = torch.max(dist).item()
        hardest_dist_an, an_idx = torch.min((dist + (mask*max)), dim=0)     # hardest negative examples

        # Compute ranking hinge loss
        y = torch.ones_like(hardest_dist_an)
        loss = self.ranking_loss(hardest_dist_an, hardest_dist_ap, y)
        return loss