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

    def forward(self, inputs:torch.Tensor, targets:torch.Tensor):
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


class CenterLoss(nn.Module):
    """
    Center loss, which idea comes from: H. Luo et al., "A Strong Baseline and Batch Normalization Neck for Deep Person Re-Identification".

    Minimizing Center loss increases intra-class compactness. mINP metric should have a strong benefit by this loss minimization.
    """
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor):
        """
        Args:
        -------------- 
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """

        device = inputs.device
        batch_size, feat_dim = inputs.shape[0], inputs.shape[1]
        ids, inverse_indices = torch.unique(targets, return_inverse=True) #inverse_indices is (batch_size,)
        n_ids = ids.shape[0]

        #if there are as ids as batch elements, then every elment is unique --> loss will be 0.0
        if n_ids == batch_size:
            return torch.tensor((0)).to(device) 

        centers = torch.zeros(size=(n_ids, feat_dim), dtype=torch.float).to(device)
        calculated = torch.zeros(size=(n_ids,), dtype=torch.bool).to(device)
        distances = torch.zeros(size=(batch_size,), dtype=torch.float).to(device)

        for i in range(batch_size):
            if not calculated[inverse_indices[i]].item():           # we havent calculated yet the center relative to this id.
                same_id = (targets == targets[i])
                count = same_id.sum()                               # how many elements have this id?
                same_id = torch.nonzero(same_id).flatten()

                points = inputs.index_select(0, same_id)
                centers[inverse_indices[i]] = points.sum(0) / count    # we have calculated the center of this id, let's store it.
                calculated[inverse_indices[i]] = True                                           # we mark this id as already calculated.
            
            # here we already have the center corresponding to this id. let's then calculate the distance from this point.
            distances[i] = L2_distance(inputs[i].unsqueeze(0), centers[inverse_indices[i]].unsqueeze(0), squared_distance=True)

        return 0.5 * distances.sum()   
