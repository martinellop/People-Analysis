import torch


def calculate_CMC(distances:torch.Tensor, query_ids:torch.Tensor, gallery_ids:torch.Tensor, rank:int):
    """
    Args:

        distances:      matrix [k1, k2], where `Distances[i,j]` is the pairwise distances between query-i and gallery-j.
        query_ids:      vector [k1] where `query_ids[i]` is the ID corresponding to query-i.
        gallery_ids:    vector [k2] where `gallery_ids[j]` is the ID corresponding to gallery-j.
        rank:           in the ranking list wil be considered successfull the first `rank` positions. 
                        It must be satisfied `1 <= rank <= k2`

    Returns:

        the CMC rank-`rank` value [0->1].
    """
    assert distances.dim() == 2 and query_ids.dim() == 1 and gallery_ids.dim() == 1
    assert distances.shape[0] == query_ids.shape[0] and distances.shape[1] == gallery_ids.shape[0]
    assert rank > 0 and rank <= gallery_ids.shape[0]

    device = distances.device
    k1, k2  = distances.shape
    idxs = torch.argsort(distances, dim=-1)[:, :rank]       # (k1, rank)

    # let's retrieve gallery IDs corresponding to these indexes
    calc_ids = torch.zeros_like(idxs, dtype=torch.int).to(device)
    for i in range(k1):
        calc_ids[i] = gallery_ids.index_select(-1, idxs[i])

    query_ids = query_ids.unsqueeze(1).expand(-1, rank)     # (k1, rank)

    max_res, max_idx = (calc_ids == query_ids).max(1)
    well_calc = max_res.sum().item()
    return  well_calc / k1
