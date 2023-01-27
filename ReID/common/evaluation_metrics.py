import torch


def calculate_ids(distances:torch.Tensor, gallery_ids:torch.Tensor, limit_to_first_elements:int = -1):
    
    idxs = torch.argsort(distances, dim=-1)
    if limit_to_first_elements > 0:
        idxs = idxs[:, :limit_to_first_elements]
    # let's retrieve gallery IDs corresponding to these indexes
    calc_ids = torch.zeros_like(idxs, dtype=torch.int, requires_grad=False).to(distances.device)
    for i in range(distances.shape[0]):
        calc_ids[i,:] = gallery_ids.index_select(-1, idxs[i,:])
    return calc_ids


def calculate_CMC(distances:torch.Tensor, query_ids:torch.Tensor, gallery_ids:torch.Tensor, rank:int=1):
    """
    Args:

        distances:      matrix [k1, k2], where `Distances[i,j]` is the pairwise distances between query-i and gallery-j.
        query_ids:      vector [k1] where `query_ids[i]` is the ID corresponding to query-i.
        gallery_ids:    vector [k2] where `gallery_ids[j]` is the ID corresponding to gallery-j.
        rank:           in the ranking list wil be considered successfull the first `rank` positions. 
                        It must be satisfied `1 <= rank <= k2`
    Note:

        In the gallery there must be at least one instance for each ID in the queries. If not, CMC will be calculated as 0.0 fot that query.

    Returns:

        the CMC rank-`rank` value [0->1].
    """
    assert distances is not None and query_ids is not None and gallery_ids is not None
    assert distances.dim() == 2 and query_ids.dim() == 1 and gallery_ids.dim() == 1
    assert distances.shape[0] == query_ids.shape[0] and distances.shape[1] == gallery_ids.shape[0]
    assert rank > 0 and rank <= gallery_ids.shape[0]

    k1, k2  = distances.shape
    calc_ids = calculate_ids(distances, gallery_ids, limit_to_first_elements=rank)

    query_ids = query_ids.unsqueeze(1).expand(-1, rank)     # (k1, rank)

    max_res, max_idx = (calc_ids == query_ids).max(1)
    well_calc = max_res.sum().item()
    return  well_calc / k1


def calculate_mAP(distances:torch.Tensor, query_ids:torch.Tensor, gallery_ids:torch.Tensor):
    """
    Args:

        distances:      matrix [k1, k2], where `Distances[i,j]` is the pairwise distances between query-i and gallery-j.
        query_ids:      vector [k1] where `query_ids[i]` is the ID corresponding to query-i.
        gallery_ids:    vector [k2] where `gallery_ids[j]` is the ID corresponding to gallery-j.

    Note:

        In the gallery there must be at least one instance for each ID in the queries. If not, an exception will be raised.

    Returns:

        the mAP value [0->1].
    """
    assert distances is not None and query_ids is not None and gallery_ids is not None
    assert distances.dim() == 2 and query_ids.dim() == 1 and gallery_ids.dim() == 1
    assert distances.shape[0] == query_ids.shape[0] and distances.shape[1] == gallery_ids.shape[0]

    device = distances.device
    k1, k2  = distances.shape

    calc_ids = calculate_ids(distances, gallery_ids)
    
    query_ids = query_ids.unsqueeze(1).expand(-1, k2)               # (k1, k2)
    correct_matches = (calc_ids == query_ids)

    well_classified = torch.zeros(size=(k1,), dtype=torch.int, requires_grad=False).to(device=device)
    aps = torch.zeros(size=(k1,), dtype=torch.float, requires_grad=False).to(device=device)
    for i in range(k2):
        well_classified = well_classified + correct_matches[:, i]
        aps = aps + ((correct_matches[:, i]) * well_classified / (i+1))

    try:    
        aps = aps / correct_matches.sum(-1)
    except ZeroDivisionError:
        print("Seems that at least one query was not in the gallery vector. Please fix.")
        return
    
    return aps.sum(-1).item() / k1


def calculate_mINP(distances:torch.Tensor, query_ids:torch.Tensor, gallery_ids:torch.Tensor):
    """
    mINP is a new metric expressed by: M. Ye, et al.,"Deep Learning for Person Re-Identification: A Survey and Outlook" 

    Args:

        distances:      matrix [k1, k2], where `Distances[i,j]` is the pairwise distances between query-i and gallery-j.
        query_ids:      vector [k1] where `query_ids[i]` is the ID corresponding to query-i.
        gallery_ids:    vector [k2] where `gallery_ids[j]` is the ID corresponding to gallery-j.

    Note:

        In the gallery there must be at least one instance for each ID in the queries. If not, an exception will be raised.

    Returns:

        the mINP value [0->1].
    """
    assert distances is not None and query_ids is not None and gallery_ids is not None
    assert distances.dim() == 2 and query_ids.dim() == 1 and gallery_ids.dim() == 1
    assert distances.shape[0] == query_ids.shape[0] and distances.shape[1] == gallery_ids.shape[0]

    device = distances.device
    k1, k2  = distances.shape
    
    calc_ids = calculate_ids(distances, gallery_ids)
    query_ids = query_ids.unsqueeze(1).expand(-1, k2)               # (k1, k2)
    correct_matches = (calc_ids == query_ids)

    hardest_match_idx = torch.zeros(size=(k1,), dtype=torch.int, requires_grad=False).to(device=device)
    #let's calculate the position of the hardest match for each query.
    for i in range(k2):
        hardest_match_idx = torch.where(correct_matches[:,i], i+1, hardest_match_idx)

    try:
        inp = correct_matches.sum(-1) / hardest_match_idx
    except ZeroDivisionError:
        print("Seems that at least one query was not in the gallery vector. Please fix.")
        return
    
    return inp.sum(0).item() / k1