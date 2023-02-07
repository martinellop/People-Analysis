import torch
import torch.nn as nn

import math

from .evaluation_metrics import calculate_CMC, calculate_mAP, calculate_mINP, pair_ids_with_distance_matrix
from .tools import ResultsDict


@torch.no_grad()
def extract_feature(model:nn.Module, dataloader, device):

    for batch_idx, (imgs, batch_pids) in enumerate(dataloader):

        imgs = imgs.to(device=device)
        batch_features = model(imgs).data

        batch_pids = torch.Tensor(batch_pids).to(device=device, dtype=torch.int)

        if batch_idx == 0:
            features = batch_features
            pids = batch_pids
        else:
            features = torch.cat((features, batch_features),0)
            pids = torch.cat((pids, batch_pids), 0)
                
    return features, pids



def test(model, queryloader, galleryloader, dist_function, device, results_history:ResultsDict, queries_batch:int=-1, verbose:bool=False):
    """
    Using given query and gallery datasets, let's try to perform a retrieval, looking for performance.

    Metrics:
        * CMC rank-n
        * mAP
        * mINP --> see [3]
    """
    model.eval()
    if verbose:
        print("Test: Extract Query")
    qf, q_pids = extract_feature(model, queryloader, device)
    if verbose:
        print("Test: Extract Gallery")
    gf, g_pids = extract_feature(model, galleryloader, device)


    # Since calculation of the whole distance matrix (with all queries) takes too much memory,
    # let's split it in batches of queries.
    n_queries = q_pids.shape[0]
    max_queries = queries_batch
    n_batches = math.ceil(n_queries / max_queries) if max_queries > 0 else 1

    average_map = 0
    average_mINP = 0
    average_rank_1 = 0
    average_rank_2 = 0
    average_rank_5 = 0
    average_rank_10 = 0
    average_rank_15 = 0
    average_rank_20 = 0

    for i in range(n_batches):
        start_idx = i*max_queries
        max_idx = n_queries if i+1 == n_batches else (i+1)*max_queries
        batched_qf = qf[start_idx:max_idx]
        batched_q_pids = q_pids[start_idx:max_idx]
        n_elements = batched_qf.shape[0]

        # Let's calculate distance matrix between queries and gallery items
        if verbose:
            print("start distance matrix calculation..")   
        distances = dist_function(batched_qf,gf)
        if verbose:
            print("distance matrix calculated.")

        calc_ids_matrix = pair_ids_with_distance_matrix(distances, g_pids)

        average_map += calculate_mAP(calc_ids_matrix, batched_q_pids) * n_elements
        average_mINP += calculate_mINP(calc_ids_matrix, batched_q_pids) * n_elements
        average_rank_1 += calculate_CMC(calc_ids_matrix, batched_q_pids, 1) * n_elements
        average_rank_2 += calculate_CMC(calc_ids_matrix, batched_q_pids, 2) * n_elements
        average_rank_5 += calculate_CMC(calc_ids_matrix, batched_q_pids, 5) * n_elements
        average_rank_10 += calculate_CMC(calc_ids_matrix, batched_q_pids, 10) * n_elements
        average_rank_15 += calculate_CMC(calc_ids_matrix, batched_q_pids, 15) * n_elements
        average_rank_20 += calculate_CMC(calc_ids_matrix, batched_q_pids, 20) * n_elements
        if verbose:
            print(f"Calculated metrics of query_batch {i} out of {n_batches}")
    
    results_history['mAP'].append(average_map/n_queries)
    results_history['mINP'].append(average_mINP/n_queries)
    results_history['rank-1'].append(average_rank_1/n_queries)
    results_history['rank-2'].append(average_rank_2/n_queries)
    results_history['rank-5'].append(average_rank_5/n_queries)
    results_history['rank-10'].append(average_rank_10/n_queries)
    results_history['rank-15'].append(average_rank_15/n_queries)
    results_history['rank-20'].append(average_rank_20/n_queries)
    return    