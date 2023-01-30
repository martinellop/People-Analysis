import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import os
import json 
import time
import math

from deep.dataset import get_dataloader
from deep.model import ReIDModel
from deep.custom_losses import TripletLoss, CenterLoss
from deep.tools import save_checkpoint, ResultsDict

from common.distances import L2_distance, Cosine_distance
from common.evaluation_metrics import calculate_CMC, calculate_mAP, calculate_mINP, pair_ids_with_distance_matrix


"""
Main references:
[1]:
    H. Luo et al., "A Strong Baseline and Batch Normalization Neck for Deep Person Re-Identification"
    in IEEE Transactions on Multimedia, vol. 22, no. 10, pp. 2597-2609,
    Oct. 2020, doi: 10.1109/TMM.2019.2958756.

[2]:
    Z. Zhong, L. Zheng, D. Cao and S. Li, "Re-ranking Person Re-identification with k-Reciprocal Encoding"
    2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA,
    2017, pp. 3652-3661, doi: 10.1109/CVPR.2017.389.

[3]:
    M. Ye, et al.,"Deep Learning for Person Re-Identification: A Survey and Outlook" 
    in IEEE Transactions on Pattern Analysis & Machine Intelligence, vol. 44, no. 06, pp. 2872-2893, 
    2022. doi: 10.1109/TPAMI.2021.3054775        
"""



def parse_options():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--gallery_path', type=str)

    # Model structure
    parser.add_argument('--model', type=str, default="resnet18")    # choose your model here
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--use_bbneck', type=int, default=1)        # logically it's just a bool
    parser.add_argument('--use_center_loss', type=int, default=1)   # logically it's just a bool
    parser.add_argument('--num_classes', type=int, default=200)     # maximum number of identities to be classified

    # Model training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--distance_function', type=str, default='cosine')# distance function used in testing.
    parser.add_argument('--queries_batch', type=int, default=-1)    # how many queries to put together while processing distance matrix in testing?
                                                                    # -1 means 'processed all in one batch'

    parser.add_argument('--test_interval', type=int, default=1)     # how many epochs to be process before a test?
    parser.add_argument('--triplet_margin', type=float, default=0.3)# the margin used by the triplet loss function. default value taken from [1]

    # Checkpoints
    parser.add_argument('--resume_checkpoint', type=str, default="")
    parser.add_argument('--checkpoint_every', type=int, default=5)  # how many epoch to process before saving a checkpoint?

    parser.add_argument('--checkpoints_folder', type=str, default=os.path.join("deep","checkpoints"))
    parser.add_argument('--results_folder', type=str, default=os.path.join("deep","results"))
    

    args, _ = parser.parse_known_args()
    return args




def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("You are running on", torch.cuda.get_device_name(), "gpu.")

    trainloader, queryloader, galleryloader = get_dataloader(args)
    model = ReIDModel(args).to(device)
    
    triplet_loss = TripletLoss(args.triplet_margin).to(device=device)
    if args.use_center_loss:
        center_loss = CenterLoss().to(device=device)
    else:
        center_loss = None
    id_loss = nn.CrossEntropyLoss().to(device=device)

    queries_batch = args.queries_batch
    
    if args.distance_function == 'cosine':
        dist_function = Cosine_distance
    elif args.distance_function == 'euclidean':
        dist_function = L2_distance
    else:
        raise Exception("You must select a valid distance function.")

    # Variable  learning rate:
    optimizer = optim.Adam(model.parameters(), lr=0.0035)                       # starting value chose accordingly to [1]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)   # every 30 epochs, lr is multiplied by 0.1

    # here will be appended loss and metrics values after every epoch
    # --> it will be also saved from checkpoints
    results_history = ResultsDict()   

    start_epoch = 0
    if args.resume_checkpoint != "":
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        results_history = checkpoint["results_history"]
        print("Restarting from Epoch", start_epoch)

    print_metrics_while_training = True
    print("Start Train")
    for epoch in range(start_epoch, args.max_epoch):
        print("++ Starting epoch", epoch)
        t1 = time.time()
        train(epoch, model, triplet_loss, id_loss, center_loss, optimizer, trainloader, device, results_history, verbose=False)

        if epoch % args.test_interval == 0 or epoch+1 == args.max_epoch:
            # Valuta il trainset
            print("Start test")
            test(model, queryloader, galleryloader,dist_function, device, results_history, queries_batch)

            test_finished_string = "Test Finished. "

            if print_metrics_while_training:
                for k in results_history.keys():
                    test_finished_string += k
                    value = results_history[k][-1]
                    test_finished_string += f": {value:.4f}; "
            print(test_finished_string)

            if (epoch+1) % args.checkpoint_every:

                # Let's have a checkpoint, saving model status
                state_dict = model.state_dict()
                savepath = os.path.join(args.checkpoints_folder, "checkpoint_ep"+str(epoch)+".pth.tar")
                #savepath = os.path.join(args.checkpoints_folder, "last_checkpoint.pth.tar")
                data = {"state_dict": state_dict, "epoch": epoch, "results_history":results_history}
                save_checkpoint(data, savepath)
                print(f"Saved checkpoint at {savepath}")

        scheduler.step()
        print(f"++ Finished epoch {epoch} in {(time.time() - t1):.1f} seconds.")
    
    print("+++++ Finished training +++++")


    results_dir = args.results_folder
    #saving model
    torch.save(model.state_dict(), os.path.join(results_dir, "model.bin"))
    #let's save to file metrics history
    with open(os.path.join(results_dir, "results_history.json"), "w") as outfile:
        json.dump(results_history, outfile)


def train(epoch_idx, model, triplet_loss_function, id_loss_function, center_loss_function, optimizer, trainloader, device, results_history:ResultsDict, verbose:bool=True):
    
    # Train one epoch
    model.train()
    len_trainloader = len(trainloader)
    epoch_triplet_loss = 0
    epoch_id_loss = 0
    epoch_center_loss = 0
    epoch_total_loss = 0

    use_center_loss = (center_loss_function is not None)
    center_loss_multiplier = 0.005      # the value pointed out by [1] as best compromise.

    for batch_idx, (imgs, pids) in enumerate(trainloader):
        imgs, pids = imgs.to(device=device), pids.to(device=device)

        optimizer.zero_grad()
        features_vector, class_results  = model(imgs)

        # loss calculation
        tr_loss = triplet_loss_function(features_vector, pids)
        id_loss = id_loss_function(class_results, pids)
           
        
        # loss merge 
        loss = tr_loss + id_loss                                # maybe we could add a discount value between loss values

        # eventual center loss integration
        if use_center_loss:
            center_loss = center_loss_multiplier * center_loss_function(features_vector, pids)
            loss = loss + center_loss
            epoch_center_loss += center_loss.item()

        # saving loss data for final stats
        epoch_triplet_loss += tr_loss.item()
        epoch_id_loss += id_loss.item()
        epoch_total_loss += loss.item()

        # update model
        loss.backward()
        optimizer.step()

        if verbose:
            print(f"Epoch: {epoch_idx} Batch: {batch_idx}/{len_trainloader}, Loss: {loss.item():.5f}")   
    
    
    results_history["total-loss"].append(epoch_total_loss/len_trainloader)
    results_history["triplet-loss"].append(epoch_triplet_loss/len_trainloader)
    results_history["id-loss"].append(epoch_id_loss/len_trainloader)
    if use_center_loss:
        results_history["center-loss"].append(epoch_center_loss/len_trainloader)
    

def test(model, queryloader, galleryloader, dist_function, device, results_history:ResultsDict, queries_batch:int=-1):
    """
    Using given query and gallery datasets, let's try to perform a retrieval, looking for performance.

    Metrics:
        * CMC rank-n
        * mAP
        * mINP --> see [3]
    """
    model.eval()
    #print("Test: Extract Query")
    qf, q_pids = extract_feature(model, queryloader, device)
    #print("Test: Extract Gallery")
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
        distances = dist_function(batched_qf,gf)

        calc_ids_matrix = pair_ids_with_distance_matrix(distances, g_pids)

        average_map += calculate_mAP(calc_ids_matrix, batched_q_pids) * n_elements
        average_mINP += calculate_mINP(calc_ids_matrix, batched_q_pids) * n_elements
        average_rank_1 += calculate_CMC(calc_ids_matrix, batched_q_pids, 1) * n_elements
        average_rank_2 += calculate_CMC(calc_ids_matrix, batched_q_pids, 2) * n_elements
        average_rank_5 += calculate_CMC(calc_ids_matrix, batched_q_pids, 5) * n_elements
        average_rank_10 += calculate_CMC(calc_ids_matrix, batched_q_pids, 10) * n_elements
        average_rank_15 += calculate_CMC(calc_ids_matrix, batched_q_pids, 15) * n_elements
        average_rank_20 += calculate_CMC(calc_ids_matrix, batched_q_pids, 20) * n_elements
    
    results_history['mAP'].append(average_map/n_queries)
    results_history['mINP'].append(average_mINP/n_queries)
    results_history['rank-1'].append(average_rank_1/n_queries)
    results_history['rank-2'].append(average_rank_2/n_queries)
    results_history['rank-5'].append(average_rank_5/n_queries)
    results_history['rank-10'].append(average_rank_10/n_queries)
    results_history['rank-15'].append(average_rank_15/n_queries)
    results_history['rank-20'].append(average_rank_20/n_queries)
    return


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


if __name__ == '__main__':
    config = parse_options()
    main(config)


"""
Possible improvements:
    + modify network last stage from stride 2 to stride 1  (Last stride)--> [1]

    + study if better cosine or euclidean distance in inference time. According to [1], it should be better cosine.
"""