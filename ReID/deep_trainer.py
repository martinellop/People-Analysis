import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import os
from matplotlib import pyplot as plt
import json 

from deep.dataset import get_dataloader
from deep.model import ReIDModel
from deep.triplet_loss import TripletLoss
from deep.tools import save_checkpoint

from common.distances import L2_distance
from common.evaluation_metrics import calculate_CMC, calculate_mAP, calculate_mINP


"""
Main references:
[1]:
    H. Luo et al., "A Strong Baseline and Batch Normalization Neck for Deep Person Re-Identification,"
    in IEEE Transactions on Multimedia, vol. 22, no. 10, pp. 2597-2609,
    Oct. 2020, doi: 10.1109/TMM.2019.2958756.

[2]:
    Z. Zhong, L. Zheng, D. Cao and S. Li, "Re-ranking Person Re-identification with k-Reciprocal Encoding,"
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
    parser.add_argument('--num_classes', type=int)                  # maximum number of identities to be classified

    # Model training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--test_interval', type=int, default=1)     # how many epochs to be process before a test?
    parser.add_argument('--triplet_margin', type=float, default=0.3)# the margin used by the triplet loss function. default value taken from [1]

    # Resume
    parser.add_argument('--resume', type=int, default=0)            # logically it's just a bool
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--savefolder', type=str, default=os.path.join("deep","checkpoints"))

    args, _ = parser.parse_known_args()
    return args


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("You are running on", torch.cuda.get_device_name(), "gpu.")

    trainloader, queryloader, galleryloader = get_dataloader(args)
    model = ReIDModel(args)
    
    triplet_loss = TripletLoss(args.triplet_margin).to(device=device)
    id_loss = nn.CrossEntropyLoss().to(device=device)

    # Variable  learning rate:
    optimizer = optim.Adam(model.parameters(), lr=0.0035)                       # starting value chose accordingly to [1]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)   # every 30 epochs, lr is multiplied by 0.1

    start_epoch = 0
    if bool(args.resume):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("Restarting from Epoch", start_epoch)

    model = nn.DataParallel(model).to(device=device)

    # here will be appended loss values after every epoch (triplet_loss, id_loss, total_loss)
    # --> it will be also saved from checkpoints
    model.loss_history = torch.zeros(size=(3,0), device=device, dtype=torch.float)

    # here will be appended metrics values after every epoch (which one is defined by the test() function)
    # --> it will be also saved from checkpoints
    model.test_metrics_history = {}   

    print_metrics_while_training = False  #set to false to improve perfomance.

    print("Start Train")
    for epoch in range(start_epoch, args.max_epoch):
        print("Epoch", epoch)
        train(epoch, model, triplet_loss, id_loss, optimizer, trainloader, device)

        if epoch % args.test_interval == 0 or epoch+1 == args.max_epoch:
            # Valuta il trainset
            print("Start test")
            test(model, queryloader, galleryloader, device, first_test=(epoch==0))

            test_finished_string = "Test Finished. "

            if print_metrics_while_training:
                for k in model.test_metrics_history.keys():
                    test_finished_string += k
                    value = model.test_metrics_history[k][-1]
                    test_finished_string += f": {value:.3f}; "
            print(test_finished_string)

            # Let's also have a checkpoint, saving model status
            state_dict = model.module.state_dict()
            #savepath = os.path.join(args.savefolder, "checkpoint_ep"+str(epoch)+".pth.tar")
            savepath = os.path.join(args.savefolder, "last_checkpoint.pth.tar")
            save_checkpoint({"state_dict": state_dict, "epoch": epoch}, savepath)

        scheduler.step()
    
    #let's save to file loss history
    torch.save(model.loss_history.detach().cpu() ,os.path.join("deep","results", "losses.pth"))

    #let's save to file metrics history
    with open(os.path.join("deep","results", "metrics.json"), "w") as outfile:
        json.dump(model.test_metrics_history, outfile)


def train(epoch_idx, model, triplet_loss_function, id_loss_function, optimizer, trainloader, device, verbose:bool=True):
    
    # Train one epoch
    model.train()
    len_trainloader = len(trainloader)
    epoch_triplet_loss = 0
    epoch_id_loss = 0
    epoch_total_loss = 0

    for batch_idx, (imgs, pids) in enumerate(trainloader):
        imgs, pids = imgs.to(device=device), pids.to(device=device)

        optimizer.zero_grad()
        features_vector, class_results  = model(imgs)

        # loss calculation
        tr_loss = triplet_loss_function(features_vector, pids)
        id_loss = id_loss_function(class_results, pids)
        # loss merge 
        loss = tr_loss + id_loss                                # maybe we could add a discount value between loss values

        # saving loss data for final stats
        epoch_triplet_loss += tr_loss.item()
        epoch_id_loss += id_loss.item()
        epoch_total_loss += loss.item()

        # update model
        loss.backward()
        optimizer.step()

        if verbose:
            print(f"Epoch: {epoch_idx} Batch: {batch_idx}/{len_trainloader}, Loss: {loss.item():.4f} ({tr_loss.item():.4f} + {id_loss.item():.4f})")


    losses = torch.Tensor((epoch_triplet_loss, epoch_id_loss, epoch_total_loss)).to(device=device) / len_trainloader
    model.loss_history = torch.cat((model.loss_history, losses.unsqueeze(1)),dim=1)


def test(model, queryloader, galleryloader, device, first_test:bool=False):
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


    # Let's calculate distance matrix between queries and gallery items    
    distances = L2_distance(qf,gf)

    if first_test:
        model.test_metrics_history['mAP'] = []
        model.test_metrics_history['mINP'] = []
        model.test_metrics_history['rank-1'] = []
        model.test_metrics_history['rank-2'] = []
        model.test_metrics_history['rank-5'] = []
        model.test_metrics_history['rank-10'] = []
        model.test_metrics_history['rank-15'] = []
        model.test_metrics_history['rank-20'] = []
        
    model.test_metrics_history['mAP'].append(calculate_mAP(distances, q_pids, g_pids))
    model.test_metrics_history['mINP'].append(calculate_mINP(distances, q_pids, g_pids))
    model.test_metrics_history['rank-1'].append(calculate_CMC(distances, q_pids, g_pids, 1))
    model.test_metrics_history['rank-2'].append(calculate_CMC(distances, q_pids, g_pids, 2))
    model.test_metrics_history['rank-5'].append(calculate_CMC(distances, q_pids, g_pids, 5))
    model.test_metrics_history['rank-10'].append(calculate_CMC(distances, q_pids, g_pids, 10))
    model.test_metrics_history['rank-15'].append(calculate_CMC(distances, q_pids, g_pids, 15))
    model.test_metrics_history['rank-20'].append(calculate_CMC(distances, q_pids, g_pids, 20))
    
    return


@torch.no_grad()
def extract_feature(model:nn.Module, dataloader, device):
    # data una immagine estra le feature escludendo la parte di classificazione
    pids = []
    for batch_idx, (imgs, batch_pids) in enumerate(dataloader):
        # TODO: c'e' questo codice per flippare l'immagine e calcolare il fv anche su quello e poi sommarlo
        #          non so se abbia troppo senso, nel dubbio lo lascio commentato
        '''          
        flip_imgs = torch.flip(imgs, (3,))  # flippa le immagini
        flip_imgs = flip_imgs.cuda()
        batch_features_flip = model(flip_imgs, True).data.cpu()
        batch_features += batch_features_flip
        '''

        imgs = imgs.to(device=device)
        batch_features = model(imgs).data

        batch_pids = torch.Tensor(batch_pids).to(device=device, dtype=torch.int)

        if batch_idx == 0:
            features = batch_features
            pids = batch_pids
        else:
            features = torch.cat((features, batch_features),0)
            pids = torch.cat((pids, batch_pids.flatten()), 0)
                
    return features, pids


if __name__ == '__main__':
    config = parse_options()
    main(config)


"""
Possible improvements:
    + modify network last stage from stride 2 to stride 1  (Last stride)--> [1]
    + add Center Loss to current loss function --> [1]

    + evaluation between euclidean and cosine distances in triplet loss calculation. Accordingly to [1], euclidean should be better tho.
"""