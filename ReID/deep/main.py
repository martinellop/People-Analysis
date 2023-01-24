import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import os

from dataset import get_dataloader
from model import ReIDModel
from triplet_loss import TripletLoss
from tools import save_checkpoint

# FIXME: Problema del Mac, negli altri computer si puo togliere
#        e' per scaricare i pesi dei modelli
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def parse_options():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--gallery_path', type=str)

    # Model structure
    parser.add_argument('--model', type=str, default="resnet")  # resnet/alexnet
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--num_classes', type=int)  # num di persone nel dataset di train

    # Model training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--test_interval', type=int, default=1)  # ogni quante epoche fare il test

    # Resume
    parser.add_argument('--resume', type=int, default=0)  # e' un bool ma lo scrivo come int
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--savefolder', type=str, default=".")

    args, _ = parser.parse_known_args()
    return args


def main(args):

    trainloader, queryloader, galleryloader = get_dataloader(args)
    model = ReIDModel(args)
    triplet_loss = TripletLoss()  # TODO: settare magari meglio un margine
    ce_loss = nn.CrossEntropyLoss()

    # TODO: settare meglio il lr
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # lo scheduler ogni 30 epoche moltiplica il lr di 0.1
    # TODO: modificare valori
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    start_epoch = 0
    if bool(args.resume):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("Restarting from Epoch", start_epoch)

    model = nn.DataParallel(model)#.cuda() FIXME: riattivare

    print("Start Train")
    for epoch in range(start_epoch, args.max_epoch):
        print("Epoch", epoch)
        train(epoch, model, triplet_loss, ce_loss, optimizer, trainloader)

        if epoch % args.test_interval == 0 or epoch+1 == args.max_epoch:
            # Valuta il trainset
            print("Start test")
            rank1 = test(model, queryloader, galleryloader)
            print("Test Finished. Rank1 accuracy: ", rank1)

            # TODO: magari salvare tutti i valori per poi farci un grafico
            #       se no ci guardiamo noi quando inizia a scendere

            # Dopo averlo testato, salvo il modello con un checkpoint
            state_dict = model.module.state_dict()
            savepath = os.path.join(args.savefolder, "checkpoint_ep"+str(epoch)+".pth.tar")
            save_checkpoint({"state_dict": state_dict, "epoch": epoch}, savepath)

        scheduler.step()


def train(epoch_idx, model, triplet_loss, overall_loss, optimizer, trainloader):
    # Allena un epoca del modello

    model.train()
    len_trainloader = len(trainloader)

    for batch_idx, (imgs, pids) in enumerate(trainloader):
        #imgs, pids = imgs.cuda(), pids.cuda() FIXME: riattivare
        optimizer.zero_grad()
        # Faccio la forward
        feature_vector, outputs = model(imgs)
        # Calcolo le loss
        tr_loss = triplet_loss(feature_vector, pids)
        ce_loss = overall_loss(outputs, pids)
        # TODO: Inserire un termine di discount tra le due loss?
        loss = tr_loss + ce_loss
        # Aggiornamento valori
        loss.backward()
        optimizer.step()

        # Calcolo per vedere quelli corretti
        pred_class = torch.argmax(outputs, dim=1)
        correct = torch.sum(pred_class == pids)

        print(f"Epoch: {epoch_idx} Batch: {batch_idx}/{len_trainloader}, Correct: {correct}")

        # FIXME: rimuovere, solo per test
        if batch_idx == 20:
            return


def test(model, queryloader, galleryloader):
    # Usa il dataset Query e Gallery per fare una prova di image retrieval e vedere le performance
    model.eval()
    print("Test: Extract Query")
    qf, q_pids = extract_feature(model, queryloader)
    print("Test: Extract Gallery")
    gf, g_pids = extract_feature(model, galleryloader)

    # Calcolo la matrice di distanza tra ogni elemento della query e la galleria
    m, n = qf.size(0), gf.size(0)
    # TODO: utilizzo di default la distanza euclidea, vedi se ci sono distanze migliori (tipo coseno)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    for i in range(m):
        # non ho capito bene come funziona ma moltiplica tra di loro le matrici dei parametri
        # e poi la somma con la matrice che l'ha chiamata
        # FIXME: mi dice che è deprecata, magari vedere come risolvere
        distmat[i:i + 1].addmm_(1, -2, qf[i:i + 1], gf.t())
    print("Test: Distance Matrix computed")

    # TODO: trovare la metrica migliore da usare per calcolare il test
    # in altri usavano la mAP e CMC (che non so cosa sia)

    # come esempio calcolo la rank1 accuracy
    index = torch.argsort(distmat, dim=1)
    # in questa matrice la prima dim è il numero di query,
    # e per ciascuna si ha l'elenco ordinato delle immagini della gallery piu vicine

    correct = 0
    for i in range(m):
        # Per ogni immagine della query, guardo l'immagine piu vicina e confronto se la persona è uguale
        first_img_index = index[i][0]
        if q_pids[i] == g_pids[first_img_index]:
            correct += 1

    rank1 = correct/m
    return rank1


@torch.no_grad()
def extract_feature(model, dataloader):
    # data una immagine estra le feature escludendo la parte di classificazione
    features, pids = [], []
    for batch_idx, (imgs, batch_pids) in enumerate(dataloader):
        # TODO: c'e' questo codice per flippare l'immagine e calcolare il fv anche su quello e poi sommarlo
        #          non so se abbia troppo senso, nel dubbio lo lascio commentato
        '''          
        flip_imgs = torch.flip(imgs, (3,))  # flippa le immagini
        flip_imgs = flip_imgs.cuda()
        batch_features_flip = model(flip_imgs, True).data.cpu()
        batch_features += batch_features_flip
        '''

        # imgs = imgs.cuda() FIXME: riattivare

        batch_features = model(imgs, True).data.cpu()  # metto il True cosi da far restituire solo il feature_vector
        features.append(batch_features)
        pids += batch_pids

    features = torch.cat(features, 0)  # "estraggo" da ciascun batch e unisco in un'unico tensore
    return features, pids


if __name__ == '__main__':
    config = parse_options()
    main(config)
