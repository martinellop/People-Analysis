from __future__ import print_function, absolute_import
import argparse

from reid.CelebREID_empty.reid.utils.serialization import load_checkpoint
from reid.CelebREID_empty.reid.core.train_1stream_capsule import *
from reid.CelebREID_empty.reid.models.dense1stream_capsule import DenseNet
from reid.CelebREID_empty.reid.data_loader_1stream import *
from reid.CelebREID_empty.reid.evaluator import *

import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import cv2

import sys

MODEL_PATH = r'C:\Users\Leonardo\Documents\Universita\CVCS\PRJ\reid\model.pth.tar'


DATASET_FOLDER = 'C:\\Users\\Leonardo\\Documents\\Universita\\CVCS\\PRJ\\reid\\CelebREID_empty\\datasetProva\\pedestrianUnito\\'
DATASET_FEATURES = r'C:\Users\Leonardo\Documents\Universita\CVCS\PRJ\reid\CelebREID_empty\dataset_features.npy'
DATASET_INDEX = r'C:\Users\Leonardo\Documents\Universita\CVCS\PRJ\reid\CelebREID_empty\dataset_order.txt'

TEST_FOLDER = 'C:\\Users\\Leonardo\\Documents\\Universita\\CVCS\\PRJ\\reid\\CelebREID_empty\\testImages\\'
def generate_image_feature(SR, imgfolder):
    # Data la imgfolder genere le feature per tutte le N immagini che ci sono dentro
    # e restituisce un array numpy Nx1024

    # ho copiato il codice come per la generazione del dataset, togliendo la parte di salvataggio

    # carico gli args, come per il dataset
    parser = argparse.ArgumentParser(description="ReIDCaps")
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--num_feature', type=int, default=1024)
    parser.add_argument('--num_iteration', type=int, default=4)
    parser.add_argument('--true_class', type=int, default=632)
    args, unknown = parser.parse_known_args()

    

    dataloader = get_loader(imgfolder, args.height, args.width, relabel=False,
                            batch_size=args.batch_size, mode='test', num_workers=args.workers)

    model = DenseNet(num_feature=args.num_feature, num_classes=args.true_class, num_iteration=args.num_iteration)

    checkpoint = load_checkpoint(MODEL_PATH)
    d = OrderedDict([('classifer1.classifier.weight', v) if k == 'classifer.classifier.weight' else (k, v) for k, v in checkpoint["state_dict"].items()])
    d = OrderedDict([('classifer1.classifier.bias', v) if k == 'classifer.classifier.bias' else (k, v) for k, v in d.items()])
    model.load_state_dict(d)
    model = nn.DataParallel(model)

    output = torch.Tensor()
    for i, (img, pids, name) in enumerate(dataloader):
        batchf = extract_cnn_feature(model, img, "pool5")
        output = torch.cat((output, batchf))

    npa = output.numpy()
    return npa


def prepare_research(dataset_features):
    # prende le feature e le memorizza, lo faccio un unica volta per tutte le immagini di test
    dataset = np.load(dataset_features)

    nb = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
    nb.fit(dataset)

    # Apro il file degli indici del dataset e leggo tutte le rige
    # associando ciascun id dell'immagine con il corrispettivo file
    dataset_files = {}
    with open(DATASET_INDEX, "r") as f:
        l = f.readlines()
        for i in l:
            if i.strip() == "":
                continue

            a = i.split()
            dataset_files[int(a[0])] = a[1]

    return nb, dataset_files


def search_image(nb, img_features):
    # Cerca nel dataset le immagini piu simili
    # e ritorna gli indici e le corrispettive distanze

    img_features = img_features.reshape(1,-1)

    distances, indices = nb.kneighbors(img_features)
    returnidx = indices[0]
    returndist = [round(i,4) for i in distances[0]]

    return returnidx, returndist

def compute_measure(indices, dataset_files):

    # Dati gli indici ritornati, usanto il file di indice del dataset
    # guarda quali immagini corrispondono alla persona giusta e quali no


    # Assumo che la prima immagine che viene ritornata, sia quella orginale
    # presa dal dataset, dalla quale è stata fatta quella upscalata.
    # Da quella guardo l'ID della persona per capire quali immagini sono corrette e quali no
    original_id = dataset_files[indices[0]][:3]

    # Calcolo Precision e Recall
    # la precision è il numero di immagini corrette ritornate sulle 10 ritornate
    # mentre la recall è il numero di immagini corrette ritornate su tutte quelle della stessa persona nel dataset
    # Ai nostri scopi è piu utile la recall
    num_results = 10
    correct_imgs = len([i for i in indices[1:] if dataset_files[i][:3] == original_id])
    total_correct = len([i for i in dataset_files.values() if i[:3] == original_id])
    precision = round(correct_imgs/num_results, 4)
    recall = round(correct_imgs/total_correct,4) if total_correct != 0 else 0

    print("10-Precision:", precision, "- Recall:",recall)

    return precision, recall


def view_results(indices, distances, dataset_files, precision, recall):

    original_id = dataset_files[indices[0]][:3]

    # Codice per stampare le immagini
    fig = plt.figure(figsize=(12, 9))
    rows = 3
    cols = 4

    for idx, dist, figidx in zip(indices, distances, (1,2,3,4,6,7,8,10,11,12)):
        fig.add_subplot(rows, cols, figidx)
        filename = dataset_files[idx]
        
        print(DATASET_FOLDER+filename)
        plt.imshow(cv2.imread(DATASET_FOLDER+filename)[...,::-1])
        plt.axis('off')
        find = "SAME" if filename[:3] == original_id else "NO"
        plt.title(str(dist) + " - " + find)
        

    plt.suptitle("Image: "+dataset_files[indices[0]]+" - Precision: "+str(precision)+" - Recall: "+str(recall))
    
    

    return fig

def get_filenames(indices, distances, dataset_files, precision, recall, op):
    original_id = dataset_files[indices[0]][:3]

    file_dict = {}
    file_dict["filenames"]={}
    file_dict["precision"] = precision
    file_dict["recall"] = recall
    file_dict["operation"] = op
    for idx, dist, figidx in zip(indices, distances, (1,2,3,4,6,7,8,10,11,12)):
        
        filename = dataset_files[idx]
        abs_filename = DATASET_FOLDER+filename
        find = "SAME" if filename[:3] == original_id else "NO"

        file_dict["filenames"][abs_filename] = find

    return file_dict 


def search(SR:bool, test_folder_param):

    
    # Genero in un'unica volta le feature per tutte le immagini di test
    imgs_features = generate_image_feature(SR,test_folder_param)




    # Nb è l'oggetto che mi permette di fare la ricerca
    nb, dataset_files = prepare_research(DATASET_FEATURES)

    fig_list = []
    mes_list = []
    # Analizzo poi ciascuna immagine alla volta
    for img in imgs_features:
        # Faccio la ricerca nel dataset ritornando gli indici delle immagini piu rilevanti e le loro distanze
        indices, distances = search_image(nb, img)
        # Calcolo precisione e recall, usando i file del dataset perche dal nome ricavo la persona nella foto
        precision, recall = compute_measure(indices, dataset_files)

        # E poi visualizzo i risulati
        #fig_list.append(view_results(indices, distances, dataset_files, precision, recall))
        tmp_dict = get_filenames(indices,distances,dataset_files, precision, recall, test_folder_param.split("\\")[-1])
        fig_list.append(tmp_dict)

    return fig_list

if __name__ == '__main__':
    search(False, TEST_FOLDER)