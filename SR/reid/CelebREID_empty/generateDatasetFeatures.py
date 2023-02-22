from __future__ import print_function, absolute_import
import argparse

from torch.backends import cudnn
from reid.utils.serialization import load_checkpoint
from reid.core.train_1stream_capsule import *
from reid.models.dense1stream_capsule import DenseNet
from reid.data_loader_1stream import *
from reid.evaluator import *

import numpy as np


MODEL_PATH = r'C:\Users\Leonardo\Documents\Repo Github\People-Analyzer\SR\reid\model.pth.tar'
DATASET_FOLDER = r'C:\Users\Leonardo\Documents\Repo Github\People-Analyzer\SR\reid\CelebREID_empty\datasetProva\pedestrianUnito'
OUTPUT_FILE = "dataset_features.npy"
OUTPUT_INDEX = "dataset_order.txt"


def main(args):
    # serve per velocizzare un po l'esecuzione
    cudnn.benchmark = True

    # Carico il dataloader
    dataloader = get_loader(DATASET_FOLDER, args.height, args.width, relabel=False,
                                batch_size=args.batch_size, mode='test', num_workers=args.workers)

    # Carico il modello base
    # lo chiama DenseNet ma comprende tutta l'architettura con anche le reti di supporto per il train
    model = DenseNet(num_feature=args.num_feature, num_classes=args.true_class, num_iteration=args.num_iteration)

    # Carico i pesi per il modello
    checkpoint = load_checkpoint(MODEL_PATH)
    # cambio il nome di due chiavi a causa di una differenza di versioni, magari farlo meglio
    d = OrderedDict([('classifer1.classifier.weight', v) if k == 'classifer.classifier.weight' else (k, v) for k, v in checkpoint["state_dict"].items()])
    d = OrderedDict([('classifer1.classifier.bias', v) if k == 'classifer.classifier.bias' else (k, v) for k, v in d.items()])
    model.load_state_dict(d)
    model = nn.DataParallel(model)

    output = torch.Tensor()
    output_order = []

    print("Start extracting features...\n")

    for i, (img, _, name) in enumerate(dataloader):
        print("name: ",name)
        # Per ogni batch memorizza i nomi delle immagini
        output_order = output_order + name
        # Estrae le feature del batch
        batchf = extract_cnn_feature(model, img, "pool5")
        # Le aggiunge all'output
        output = torch.cat((output, batchf))
        print(output.shape)

    print("End")
    # Converte l'output in un tensore di numpy e lo salva
    npa = output.numpy()
    np.save(OUTPUT_FILE, npa)

    # E salva anche l'elenco delle immagini con i corrispettivi indici
    with open(OUTPUT_INDEX, "w") as f:
        for i, v in enumerate(output_order):
            r = str(i) + " " + v + "\n"
            f.write(r)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReIDCaps")

    # dataloader
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    # con 0 utilizza solo un worker, con piu mi dava errore
    parser.add_argument('-j', '--workers', type=int, default=0)
    # dimensioni di input nella rete, lasciare 224x224
    parser.add_argument('--height', type=int, default=224, help="input height, default: 256")
    parser.add_argument('--width', type=int, default=224, help="input width, default: 128")

    # Parametri del modello, usati per il training, ma non modificarlo
    # numero di feature in output a densenet, e quidi in output alle due reti di supporto
    parser.add_argument('--num_feature', type=int, default=1024)
    # iterazioni del RoutingByAgreement, usato nel training
    parser.add_argument('--num_iteration', type=int, default=4)
    # numero di classi finali, quindi il numero di persone nel dataset del training
    parser.add_argument('--true_class', type=int, default=632)


    main(parser.parse_args())
