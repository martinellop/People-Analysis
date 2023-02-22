import shutil
import os
import random

# Creato per parsare i dati del dataset PRCC
# Da ciascuna persona del train, con una certa probabilità, prende alcune immagini,
# gli cambia di nome in modo da metterle come piace al modello (3num_3num.jpg) e li copia
# nella cartella

source_dir = os.getcwd() + '\\reid\\CelebREID_empty\\prcc.tar\\prcc\\prcc\\rgb\\train\\'
target_dir = os.getcwd() + '\\reid\\CelebREID_empty\\datasetProva\\pedestrianUnito\\'

people = os.listdir(source_dir)
for p in people:
    print(p)
    if p.startswith("."):
        continue
    imgs = os.listdir(source_dir+p)
    for i in imgs:
        r = random.randint(0,100)
        # con una probabilità del 5% copia il file
        if r<20:
            # gli cambia il nome in unendo il nome della cartella e gli ultimi 3 numeri del nome.
            filename = p+"_"+i[-7:]

            shutil.copy(source_dir+p+"/"+i, target_dir+filename)