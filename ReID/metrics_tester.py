import torch
import torch.nn as nn

import argparse
import os
import json 

from common.dataset import get_dataloader
from common.testing import test, get_distances
from common.distances import L2_distance, Cosine_distance
from common.tools import ResultsDict

from deep.model import ReIDModel

from HOG.HogDescriber import HOGModel

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_deep_model', type=int, default=1)    # logically it's just a bool. when set to 0, will be used a non-deep model (HOG based)

    # dataset params
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)

    # deep model params
    parser.add_argument('--model', type=str, default="resnet18")    # choose your model here
    parser.add_argument('--use_bbneck', type=int, default=1)        # logically it's just a bool
    parser.add_argument('--num_classes', type=int, default=200)     # maximum number of identities to be classified
    parser.add_argument('--force_descr_dim', type=int, default=-1)  # if you want a specific dimension for descriptor feature vector, set it here
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_weights', type=str)

    # HOG reid params 
    parser.add_argument('--hog_nbins', type=int, default=9)
    parser.add_argument('--hog_pixels_per_cell', type=int, default=8)
    parser.add_argument('--hog_use_colors', type=int, default=0)    #logically it's just a bool

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--queries_batch', type=int, default=-1)    # how many queries to put together while processing distance matrix in testing?
                                                                    # -1 means 'processed all in one batch'

    parser.add_argument('--results_folder', type=str, default=os.path.join("deep","results"))
    parser.add_argument('--save_distances', type=int, default=1)    #logically is just a bool
    
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_options()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("You are running on", torch.cuda.get_device_name(), "gpu.")

    if args.use_deep_model:
        #deep model
        model = ReIDModel(args.model, args.num_classes, args.use_bbneck, args.force_descr_dim).to(device)
        weights = torch.load(args.model_weights)
        model.load_state_dict(weights)
    else:
        # HOG model
        model = HOGModel(args.hog_nbins, args.hog_pixels_per_cell, args.hog_use_colors).to(device)


    results_dir = args.results_folder

    dataset_paths = {}
    dataset_paths['motsynth'] = "D:\\Data\\University\\MOTSynth\\dataset_singleclip"
    dataset_paths['market1501'] = "D:\\Data\\University\\market-1501"
    dataset_paths['mars'] = "D:\\Data\\University\\MARS_dataset_test"

    #let's try all possible combinations.
    #possible_dists = ['cosine', 'euclidean']
    possible_dists = ['cosine']
    possible_datasets = ['motsynth', 'market1501', 'mars']

    for dist in possible_dists:
        if dist == 'cosine':
            dist_function = Cosine_distance
        elif dist == 'euclidean':
            dist_function = L2_distance
        else:
            raise Exception("You must select a valid distance function.")   

        for dataset in possible_datasets:
            results= ResultsDict()
            args.dataset_path = dataset_paths[dataset]
            trainloader, queryloader, galleryloader = get_dataloader(args=args, dataset=dataset)
            #test(model, queryloader, galleryloader,dist_function, device, results, args.queries_batch, verbose=True)
            #print(f"++++ Finished testing with {dist} dist and {dataset} dataset. Results: {results}")
            #with open(os.path.join(results_dir, f"final_metrics_{dist}_{dataset}.json"), "w") as outfile:
            #    json.dump(results, outfile)
            
            if args.save_distances:
                distances = ResultsDict()
                print("Starting to elaborate distances..")
                get_distances(model, queryloader, galleryloader,dist_function, device, distances, args.queries_batch)
                print(f"++++ Finished distances separation, with {dist} dist and {dataset} dataset.")
                with open(os.path.join(results_dir, f"distances_{dist}_{dataset}.json"), "w") as outfile:
                    json.dump(distances, outfile)