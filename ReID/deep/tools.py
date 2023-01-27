import os
import torch


def save_checkpoint(state, savepath='checkpoint.pth.tar'):
    savefolder = os.path.dirname(savepath)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    torch.save(state, savepath)

class ResultsDict(dict):
    def __setitem__(self, key, value):
        try:
            super().__setitem__(key, value) 
        except KeyError:
            self[key] = []
            super().__setitem__(key, value)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            self[key] = []
            return super().__getitem__(key)