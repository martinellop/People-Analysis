import os
import torch


def save_checkpoint(state, savepath='checkpoint.pth.tar'):
    savefolder = os.path.dirname(savepath)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    torch.save(state, savepath)