import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale


class HOGLayer(nn.Module):
    def __init__(self, nbins=10, pool=8, max_angle=math.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        mat = torch.FloatTensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        with torch.no_grad():
            gxy = F.conv2d(x, self.weight, None, self.stride,
                            self.padding, self.dilation, 1)
            #2. Mag/ Phase
            mag = gxy.norm(dim=1)
            norm = mag[:,None,:,:]
            phase = torch.atan2(gxy[:,0,:,:], gxy[:,1,:,:])

            #3. Binning Mag with linear interpolation
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:,None,:,:]

            n, c, h, w = gxy.shape
            out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
            out.scatter_(1, phase_int.floor().long()%self.nbins, norm)
            out.scatter_add_(1, phase_int.ceil().long()%self.nbins, 1 - norm)
            
            res = self.pooler(out)
            normed_res = res.reshape(n,self.nbins,-1)
            #normalization on the whole descriptor
            torch.nn.functional.normalize(normed_res,dim=-1,out=normed_res)
            return normed_res.reshape(n,-1)


class HOGModel(nn.Module):
    def __init__(self, nbins:int=9, pixels_per_cell=8, use_colors=False):
        super(HOGModel, self).__init__()
        self.model = HOGLayer(nbins, pixels_per_cell)
        self.use_colors = use_colors


    def forward(self, x:torch.Tensor):
        if not self.use_colors:
            x = rgb_to_grayscale(x).float()
            #print(x.shape)
            return self.model(x)
        else:
            #print(x[:,0].shape)
            r = self.model(x[:,0].unsqueeze(1).float())
            g = self.model(x[:,1].unsqueeze(1).float())
            b = self.model(x[:,2].unsqueeze(1).float())
            return torch.cat((r,g,b),1)
