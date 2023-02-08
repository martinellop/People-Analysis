import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from common.PersonDescriber import PersonDescriber
from skimage.feature import hog


class HogHyperParams():
    def __init__(self, nbins:int=9, pixels_per_cell:int=8, use_colors:bool=False):
        self.nbins= nbins
        self.pixels_per_cell = pixels_per_cell
        self.use_colors = use_colors


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


class HogDescriber_scikit(PersonDescriber):
    def __init__(self, hyperprms : HogHyperParams, device):
        self._nbins_= hyperprms.nbins
        self._pixels_per_cell_ = hyperprms.pixels_per_cell
        self._cells_per_block_ = hyperprms.cells_per_block

    def Extract_Description(self, x:torch.tensor):
        #unfortunately, this function doesn't work with batches and it uses np
        data = x.numpy()
        res = torch.zeros((0,), dtype=torch.float32)
        for i in range(data.shape[0]):
            crop = data[i]
            #print(crop.shape)
            descr = hog(crop, orientations=self._nbins_, pixels_per_cell=(self._pixels_per_cell_, self._pixels_per_cell_),
                    cells_per_block=(self._cells_per_block_, self._cells_per_block_), channel_axis=0)
            descr_tensor = torch.from_numpy(descr).reshape(1,-1)
            if i == 0:
                res = res.reshape(0, descr_tensor.shape[1])
            res = torch.cat((res,descr_tensor))
        return res






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


# TO BE UPDATED
# much faster then scikit version
class HogDescriber_torch(PersonDescriber):
    def __init__(self, hyperprms : HogHyperParams, device:torch.device):
        self._cells_per_block_ = hyperprms.cells_per_block
        self._device_ = device
        self.model = HOGLayer(nbins=hyperprms.nbins, pool=hyperprms.pixels_per_cell).to(device=device)

    def Extract_Description(self, x:torch.Tensor, use_colors:bool=False):
        if not use_colors:
            x = rgb_to_grayscale(x)
            #print(x.shape)
            x = x.to(self._device_).float()
            return self.model(x)
        else:
            #print(x[:,0].shape)
            r = self.model(x[:,0].to(self._device_).unsqueeze(1).float())
            g = self.model(x[:,1].to(self._device_).unsqueeze(1).float())
            b = self.model(x[:,2].to(self._device_).unsqueeze(1).float())
            return torch.cat((r,g,b),1)

