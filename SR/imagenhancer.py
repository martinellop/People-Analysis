import cv2
import torch
import numpy as np
from RealESRGANmaster.realesrgan import utils
from basicsr.archs.rrdbnet_arch import RRDBNet
import math
import matplotlib.pyplot as plt

# PSNR is an approximation to human perception of reconstruction quality.
def PSNR(input, target):
    mse = torch.mean((input - target) ** 2)
    return 20 * math.log10(255 / math.sqrt(mse))

def MSE(input, target):
    return torch.mean((input - target) ** 2)[0]


def gaussian_blur(img):
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img,-1,kernel)

def noise_addiction(img):
    noise = np.random.normal(loc=0., scale=1., size=img.shape)
    res= img + noise

    res = res.astype(np.uint8)
    return res

def downsampling(img,scale):
    H,W,C = img.shape
    H /= scale
    W /= scale
    dim = (int(W), int(H))
    print(f"W: {dim[0]}, H: {dim[1]}")
    res = cv2.resize(img,dim,interpolation=cv2.INTER_NEAREST)
    return res

def jpeg_compression(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./.tmp.jpeg",tmp,[int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return cv2.cvtColor(cv2.imread("./.tmp.jpeg"),cv2.COLOR_BGR2RGB)

def image_degradation(img,scale = 2):

    res=img
    # Gaussian Blur
    res = gaussian_blur(res)
    # Noise addiction
    res = noise_addiction(res)
    # Downsampling
    res = downsampling(res, scale)
    #Jpeg Compression
    res = jpeg_compression(res)
    return res

def sharpening(img):
    sharpening_kernel = np.zeros((3,3)) 
    sharpening_kernel[1,1]=2
    sharpening_kernel = sharpening_kernel - np.ones((3,3))/9
    return cv2.filter2D(img,ddepth=3,kernel=sharpening_kernel)
    



def plotter(image_list:list, measure:list):
    fig = plt.figure()
    k = len(image_list)
    ax=list()
    for i,img in enumerate(image_list):
        ax.append(fig.add_subplot(1,k,i+1))
        ax[i].set_title(str(measure[i]))
        plt.axis('off')
        plt.imshow(img)

class RealESRGANx2:

   

    def __init__(self, model_path, scale_, output=(224,224)):
        model2 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale_)
        netscale2 = 2

        loadnet2 = torch.load(model_path, map_location=torch.device('cpu'))
        keyname = "params_ema"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

        model.load_state_dict(loadnet2[keyname], strict=True)
        model2.eval()
        model2 = model2.to("cuda")

        self._upsamplerx2 = utils.RealESRGANer(
            scale=netscale2, # Netscale che diciamo noi
            model_path=model_path,  # Pesi che diciamo noi
            dni_weight=None, # ? ? ?, None suppongo a meno che non usiamo reales-general-x4v3
            model=model2,    # Modello
            tile=0, # Tile size, 0 for testing
            tile_pad=10, # default 10 , tile padding
            pre_pad=0,   # def 0, prepadding size
            half=True, # Precision 16 or 32 fp, default fp16
            gpu_id=0)
        
        self._output_dim = output
        self._treshold = (output[0]+output[1])/4

    def upsample(self, img):
        res, _ = self._upsamplerx2.enhance(img)
        return res

    def filtering(self,img):
        '''
        gaussian -> blurring
        laplacian
        sharpening
        median
        bilateral                   -> remove texture, no good
        deblurring/denoising w/ cnn
        wiener filter/deconv
        '''
        img = sharpening(img)
        return img
    
    def enhance(self,img):
        h,w,c = img.shape
        
        if h <= self._treshold or w <= self._treshold:
            return self.filtering(self.upsample(img))
        else:
            return self.filtering(img)
    

