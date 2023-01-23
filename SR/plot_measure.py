import imagenhancer as ie
import numpy as np
import torch
import matplotlib.pyplot as plt 
import cv2 as cv





def transform_images(base_filename:str, enhancer):
    # Return a list of a image and all the restored one starting from the base ones
    # with all the same shape

    '''
    0: base
    1: deg + resize
    2: deg + pyrUp
    3: upscale
    4: upscale + filter -> works very bad  , remove? -> its because of the GAN's artifacts
    5: filter + upscale
    6: pyrup + sharpening
    '''

    base_img_bgr = cv.imread(base_filename)
    base_img = cv.cvtColor(base_img_bgr,cv.COLOR_RGB2BGR)
    from UnpairedSR.codes.config.PDMSR import inference as pdm

    img = pdm.inf(base_img)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    deg_img = ie.image_degradation(base_img)
    h,w,c = base_img.shape
    res_img = cv.resize(deg_img, dsize=(w,h))
    pyr_img = cv.pyrUp(deg_img)
    upscale_img = enhancer.upsample(deg_img)
    second_model = cv.pyrDown(cv.pyrDown(img))
    up_filt_img = enhancer.filtering(enhancer.upsample(deg_img))
    filt_up_img = enhancer.upsample(np.uint8(enhancer.filtering(deg_img).clip(0,255)))
    filter_pyr_img = enhancer.filtering(pyr_img)

    
    image_list = list()
    image_list.append(base_img.astype(np.uint8))
    image_list.append(res_img.astype(np.uint8))
    image_list.append(pyr_img.astype(np.uint8))
    image_list.append(upscale_img.astype(np.uint8))
    image_list.append(second_model.astype(np.uint8))
    image_list.append(up_filt_img.astype(np.uint8))
    image_list.append(filt_up_img.astype(np.uint8))
    image_list.append(filter_pyr_img.astype(np.uint8))

    min_w = base_img.shape[1]
    min_h = base_img.shape[0]
    for i in image_list:
        h,w,c = i.shape
        if h< min_h: min_h = h
        if w < min_w : min_w = w
    
    for i in range(len(image_list)):
        image_list[i] = image_list[i][:min_h,:min_w,:]


    return image_list


def plotter(image_list:list, measure:list, labels:list):
    # Plot a list of images with the respective measure relative to 
    fig = plt.figure(figsize=(16, 12), dpi=200)
    k = len(image_list)
    ax=list()
    for i,img in enumerate(image_list):
        ax.append(fig.add_subplot(1,k,i+1))
        ax[i].set_title(str(measure[i]))
        ax[i].set_xlabel(labels[i])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        
        
        plt.imshow(img)



def elab(base_img_name):
    model_path_x2 = '.\\RealESRGANmaster\\weights\\RealESRGAN_x2plus.pth'

    enhancer = ie.RealESRGANx2(model_path = model_path_x2,scale_= 2)

    image_list = transform_images(base_img_name, enhancer)

    n_trans = 8

    h,w,c = image_list[0].shape

    w_ext=w*n_trans

    full_img = np.empty((h,w_ext,c))

    for i in range(n_trans):
        full_img[:,i*w:(i+1)*w,:] = image_list[i]

    full_img = full_img.astype(np.uint8)
    #plt.imshow(full_img)


    image_list_t = list()
    for i in range(n_trans):
        image_list_t.append(torch.Tensor(image_list[i]))

    measure = list()
    measure.append('0')
    for i in range(n_trans-1):
        measure.append(f'{(ie.PSNR(image_list_t[0],image_list_t[i+1])):.3f}')

    labels = ["base", "deg+\nres", "deg+\npyrup","upsc","sec\nmod","up+\nsharp","sharp+\nup","pyr+\nsharp"]
    plotter(image_list, measure,labels)


if __name__ == '__main__':
    elab("069/cropped_rgb151.jpg")