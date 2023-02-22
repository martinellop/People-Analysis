import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import scipy.stats

class HOG:
    def __init__(self, img:np.ndarray, nbins:int = 9, cell_w_h:int = 8, gray_image:bool = False):
        '''input img must be in 0-255 values'''
        assert img is not None
        self.original_image = img
        self.__nbins__ = nbins
        self.__cell_w_h__ = cell_w_h
        self.__only_gray__ = gray_image
        self.__cells_grid_columns__ = int(self.original_image.shape[1]/cell_w_h)
        self.__cells_grid_rows__ = int(self.original_image.shape[0]/cell_w_h)

        self.__target_image__ = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY) if self.__only_gray__ else self.original_image 

        self.__target_image__ = np.float32(self.__target_image__) / 255.0
        gx = cv2.Sobel(self.__target_image__, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(self.__target_image__, cv2.CV_32F, 0, 1, ksize=1)

        # Gradient magnitudes and gradient directions.
        self.gmag, self.gangles = cv2.cartToPolar(gx,gy, angleInDegrees=True)
        if self.gangles.ndim == 2:
            self.gangles = self.gangles[:,:,np.newaxis]
            self.gmag = self.gmag[:,:,np.newaxis]

        self.hogs = np.empty([self.__cells_grid_rows__, self.__cells_grid_columns__, nbins, self.gangles.shape[-1]], dtype=np.float32)
        for i in range(self.__cells_grid_rows__):
            for j in range(self.__cells_grid_columns__):
                start_x, start_y = i*cell_w_h, j*cell_w_h
                end_x, end_y = (i+1)*cell_w_h, (j+1)*cell_w_h
                self.hogs[i,j] = self.__ToBins__(self.gangles[start_x:end_x, start_y:end_y],self.gmag[start_x:end_x, start_y:end_y])

        assert np.all(self.hogs>=0), "Not all hog values are >= 0."
        print(self.hogs.shape)

        self.norm_hogs = np.empty([self.__cells_grid_rows__-1, self.__cells_grid_columns__-1, nbins, self.gangles.shape[-1]], dtype=np.float32)
        for i in range(self.__cells_grid_rows__-1):
            for j in range(self.__cells_grid_columns__-1):
                for k in range(self.hogs.shape[-1]):
                    self.norm_hogs[i,j,:,k] = Normalize(self.hogs[i:i+1,j:j+1,:,k])


    def __ToBins__(self, cell_dir:np.ndarray, cell_mag:np.ndarray):
        'Takes a 8x8 cell as input, and returns a np.array representing the HOG of that cell.'
        #direction angles in degrees!
        assert cell_dir.shape == cell_mag.shape, f"dir {cell_dir.shape} and mag {cell_mag.shape} matrixes have a different shape."
        #assert np.all(cell_mag <= 1), "all magnitudes should be <= 1."
        if cell_dir.ndim == 2:
            cell_dir = cell_dir[:,:,np.newaxis]
            cell_mag = cell_mag[:,:,np.newaxis]

        cell_dir = cell_dir % 180
        bins = np.zeros(shape=(self.__nbins__,cell_dir.shape[-1]), dtype=np.float32)
        step = 180.0/self.__nbins__
        
        #in this array, idx 0 correspond to angle 180-step, and idx -1 correspond to angle 0
        angles = np.arange(-step,180.0+step,step)
        angles = angles[:,np.newaxis]   #for broadcasting

        idx_helper = np.arange(cell_dir.shape[-1])
        #print(angles)
        for i in range(self.__cell_w_h__):
            for j in range(self.__cell_w_h__):
                dir, mag = cell_dir[i,j], cell_mag[i,j]
                diff = np.abs(angles-dir)
                idx_1 = np.argmin(diff, 0)
                tmp = diff[idx_1, idx_helper]
                diff[idx_1, idx_helper] = 180
                idx_2 = np.argmin(diff, 0)
                diff[idx_1, idx_helper] = tmp
                factor_1 = 1.0 - diff[idx_1, idx_helper] / step
                factor_2 = 1.0 - diff[idx_2, idx_helper] / step
                assert np.all(factor_1 >= 0), print(factor_1)
                assert np.all(factor_2 >= 0), print(diff,idx_1,idx_2,factor_2)
                #now we just have to correct the indexes.
                idx_1 = np.where(idx_1==0, self.__nbins__-1, idx_1-1)
                idx_1 = np.where(idx_1==self.__nbins__, 0, idx_1)
                idx_2 = np.where(idx_2==0, self.__nbins__-1, idx_2-1)
                idx_2 = np.where(idx_2==self.__nbins__, 0, idx_2)
                bins[idx_1] = bins[idx_1] + factor_1 * mag
                bins[idx_2] = bins[idx_2] + factor_2 * mag
        return bins


def Normalize(v:np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

    #visualization

def PlotAngles(angles:np.ndarray, mags:np.ndarray, thickness:int=1, plot_dim=35):
    assert plot_dim > 1, "plot_dimf must be greater than 1."
    assert thickness >= 1, "thickness must be greater then 0."
    assert angles.dtype is not np.float32 , f"dtype for tensor angles must be float32, but it's {angles.dtype}"
    assert mags.dtype is not np.float32 , f"dtype for tensor mags must be float32, but it's {mags.dtype}"
    assert mags.shape == angles.shape, f"tensor mags {mags.shape} has a different shape respect {angles.shape}"
    #assert np.all(mags >= 0), f"mag's value should be >= 0. You passed {mags}"
    margin = int(plot_dim/8)

    #print(f"mags__:{mags} std:{mags.std()}")

    plot_matrix = np.zeros(shape=(plot_dim,plot_dim), dtype=np.float32)
    angles = angles*math.pi / 180.0
    dir = np.array([ -np.cos(angles), np.sin(angles) ])
    #print(f"dirs: {dir}")
    cy = cx = round(plot_dim/2)


    mean = mags.mean()
    std = mags.std()
    max = mags.max()
    min = mags.min()
    #print(f"mean:{mean}; std:{std};")
    mags = (mags*mags-mean)*std*std
    #for i in range(mags.size):
    #  mags[i] = scipy.stats.norm(0.75, pow(std,2.0)).cdf(mags[i])
    mags[mags<0.01] = 0.0000001
    #mags = np.power(mags, 0.5)
    #print(f"++mags2: {mags}")

    w1 = math.ceil(thickness/2)
    w2 = thickness - w1 +1

    #let's iterate on the maximum number of pixels which could be written.
    for i in range( math.ceil(plot_dim/2) -math.floor(thickness/2) - margin):
        mov =  np.round(dir * i).astype(np.int16)
        dx_vec = mov[0,:]
        dy_vec = mov[1,:]

        for j in range(dx_vec.shape[0]): #probably this could be avoided and optimized..
            dx = dx_vec[j]
            dy = dy_vec[j]
            plot_matrix[cx+dx-w1:cx+dx+w2,cy+dy-w1:cy+dy+w2] += mags[j]
            plot_matrix[cx-dx-w1:cx-dx+w2,cy-dy-w1:cy-dy+w2] += mags[j]

    clamp_idx = plot_matrix>1
    plot_matrix[clamp_idx] = 1.0

    return plot_matrix


#visualize a full histogram
def DrawHistograms(histogram:np.ndarray, nbins:int, plot_dim:int, thick=1):
    step = 180.0/nbins
    angles = np.arange(0,180,step)
    img = np.empty(shape=(plot_dim,plot_dim,histogram.shape[-1]), dtype=np.float32)
    for i in range(histogram.shape[-1]): #color channels
        img[:,:,i] = PlotAngles(angles,histogram[:,i], thickness=thick, plot_dim=plot_dim)
    return img


def Get_HOG_Graphics(hogs:np.ndarray, nbins:int, thick=1, plot_dim=35):
    image = np.empty(shape=(hogs.shape[0]*plot_dim, hogs.shape[1]*plot_dim, hogs.shape[-1]), dtype=np.float32)

    for i in range(hogs.shape[0]):
        for j in range(hogs.shape[1]):
            image[i*plot_dim:(i+1)*plot_dim, j*plot_dim:(j+1)*plot_dim] = DrawHistograms(hogs[i,j], nbins, plot_dim, thick=thick)
    return np.uint8(image * 255)
