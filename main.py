import os
import numpy as np 
from PIL import Image
import pylab
from skimage.filters import roberts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d as conv2 
from stl import mesh
from skimage import color, data, restoration, img_as_float
from skimage.segmentation import chan_vese
from mpl_toolkits import mplot3d


# File names [will be generated if saving]
imageFolder = "Modified Images"


# TO ALLOW EASY ACCESS TO SUBFOLDERS
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def processedImage(image, final_size, show=False, save=True):
    """
    Converts image into silhouette with proper size for final cube

    Input: image name or location of image [String contatining a .jpg/.png/etc]

    Output: silhouette [2D array]
    """

    # load image, resize, convert to array, and make grayscale
    img0 = color.rgb2gray(np.asarray(Image.open(image).resize((final_size, final_size))))



    # [using method from skimage restoration]
    # sharpen image after rescaling
    psf = np.ones((5,5))/25

    img0 = conv2(img0, psf, 'same')

    img_noisy = img0.copy()
    img_noisy += (np.random.poisson(lam=25, size=img0.shape)-10)/255.

    deconvolved_RL = restoration.richardson_lucy(img_noisy, psf, iterations=30)


    # [combining skimage functions]
    # use chan_vese to separate into distinct shades
    float_deconv = img_as_float(deconvolved_RL)

    cv = chan_vese(float_deconv, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
                    dt=0.5, init_level_set="checkerboard", extended_output=True)

    # make ~silhouette by taking the hard edge between contrasting colors
    edge = roberts(cv[0]) 

    # debugging/visual confirmation [toggle-able]
    if show:
        plt.plot()
        plt.gray()
        plt.imshow(edge)
        plt.show()        

    if save:
        if not os.path.exists(imageFolder):
            os.makedirs(imageFolder)
        
        with cd(imageFolder):
            np.save(image + '{}'.format(final_size), edge)

    return edge       