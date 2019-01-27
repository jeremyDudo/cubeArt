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
        
        image = os.path.splitext(image)[0]

        with cd(imageFolder):
            np.save(image + '_{}'.format(final_size), edge)

    return edge     


#Note! Both image3D objects could be made more modular by allowing for more 
def image3D_fromFunc(imageP, ax=0):
    """
    Take nxn image, return nxnxn array of the image 
    Retains only the image's original pixels while extended in the 3rd axis

    Input: result from processedImage()
            ax over which to rotate the image before matrix is made

    Output: nxnxn array of image projected over 3rd axis
    """
    image3D = np.stack([imageP for _ in range( len(imageP[0]) )], axis=ax)

    return image3D
    

def image3D_fromName(image, final_size, ax=0):
    """
    Take nxn image, return nxnxn array of the image 
    Retains only the image's original pixels while extended in the 3rd axis

    Input: Take the image name (same input as the processedImage func)
            ax over which to rotate the image before matrix is made

    Output: nxnxn array of image projected over 3rd axis
    """
    with cd(imageFolder):
        image = image = os.path.splitext(image)[0]
        image = np.load(image + '_{}'.format(final_size))
    
    image3D = np.stack([image for _ in range( len(image[0]) )], axis=ax)

    return image3D
    

def overlay(image3Ds):
    """
    Takes a list of >2 image3D matrices of same size, performs
    logic to find the mutual "pixels" or "space" shared by the image3Ds


    Input: array of image3D objects with SAME SIZE (this could be made more modular)

    Output: array of positions of 'object' positions retained after carving out


    Note: image3D objects can be understood as a matrix of 1's and 0's
        1: there is 'object' here
        0: there is not 'object' here
    ** At the moment, I cannot verify if this exact mapping truly representative, but it gets the gist across **
    """

    exists = np.logical_and(image3Ds[0], image3Ds[1])
    
    for indx in range(len(image3Ds) - 1):
        exists = np.logical_and(exists, image3Ds[indx+1])
    
    exists = np.nonzero(exists)

    return exists


def graph(exists, save=True, show=False):

    x, y, z = exists[0], exists[1], exists[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x,y,z)

    plt.savefig()
