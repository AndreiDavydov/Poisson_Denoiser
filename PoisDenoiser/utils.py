
import torch as th
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

from torch.distributions import Poisson

filePath = '/home/andrey/Рабочий стол/Diploma/'

def psnr(im1, im2):
    '''
    Computes Peak Signal-to-Noise Ratio.
    
    Assumes that dimensions of inputs coincide.
    '''
    max_val = th.max(im1)
    err = th.mean((im1-im2)**2)
    return 10*th.log10(max_val**2 / err)


def get_poisson_pair(img, k, seed=1):
    '''
    Input image must be in (0,1] range.
    '''

    # th.manual_seed(1)
    
    if k == 0:
        return img    

    # np.random.seed(seed)             ###################################################
    # noisy = np.random.poisson(img/k) ###################################################

    img = th.FloatTensor(img)
    noisy = Poisson(img/k).sample()
    # noisy = th.FloatTensor(noisy)



    return img/k, noisy


def generate_test_pair(img_type='arange', size=256, k=0.01, seed=1):
    
    if img_type == 'file':

        img = imread(filePath+'DATASETS/BSDS500/BSDS500_256x256/val/ground_truth/78004.jpg').astype(np.float32)
        img = (img[...,0] + img[...,1] + img[...,2])/3

        img[img<1] = 1 # All values in pixels must be more than zero. 
                       # Instead there would be no stochastic process on these values.
        img = img/255
        img = img[:size, :size]

    elif img_type == 'arange':
        img = np.meshgrid(np.arange(1, size+1), np.arange(1, size+1))[0]/size

        img[size//5   :size*2//5, size//5   :size*2//5] = 1
        img[size*3//5 :size*4//5, size*3//5 :size*4//5] = 0

    img, noisy = get_poisson_pair(img=img, k=k, seed=seed)

    return img, noisy


def generate_batch(k=0.1, size=256, seed=1):

    clear1, noisy1 = generate_test_pair(img_type='arange', size=size, k=k, seed=seed)
    clear1.unsqueeze_(0).unsqueeze_(0)
    noisy1.unsqueeze_(0).unsqueeze_(0)

    clear2, noisy2 = generate_test_pair(img_type='file',   size=size, k=k, seed=seed)
    clear2.unsqueeze_(0).unsqueeze_(0)
    noisy2.unsqueeze_(0).unsqueeze_(0)   

    clear_batch = th.cat([clear1, clear2], dim=0)
    noisy_batch = th.cat([noisy1, noisy2], dim=0)

    return clear_batch, noisy_batch


def show_images(images, titles=None):
    '''
    All images are considered to be Tensor and have dimensions of CxHxW or HxW.
    If the number of images to show is more than one, then images is considered to be a list.
    The titles must be either string (1 image for input) or a list of strings.
    '''
    if not isinstance(images, list):
        images = [images]        

    images_corrected_dims = []
    for i, img in enumerate(images):
        if img.dim() == 4:
            img = img[0]

        img = img[0] if img.size()[0] == 1 else img.permute(1,2,0)
        images_corrected_dims.append(img)

    images = images_corrected_dims

    num_images = len(images)

    figsize = (8+num_images,8*num_images)
    fig, ax = plt.subplots(1, num_images, figsize=figsize)

    if num_images == 1:
        ax = [ax]

    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].set_axis_off()

    # Now only titles have to be distributed.
    if titles is not None:
        if isinstance(titles, str): # Title for one image
            ax[0].set_title(titles)

        elif isinstance(titles, list):
            for i, title in enumerate(titles):
                ax[i].set_title(title)

            if len(titles) < num_images:
                for i in range(len(titles), num_images):
                    ax[i].set_title('')