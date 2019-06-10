from imageio import imread
# import numpy as np

# from PoisDenoiser.projection_batch import Projection as proj, is_in_C as cond
from PoisDenoiser.projection_utils import Projection as proj, is_in_C as cond

from PoisDenoiser.utils import psnr

import torch as th

from tqdm import tqdm


'''
All functions mimic a functional of "PGM.py" module 
but do it in a Pytorch nn mode, when the input size of Tensors 
is B x C x H x W


All computations are made in Double-precision. Further tests will show whether one needs such accuracy or not.
'''

'''
This module contains all necessary blocks for PGM algorithm, 
including TV regularizations with gradients.
'''

def get_deltas(img):
    '''
    Returns discrete derivatives of the image "img" by columns and rows, respectively.
    If img.shape == (B,C,H,W) then returned delta_x.shape == delta_y.shape == (B,C,H,W).
    For delta_x the last column will always be 0, as delta_y's last row.
    
    '''
    assert img.dim() == 4 # unsqueeze the dimension if necessary.

    img = img#.double()#################################################################

    img_pad_x = th.nn.functional.pad(img, pad=(0,1, 0,0), mode='replicate')
    img_pad_y = th.nn.functional.pad(img, pad=(0,0, 0,1), mode='replicate')
    
    delta_x = img_pad_x[:,:,:,1:] - img_pad_x[:,:,:,:-1]
    delta_y = img_pad_y[:,:,1:,:] - img_pad_y[:,:,:-1,:]
    
    return delta_x, delta_y


def reg_TV1_1(img=None, deltas=None):
    
    deltas = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate TV regularizator...

    batch_size = img.size(0)
    return th.sum( th.abs(deltas[0].view(batch_size, -1)) \
                 + th.abs(deltas[1].view(batch_size, -1)) )


def reg_TV1_2(img):
    
    deltas = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate TV regularizator...
    
    assert (img.dim() == 4), \
    'Input size must be like BxCxHxW.'

    batch_size = img.size(0)
    return th.sum( th.norm(deltas[0].view(batch_size, -1), dim=1) \
                 + th.norm(deltas[1].view(batch_size, -1), dim=1) )


def reg_TV1_1_grad(img):

    deltas = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate the gradient of TV regularizator...
    
    sign_x = th.sign(deltas[0])
    sign_y = th.sign(deltas[1])
    
    sign_x_pad = th.nn.functional.pad(sign_x, (1,0, 0,0), mode='constant')
    sign_y_pad = th.nn.functional.pad(sign_y, (0,0, 1,0), mode='constant')    
    
    grad = - sign_y_pad[:,:,1:,:] - sign_x_pad[:,:,:,1:] + sign_y_pad[:,:,:-1,:] + sign_x_pad[:,:,:,:-1]
    return grad 

def reg_TV1_2_grad(img):

    delta_x, delta_y = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate the gradient of TV regularizator...
  
    delta_x_pad = th.nn.functional.pad(delta_x, (1,0, 0,0), mode='constant')
    delta_y_pad = th.nn.functional.pad(delta_y, (0,0, 1,0), mode='constant')
    
    batch_size = img.size(0)

    grad = ( delta_y_pad[:,:,:-1,:] - delta_y_pad[:,:,1:,:] ) / th.norm(delta_y.view(batch_size, -1), dim=1).view(batch_size, 1,1,1) + \
           ( delta_x_pad[:,:,:,:-1] - delta_x_pad[:,:,:,1:] ) / th.norm(delta_x.view(batch_size, -1), dim=1).view(batch_size, 1,1,1) 
    return grad 


def get_lr(epoch, lr0=1, k=0.1):
    # Let's make it exponentially decreasing...
    arg = th.Tensor([-k*epoch])#.double()        ##################################################
    lr = th.Tensor([lr0])   * th.exp(arg) #.double() ##################################################
    return lr


def PGM_step(img, noisy, epoch, reg, lr0_k=(1,0.01), \
    fixed=True, num_iters=None):
    
    lr0, k = lr0_k[0], lr0_k[1]
    lr = get_lr(epoch, lr0=lr0, k=k)

    img_new = img - lr*reg(img=img)

    lower_bound = 1e-8
    img_new = th.clamp(img_new, lower_bound, img_new.max())
    
    if fixed:
        # img_new = proj(img_new, noisy, fixed=fixed, num_iters=num_iters) # projection on Poissonian set
        img_new, root = proj(img_new, noisy, num_iters=num_iters)
    else:
        img_new = proj(img_new, noisy, eps=1e-2, fixed=fixed)
    return img_new


from tqdm import tqdm

def do_denoise(noisy_img, reg, num_epochs=10, lr0_k=(1,0.1), \
    fixed=True, num_iters=2, out_psnrs=False, ref_image=None, img_index=None):
        
    img_estim = noisy_img.clone()#.double() ########################################################
    if out_psnrs and ref_image is not None:
        if img_index is None:
            img_index = 0
        psnrs = [psnr(ref_image[img_index]    , img_estim[img_index])] #.double()

    for epoch in tqdm(range(num_epochs)):
        if fixed:
            img_estim = PGM_step(img_estim, noisy_img, epoch, reg=reg, \
                lr0_k=lr0_k, fixed=fixed, num_iters=num_iters)
        else:
            img_estim = PGM_step(img_estim, noisy_img, epoch, reg=reg, \
                lr0_k=lr0_k, fixed=fixed)
        
        if out_psnrs and ref_image is not None:     
            psnrs.append(psnr(ref_image[img_index]    , img_estim[img_index]))    #.double()      
        
            
    return (img_estim, psnrs) if out_psnrs else img_estim