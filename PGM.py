from imageio import imread

from PoisDenoiser.nnLayers.functional import poisProx
from pydl.nnLayers.functional.functional import L2Prox

from PoisDenoiser.utils import psnr

import torch as th
from tqdm import tqdm

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

    img_pad_x = th.nn.functional.pad(img, pad=(0,1, 0,0), mode='replicate')
    img_pad_y = th.nn.functional.pad(img, pad=(0,0, 0,1), mode='replicate')
    
    delta_x = img_pad_x[:,:,:,1:] - img_pad_x[:,:,:,:-1]
    delta_y = img_pad_y[:,:,1:,:] - img_pad_y[:,:,:-1,:]
    
    return delta_x, delta_y


def reg_TV1_1(img):
    
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
    
    grad = - sign_y_pad[:,:,1:,:] \
           - sign_x_pad[:,:,:,1:] \
           + sign_y_pad[:,:,:-1,:] \
           + sign_x_pad[:,:,:,:-1]

    return grad 

def reg_TV1_2_grad(img):

    delta_x, delta_y = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate the gradient of TV regularizator...
  
    delta_x_pad = th.nn.functional.pad(delta_x, (1,0, 0,0), mode='constant')
    delta_y_pad = th.nn.functional.pad(delta_y, (0,0, 1,0), mode='constant')
    
    batch_size = img.size(0)

    grad = ( delta_y_pad[:,:,:-1,:] - delta_y_pad[:,:,1:,:] ) \
        / th.norm(delta_y.view(batch_size, -1), dim=1).view(batch_size, 1,1,1) + \
           ( delta_x_pad[:,:,:,:-1] - delta_x_pad[:,:,:,1:] ) \
        / th.norm(delta_x.view(batch_size, -1), dim=1).view(batch_size, 1,1,1) 
    return grad 


def get_lr(epoch, lr0=1, k=0.1):
    # Let's make it exponentially decreasing...
    arg = th.Tensor([-k*epoch])
    lr = th.Tensor([lr0]) * th.exp(arg)
    return lr


def PGM_step(img, noisy, epoch, reg, lr0_k=(1,0.01), prox_type='l2', stdn=None):
    
    lr0, k = lr0_k[0], lr0_k[1]
    lr = get_lr(epoch, lr0=lr0, k=k)

    img_new = img - lr*reg(img=img)

    lower_bound = 1e-8
    img_new = th.clamp(img_new, lower_bound, img_new.max())
    
    if prox_type =='pois':
        img_new = poisProx(img_new, noisy)
    elif prox_type == 'l2':
        if stdn is None:
            stdn = th.Tensor([0.6]).type_as(noisy)
        l2Prox = L2Prox.apply
        alpha = None
        img_new = l2Prox(img_new, noisy, alpha, stdn)
    elif prox_type == 'no_prox':
        return img_new

    return img_new


def do_denoise(noisy, reg, num_epochs=10, lr0_k=(1,0.1), \
    out_psnrs=False, ref_image=None, img_index=None, prox_type='l2', stdn=None):

    '''
    prox_type can be either 'l2', 'pois' or 'no_prox'.
    '''
    
    assert (prox_type == 'l2' or prox_type == 'pois' or prox_type == 'no_prox'),\
        'prox_type is not valid. It must be either "l2", "pois" or "no_prox".'

    img_estim = noisy.clone()
    if out_psnrs and ref_image is not None:
        if img_index is None:
            img_index = 0
        cur_psnr = psnr(ref_image[img_index], img_estim[img_index])
        psnrs = [cur_psnr]

    for epoch in tqdm(range(num_epochs)):
        img_estim = PGM_step(img_estim, noisy, epoch, \
            prox_type=prox_type, reg=reg, lr0_k=lr0_k, stdn=stdn)
        
        if out_psnrs and ref_image is not None:     
            cur_psnr = psnr(ref_image[img_index], img_estim[img_index])
            psnrs.append(cur_psnr)
        
            
    return (img_estim, psnrs) if out_psnrs else img_estim