from imageio import imread
# import numpy as np

from PoisDenoiser.projection import Projection as proj, is_in_C as cond
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

    img = img.type(th.FloatTensor)

    img_pad_x = th.nn.functional.pad(img, pad=(0,1, 0,0), mode='replicate')
    img_pad_y = th.nn.functional.pad(img, pad=(0,0, 0,1), mode='replicate')
    
    delta_x = img_pad_x[:,:,:,1:] - img_pad_x[:,:,:,:-1]
    delta_y = img_pad_y[:,:,1:,:] - img_pad_y[:,:,:-1,:]
    
    return delta_x, delta_y


def reg_TV1_1(img=None, deltas=None):
    if deltas is None and img is None:
        raise Exception('Check inputs. Some input must be given!')
    
    if img is not None:
        deltas = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate TV regularizator...
    
    return th.sum( th.abs(deltas[0]) + th.abs(deltas[1]) )


def reg_TV1_2(img=None, deltas=None):
    if deltas is None and img is None:
        raise Exception('Check inputs. Some input must be given!')
    
    if img is not None:
        deltas = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate TV regularizator...
    
    return th.sum( th.norm(deltas[0]) + th.norm(deltas[1]) )


def reg_TV1_1_grad(img=None, deltas=None):
    if deltas is None and img is None:
        raise Exception('Check inputs. Some input must be given!')
        
    if not (img is None):
        deltas = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate the gradient of TV regularizator...
    
    sign_x = th.sign(deltas[0])
    sign_y = th.sign(deltas[1])
    
    sign_x_pad = th.nn.functional.pad(sign_x, (1,0, 0,0), mode='constant')
    sign_y_pad = th.nn.functional.pad(sign_y, (0,0, 1,0), mode='constant')    
    
    grad = - sign_y_pad[:,:,1:,:] - sign_x_pad[:,:,:,1:] + sign_y_pad[:,:,:-1,:] + sign_x_pad[:,:,:,:-1]
    return grad 

def reg_TV1_2_grad(img=None, deltas=None):
    if deltas is None and img is None:
        raise Exception('Check inputs. Some input must be given!')
        
    if not (img is None):
        delta_x, delta_y = get_deltas(img)
        
    # Now "deltas" are defined. 
    # Let's calculate the gradient of TV regularizator...
  
    delta_x_pad = th.nn.functional.pad(delta_x, (1,0, 0,0), mode='constant')
    delta_y_pad = th.nn.functional.pad(delta_y, (0,0, 1,0), mode='constant')
    
    grad = ( delta_y_pad[:,:,:-1,:] - delta_y_pad[:,:,1:,:] ) / th.norm(delta_y) + \
           ( delta_x_pad[:,:,:,:-1] - delta_x_pad[:,:,:,1:] ) / th.norm(delta_x)
    return grad 


def get_lr(epoch, lr0=1, k=0.1):
    # Let's make it exponentially decreasing...
    arg = th.FloatTensor([-k*epoch])
    lr = th.FloatTensor([lr0]) * th.exp(arg)
    return lr

def PGM_step(img, noisy, epoch, reg, lr0_k=(1,0.01), \
    fixed=True, num_iters=None):
    
    lr0, k = lr0_k[0], lr0_k[1]
    
    lr = get_lr(epoch, lr0=lr0, k=k)
    img_new = img - lr*reg(img=img)

    lower_bound = 1e-8
    img_new = th.clamp(img_new, lower_bound, img_new.max())
    
    
    if fixed:
        img_new = proj(img_new, noisy, fixed=fixed, num_iters=num_iters) # projection on Poissonian set
    else:
        img_new = proj(img_new, noisy, eps=1e-2, fixed=fixed)
        
    return img_new


from tqdm import tqdm

def do_denoise(noisy_img, reg, num_epochs=100, lr0_k=(1,0.1), \
    fixed=True, num_iters=10, out_psnrs=False, ref_image=None):
        
    img_estim = noisy_img.clone()
    if out_psnrs and ref_image is not None:
        psnrs = [psnr(ref_image, img_estim)] 

    for epoch in tqdm(range(num_epochs)):
        if fixed:
            img_estim = PGM_step(img_estim, noisy_img, epoch, reg=reg, \
                lr0_k=lr0_k, fixed=fixed, num_iters=num_iters)
        else:
            img_estim = PGM_step(img_estim, noisy_img, epoch, reg=reg, \
                lr0_k=lr0_k, fixed=fixed)
        
        if out_psnrs and ref_image is not None:     
            psnrs.append(psnr(ref_image, img_estim))          
        
            
    return (img_estim, psnrs) if out_psnrs else img_estim