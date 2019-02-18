#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 08:20:50 2018

@author: stamatis
@email : s.lefkimmiatis@skoltech.ru
"""
#import math
import numpy as np
import torch as th
from torch.autograd import Variable
from pydl.nnLayers.functional import functional

def symmetricPad2D(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    symmetricPad2DF = functional.SymmetricPad2D.apply
    
    x = th.randn(4,3,40,40).type(dtype)
    pad = tuple(np.random.randint(0,20,(4,)))
    
    if GPU and th.cuda.is_available():
        x = x.cuda()
    
    sz_x = x.size()
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone() 
    cost = lambda input: cost_symmetricPad2D(input,pad)
    
    
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(sz_x)

    x_var = Variable(x,requires_grad = True)
    
    y = symmetricPad2DF(x_var,pad)
    grad_output = th.ones_like(y)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))

    return err_x, x_var.grad, x_numgrad

def symmetricPad_transpose2D(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    symmetricPad_transpose2DF = functional.SymmetricPad_transpose2D.apply
    
    x = th.randn(4,3,20,20).type(dtype)
    crop = tuple(np.random.randint(0,10,(4,)))
    x = functional.SymmetricPad2D.apply(x,crop)
    
    if GPU and th.cuda.is_available():
        x = x.cuda()
    
    sz_x = x.size()
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone() 
    cost = lambda input: cost_symmetricPad_transpose2D(input,crop)
    
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(sz_x)

    x_var = Variable(x,requires_grad = True)
    
    y = symmetricPad_transpose2DF(x_var,crop)
    grad_output = th.ones_like(y)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))

    return err_x, x_var.grad, x_numgrad

def l2Proj(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    
    l2ProjF = functional.L2Proj.apply
    
    x = th.randn(4,3,40,40).type(dtype)
    x -= x.view(x.size(0),-1).min().view(-1,1,1,1)
    x /= x.view(x.size(0),-1).max().view(-1,1,1,1) 
    x = x*255
    alpha  = th.Tensor(np.random.randint(0,3,(1,))).type(dtype)
    stdn = th.Tensor(np.random.randint(5,20,(4,1))).type(dtype)
    
    if GPU and th.cuda.is_available():
        x = x.cuda()
        alpha = alpha.cuda()
        stdn = stdn.cuda()
    
    sz_x = x.size()
    grad_output = th.randn_like(x)
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone() 
    cost = lambda input: cost_l2Proj(input,alpha,stdn,grad_output)
        
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(sz_x)
    
    sz_alpha = alpha.size()
    alpha_numgrad = th.zeros_like(alpha).view(-1)
    perturb = alpha_numgrad.clone()
    cost = lambda input : cost_l2Proj(x,input,stdn,grad_output)
    
    for k in range(0,alpha.numel()):
        perturb[k]  = epsilon
        loss1 = cost(alpha.view(-1).add(perturb).view(sz_alpha))
        loss2 = cost(alpha.view(-1).add(-perturb).view(sz_alpha))
        alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0    
    
    alpha_numgrad = alpha_numgrad.view(sz_alpha)
    
    x_var = Variable(x,requires_grad = True)
    alpha_var = Variable(alpha,requires_grad = True)
    
    y = l2ProjF(x_var,alpha_var,stdn)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))
            
    err_a = th.norm(alpha_var.grad.data.view(-1) - alpha_numgrad.view(-1))/\
            th.norm(alpha_var.grad.data.view(-1) + alpha_numgrad.view(-1))            
    
    
    return err_x, x_var.grad.data, x_numgrad, err_a, alpha_var.grad.data, alpha_numgrad


def SVl2Proj(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    
    SVl2ProjF = functional.SVL2Proj.apply
    
    x = th.randn(2,3,30,30).type(dtype)
    x -= x.view(x.size(0),-1).min().view(-1,1,1,1)
    x /= x.view(x.size(0),-1).max().view(-1,1,1,1) 
    x = x*255
    alpha  = th.Tensor(np.random.randint(0,3,(1,))).type(dtype)
    stdn = th.Tensor(np.random.randint(5,20,(x.numel(),1))).type(dtype)
    stdn = stdn.view_as(x).contiguous()
    
    if GPU and th.cuda.is_available():
        x = x.cuda()
        alpha = alpha.cuda()
        stdn = stdn.cuda()
    
    sz_x = x.size()
    grad_output = th.randn_like(x)
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone() 
    cost = lambda input: cost_SVl2Proj(input,alpha,stdn,grad_output)
        
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(sz_x)
    
    sz_alpha = alpha.size()
    alpha_numgrad = th.zeros_like(alpha).view(-1)
    perturb = alpha_numgrad.clone()
    cost = lambda input : cost_SVl2Proj(x,input,stdn,grad_output)
    
    for k in range(0,alpha.numel()):
        perturb[k]  = epsilon
        loss1 = cost(alpha.view(-1).add(perturb).view(sz_alpha))
        loss2 = cost(alpha.view(-1).add(-perturb).view(sz_alpha))
        alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0    
    
    alpha_numgrad = alpha_numgrad.view(sz_alpha)
    
    x_var = Variable(x,requires_grad = True)
    alpha_var = Variable(alpha,requires_grad = True)
    
    y = SVl2ProjF(x_var,alpha_var,stdn)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))
            
    err_a = th.norm(alpha_var.grad.data.view(-1) - alpha_numgrad.view(-1))/\
            th.norm(alpha_var.grad.data.view(-1) + alpha_numgrad.view(-1))            
    
    
    return err_x, x_var.grad.data, x_numgrad, err_a, alpha_var.grad.data, alpha_numgrad


def l2Prox(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    
    l2ProxF = functional.L2Prox.apply
    
    x = th.randn(4,3,40,40).type(dtype)
    x -= x.view(x.size(0),-1).min().view(-1,1,1,1)
    x /= x.view(x.size(0),-1).max().view(-1,1,1,1) 
    x = x*255
    z = th.randn(4,3,40,40).type(dtype)
    z -= z.view(z.size(0),-1).min().view(-1,1,1,1)
    z /= z.view(z.size(0),-1).max().view(-1,1,1,1) 
    z = z*255    
    alpha  = th.Tensor(np.random.randint(0,3,(1,))).type(dtype)
    stdn = th.Tensor(np.random.randint(5,20,(4,1))).type(dtype)
    
    if GPU and th.cuda.is_available():
        x = x.cuda()
        z = z.cuda()
        alpha = alpha.cuda()
        stdn = stdn.cuda()
    
    sz_x = x.size()
    grad_output = th.randn_like(x)
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone() 
    cost = lambda input: cost_l2Prox(input,z,alpha,stdn,grad_output)
        
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(sz_x)
    
    sz_alpha = alpha.size()
    alpha_numgrad = th.zeros_like(alpha).view(-1)
    perturb = alpha_numgrad.clone()
    cost = lambda input : cost_l2Prox(x,z,input,stdn,grad_output)
    
    for k in range(0,alpha.numel()):
        perturb[k]  = epsilon
        loss1 = cost(alpha.view(-1).add(perturb).view(sz_alpha))
        loss2 = cost(alpha.view(-1).add(-perturb).view(sz_alpha))
        alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0    
    
    alpha_numgrad = alpha_numgrad.view(sz_alpha)
    
    x_var = Variable(x,requires_grad = True)
    alpha_var = Variable(alpha,requires_grad = True)
    
    y = l2ProxF(x_var,z,alpha_var,stdn)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))
            
    err_a = th.norm(alpha_var.grad.data.view(-1) - alpha_numgrad.view(-1))/\
            th.norm(alpha_var.grad.data.view(-1) + alpha_numgrad.view(-1))            
    
    
    return err_x, x_var.grad.data, x_numgrad, err_a, alpha_var.grad.data, alpha_numgrad


def SVl2Prox(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    
    SVl2ProxF = functional.SVL2Prox.apply
    
    x = th.randn(2,3,30,30).type(dtype)
    x -= x.view(x.size(0),-1).min().view(-1,1,1,1)
    x /= x.view(x.size(0),-1).max().view(-1,1,1,1) 
    x = x*255
    z = th.randn_like(x)
    z -= z.view(z.size(0),-1).min().view(-1,1,1,1)
    z /= z.view(z.size(0),-1).max().view(-1,1,1,1) 
    z = z*255    
    alpha  = th.Tensor(np.random.randint(0,3,(1,))).type(dtype)
    stdn = th.Tensor(np.random.randint(5,20,(x.numel(),1))).type(dtype)
    stdn = stdn.view_as(x).contiguous()
    
    if GPU and th.cuda.is_available():
        x = x.cuda()
        z = z.cuda()
        alpha = alpha.cuda()
        stdn = stdn.cuda()
    
    sz_x = x.size()
    grad_output = th.randn_like(x)
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone() 
    cost = lambda input: cost_SVl2Prox(input,z,alpha,stdn,grad_output)
        
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0
        print("{}\n".format(k))

    x_numgrad = x_numgrad.view(sz_x)
    
    sz_alpha = alpha.size()
    alpha_numgrad = th.zeros_like(alpha).view(-1)
    perturb = alpha_numgrad.clone()
    cost = lambda input : cost_SVl2Prox(x,z,input,stdn,grad_output)
    
    for k in range(0,alpha.numel()):
        perturb[k]  = epsilon
        loss1 = cost(alpha.view(-1).add(perturb).view(sz_alpha))
        loss2 = cost(alpha.view(-1).add(-perturb).view(sz_alpha))
        alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0            
    
    alpha_numgrad = alpha_numgrad.view(sz_alpha)
    
    x_var = Variable(x,requires_grad = True)
    alpha_var = Variable(alpha,requires_grad = True)
    
    y = SVl2ProxF(x_var,z,alpha_var,stdn)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))
            
    err_a = th.norm(alpha_var.grad.data.view(-1) - alpha_numgrad.view(-1))/\
            th.norm(alpha_var.grad.data.view(-1) + alpha_numgrad.view(-1))            
    
    
    return err_x, x_var.grad.data, x_numgrad, err_a, alpha_var.grad.data, alpha_numgrad

def grbf_lut(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    
    grbf_lutF = functional.Grbf_lut.apply
    
    x = th.randn(2,3,20,20).type(dtype)
    x = x - x.min()
    x = x/x.max()
    x = (x*208)-104
    origin,step,sigma,centers = -104,0.1,4,th.range(-100,100,4).type_as(x)
    weights = th.randn(x.size(1),centers.numel()).type_as(x)
    
    if GPU and th.cuda.is_available():
        x = x.cuda()
        weights = weights.cuda()
    
    sz_x = x.size()
    grad_output = th.randn_like(x)
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone() 
    cost = lambda input: cost_grbf_lut(input,weights,centers,sigma,origin,step,grad_output)
        
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(sz_x)
    
    sz_weights = weights.size()
    weights_numgrad = th.zeros_like(weights).view(-1)
    perturb = weights_numgrad.clone()
    cost = lambda input :  cost_grbf_lut(x,input,centers,sigma,origin,step,grad_output)
    
    for k in range(0,weights.numel()):
        perturb[k]  = epsilon
        loss1 = cost(weights.view(-1).add(perturb).view(sz_weights))
        loss2 = cost(weights.view(-1).add(-perturb).view(sz_weights))
        weights_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0    
    
    weights_numgrad = weights_numgrad.view(sz_weights)
    
    x_var = Variable(x,requires_grad = True)
    weights_var = Variable(weights,requires_grad = True)
    
    y = grbf_lutF(x_var,weights_var,centers,sigma,origin,step)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))
            
    err_w = th.norm(weights_var.grad.data.view(-1) - weights_numgrad.view(-1))/\
            th.norm(weights_var.grad.data.view(-1) + weights_numgrad.view(-1))            
    
    
    return err_x, x_var.grad.data, x_numgrad, err_w, weights_var.grad.data, weights_numgrad

def weightNormalization(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False,\
                        normalizedWeights=False,zeroMeanWeights=False):
    
    weightNormalizationF = functional.WeightNormalization.apply
    
    x = th.randn(4,3,40,40).type(dtype)*100+10
    alpha  = th.randn(4,1).type(dtype)   
    
    if GPU and th.cuda.is_available():
        x = x.cuda()
        alpha = alpha.cuda()
            
    sz_x = x.size()
    grad_output = th.randn_like(x)
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone()
    cost = lambda input: cost_weightNormalization(input,alpha,normalizedWeights,zeroMeanWeights,grad_output)
    
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0
        
    x_numgrad = x_numgrad.view(sz_x)
        
    sz_alpha = alpha.size()
    alpha_numgrad = th.zeros_like(alpha).view(-1)
    perturb = alpha_numgrad.clone()
    cost = lambda input: cost_weightNormalization(x,input,normalizedWeights,zeroMeanWeights,grad_output)
    
    for k in range(0,alpha.numel()):
        perturb[k]  = epsilon
        loss1 = cost(alpha.view(-1).add(perturb).view(sz_alpha))
        loss2 = cost(alpha.view(-1).add(-perturb).view(sz_alpha))
        alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0    
    
    alpha_numgrad = alpha_numgrad.view(sz_alpha)
    
    x_var = Variable(x,requires_grad = True)
    alpha_var = Variable(alpha,requires_grad = True)
    
    y = weightNormalizationF(x_var,alpha_var,normalizedWeights,zeroMeanWeights)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))
    
    if normalizedWeights :
        err_a = th.norm(alpha_var.grad.data.view(-1) - alpha_numgrad.view(-1))/\
        th.norm(alpha_var.grad.data.view(-1) + alpha_numgrad.view(-1))    
    else:
        err_a = None
    
    
    return err_x, x_var.grad.data, x_numgrad, err_a, alpha_var.grad.data, alpha_numgrad    

def EdgeTaper(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    
    from pydl.utils import gaussian_filter    
    
    EdgeTaperF = functional.EdgeTaper.apply
    
    blurKernel = th.from_numpy(gaussian_filter((31,33),10)).type(dtype)
    x =200*th.randn(2,3,50,50).type(dtype).abs()
    
    if GPU and th.cuda.is_available():
        blurKernel = blurKernel.cuda()
        x = x.cuda()
        
    grad_output = 200*th.randn(2,3,50,50).type(dtype)
    
    sz_x = x.size()
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone()
    cost = lambda input : cost_edgetaper(input,blurKernel,grad_output)
    
    for k in range(0,x.numel()):
        perturb[k] = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0        
    
    x_numgrad = x_numgrad.view(sz_x)
    
    x.requires_grad_()
    y = EdgeTaperF(x,blurKernel)
    y.backward(grad_output)    
    
    err_x = th.norm(x.grad.view(-1) - x_numgrad.view(-1))/\
        th.norm(x.grad.view(-1) + x_numgrad.view(-1))       
    
    return err_x, x.grad, x_numgrad

def WienerFilter(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False,\
     sharedChannels=False,sharedFilters=False,alphaSharedChannels=False,
     gradWeights=True,gradAlpha=True,gradInput=True,color=True):
    
    WienerFilterF = functional.WienerFilter.apply
    
    blurKernel = th.randn(5,5).type(dtype)
    batch,height,width = 2,50,50 
    channels = 3 if color else 1
    x = 200*th.randn(batch,channels,height,width).type(dtype)
    
    N = 4 # how many different wiener filters we use
    D = 8 # how many regularization filters we use
    if alphaSharedChannels:
        alpha = np.random.randint(1,10,(N,1))/100
    else:
        alpha = np.random.randint(1,10,(N,channels))/100
        
    alpha = th.from_numpy(alpha).type(dtype)
    alpha = alpha.log()
    
    wchannels = 1 if sharedChannels else channels        
    if sharedFilters:
        weights = th.randn(D,wchannels,3,3).type(dtype)
    else:
        weights = th.randn(N,D,wchannels,3,3).type(dtype)
    
    if GPU and th.cuda.is_available():
        weights = weights.cuda()
        x = x.cuda()
        alpha = alpha.cuda()
        blurKernel = blurKernel.cuda()
        
    grad_output = th.randn(x.size(0),N,*x.shape[1:]).type(dtype)
    
    if gradInput:
        sz_x = x.size()
        x_numgrad = th.zeros_like(x).view(-1)
        perturb = x_numgrad.clone()
        cost = lambda input: cost_WienerFilter(input,blurKernel,weights,alpha,grad_output)
        
        for k in range(0,x.numel()):
            perturb[k]  = epsilon
            loss1 = cost(x.view(-1).add(perturb).view(sz_x))
            loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
            x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
            perturb[k] = 0
            
        x_numgrad = x_numgrad.view(sz_x)
    
    if gradWeights:
        sz_w = weights.size()
        weights_numgrad = th.zeros_like(weights).view(-1)
        perturb = weights_numgrad.clone()
        cost = lambda input: cost_WienerFilter(x,blurKernel,input,alpha,grad_output)
        
        for k in range(0,weights.numel()):
            perturb[k]  = epsilon
            loss1 = cost(weights.view(-1).add(perturb).view(sz_w))
            loss2 = cost(weights.view(-1).add(-perturb).view(sz_w))
            weights_numgrad[k] = (loss1-loss2)/(2*perturb[k])
            perturb[k] = 0
            
        weights_numgrad = weights_numgrad.view(sz_w)
    
    if gradAlpha:        
        sz_a = alpha.size()
        alpha_numgrad = th.zeros_like(alpha).view(-1)
        perturb = alpha_numgrad.clone()
        cost = lambda input: cost_WienerFilter(x,blurKernel,weights,input,grad_output)
        
        for k in range(0,alpha.numel()):
            perturb[k]  = epsilon
            loss1 = cost(alpha.view(-1).add(perturb).view(sz_a))
            loss2 = cost(alpha.view(-1).add(-perturb).view(sz_a))
            alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
            perturb[k] = 0
            
        alpha_numgrad = alpha_numgrad.view(sz_a)
    
    if gradInput:
        x.requires_grad_()
    if gradWeights:
        weights.requires_grad_()
    if gradAlpha:
        alpha.requires_grad_()
    
    y = WienerFilterF(x,blurKernel,weights,alpha)[0]
    y.backward(grad_output)

    if gradInput:
        err_x = th.norm(x.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x.grad.data.view(-1) + x_numgrad.view(-1))   
    else:
        err_x = None
        x_numgrad = None
            
    if gradWeights:    
        err_w = th.norm(weights.grad.data.view(-1) - weights_numgrad.view(-1))/\
            th.norm(weights.grad.data.view(-1) + weights_numgrad.view(-1))    
    else:
        err_w = None
        weights_numgrad = None
        
    if gradAlpha:
        err_a = th.norm(alpha.grad.data.view(-1) - alpha_numgrad.view(-1))/\
            th.norm(alpha.grad.data.view(-1) + alpha_numgrad.view(-1))      
    else:
        err_a = None
        alpha_numgrad = None
    
    return err_x, x.grad, x_numgrad, err_w, weights.grad, weights_numgrad,\
    err_a, alpha.grad, alpha_numgrad
    
def imloss(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False,loss='psnr',peakVal=255):
    
    imlossF = functional.imLoss.apply
    
    x = th.randn(4,3,40,40).abs().type(dtype)
    x = x.div(x.max())*peakVal
    y = th.randn(4,3,40,40).abs().type(dtype)
    y = y.div(y.max())*peakVal
        
    if GPU and th.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
            
    sz_x = x.size()
    grad_output = th.ones(1).type_as(x)
    
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone()
    cost = lambda input: cost_imloss(input,y,loss,peakVal)
    
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0
        
    x_numgrad = x_numgrad.view(sz_x)
            
    x_var = Variable(x,requires_grad = True)
    
    z = imlossF(x_var,y,peakVal,loss)
    z.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))
            
    return err_x, x_var.grad.data, x_numgrad

def MSELoss(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False,peakVal=255,grad=False):
    
    MSELossF = functional.mseLoss.apply
    
    x = th.randn(4,3,40,40).abs().type(dtype)
    x = x.div(x.max())*peakVal
    y = th.randn(4,3,40,40).abs().type(dtype)
    y = y.div(y.max())*peakVal
        
    if GPU and th.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
            
    sz_x = x.size()
    grad_output = th.ones(1).type_as(x)
    
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone()
    cost = lambda input: cost_MSELoss(input,y,grad)
    
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0
        
    x_numgrad = x_numgrad.view(sz_x)
            
    x.requires_grad_()
    
    z = MSELossF(x,y,grad)
    z.backward(grad_output)
    
    err_x = th.norm(x.grad.view(-1) - x_numgrad.view(-1))/\
            th.norm(x.grad.view(-1) + x_numgrad.view(-1))
            
    return err_x, x.grad, x_numgrad


def cost_symmetricPad2D(x,pad):
    F = functional.SymmetricPad2D.apply
    out = F(x,pad)
    return out.sum()

def cost_symmetricPad_transpose2D(x,crop):
    F = functional.SymmetricPad_transpose2D.apply
    out = F(x,crop)
    return out.sum()
    
def cost_l2Proj(x,alpha,stdn,weights):
    F = functional.L2Proj.apply
    out = F(x,alpha,stdn)
    return out.mul(weights).sum()

def cost_SVl2Proj(x,alpha,stdn,weights):
    F = functional.SVL2Proj.apply
    out = F(x,alpha,stdn)
    return out.mul(weights).sum()

def cost_l2Prox(x,z,alpha,stdn,weights):
    F = functional.L2Prox.apply
    out = F(x,z,alpha,stdn)
    return out.mul(weights).sum()

def cost_SVl2Prox(x,z,alpha,stdn,weights):
    F = functional.SVL2Prox.apply
    out = F(x,z,alpha,stdn)
    return out.mul(weights).sum()

def cost_grbf_lut(x,weights,centers,sigma,origin,step,grad_weights):
    F = functional.Grbf_lut.apply
    out = F(x,weights,centers,sigma,origin,step)
    return out.mul(grad_weights).sum()

def cost_weightNormalization(x,alpha,normalizedWeights,zeroMeanWeights,weights):
    F = functional.WeightNormalization.apply
    out = F(x,alpha,normalizedWeights,zeroMeanWeights)
    return out.mul(weights).sum()

def cost_WienerFilter(x,blurKernel,weights,alpha,gweights):
    F = functional.WienerFilter.apply
    out = F(x,blurKernel,weights,alpha)[0]
    return out.mul(gweights).sum()

def cost_imloss(x,y,loss,peakVal):
    F = functional.imLoss.apply
    out = F(x,y,peakVal,loss)
    return out

def cost_MSELoss(x,y,grad,mode="normal"):
    F = functional.mseLoss.apply
    out = F(x,y,grad,mode)
    return out

def cost_edgetaper(x,blurKernel,weights):
    F = functional.EdgeTaper.apply
    out = F(x,blurKernel)
    return out.mul(weights).sum()