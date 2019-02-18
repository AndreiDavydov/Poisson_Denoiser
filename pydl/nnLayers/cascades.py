#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:50:33 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import torch as th
from pydl.nnLayers.functional import functional as F
from pydl.utils import formatInput2Tuple, getPad2RetainShape

pad2D = F.Pad2D.apply
pad_transpose2D = F.Pad_transpose2D.apply
l2Prox = F.L2Prox.apply
weightNormalization = F.WeightNormalization.apply
grbf = F.grbf_LUT.apply
conv2d = th.nn.functional.conv2d
conv2d_t = th.nn.functional.conv_transpose2d

def nconv2D(input,weights,bias=None,stride=1,pad=0,padType='zero',\
            dilation=1,scale=None,normalizedWeights=False,\
            zeroMeanWeights=False):
    r"""2D Convolution of an input tensor of size B x C X H x W where the 
    weights of the filters are normalized. 
    """
    
    stride = formatInput2Tuple(stride,int,2)
    pad = formatInput2Tuple(pad,int,4)
    dilation = formatInput2Tuple(dilation,int,2)
    
    assert(input.dim() == 4), "The dimensions of the input tensor are "\
        +"expected to be equal to 4."    
    
    if sum(pad) != 0 :
        input = pad2D(input,pad,padType)
    
    weights = weightNormalization(weights,scale,normalizedWeights,zeroMeanWeights)
    
    return th.nn.functional.conv2d(input,weights,bias,stride=stride,\
                                   dilation=dilation)

def nconv_transpose2D(input,weights,bias=None,stride=1,pad=0,padType='zero',\
                      dilation=1,scale=None,normalizedWeights=False,\
                      zeroMeanWeights=False):
    r"""Transpose 2D normalized convolution."""
    
    stride = formatInput2Tuple(stride,int,2)
    pad = formatInput2Tuple(pad,int,4)
    dilation = formatInput2Tuple(dilation,int,2)
    
    assert(input.dim() == 4), "The dimensions of the input tensor are "\
        +"expected to be equal to 4."    
    
    weights = weightNormalization(weights,scale,normalizedWeights,zeroMeanWeights)
    
    out = th.nn.functional.conv_transpose2d(input,weights,bias,stride=stride,\
                                            dilation=dilation)
    
    if sum(pad) != 0:
        out = pad_transpose2D(out,pad,padType)
        
    return out

def residualDenoise_grbf(input,net_input,weights,weights_t,rbf_weights,rbf_centers,\
                  rbf_precision,data_lut,alpha_prox,stdn,pad=0,padType='zero',\
                  alpha=None,alpha_t=None,normalizedWeights=False,\
                  zeroMeanWeights=False,lb=-100,ub=100):
    r""" residual layer with independent weights between the convolutional and 
    transpose convolutional layers."""
    
    if (normalizedWeights and alpha is not None) or zeroMeanWeights:
        weights = weightNormalization(weights,alpha,normalizedWeights,zeroMeanWeights)
    if (normalizedWeights and alpha_t is not None) or zeroMeanWeights:
        weights_t = weightNormalization(weights_t,alpha_t,normalizedWeights,zeroMeanWeights)
    
    # conv2d
    out = conv2d(pad2D(input,pad,padType),weights,bias = None,stride = 1)
    #clipping of the values before feeding them to the grbf layer
    out = th.clamp(out,lb,ub)
    # gaussian rbf
    out = grbf(out,rbf_weights,rbf_centers,rbf_precision,data_lut)
    # conv_tranpose2d
    out = pad_transpose2D(conv2d_t(out,weights_t,bias = None,stride = 1),pad,padType)
    # Projection of the result, given the input of the network
    out = l2Prox(input-out,net_input,alpha_prox,stdn)
    
    return out

def residualDenoise_grbf_sw(input,net_input,weights,rbf_weights,rbf_centers,\
                  rbf_precision,data_lut,alpha_prox,stdn,pad=0,padType='zero',\
                  alpha=None,normalizedWeights=False,zeroMeanWeights=False,\
                  lb=-100,ub=100):
    r""" residual layer with shared weights between the convolutional and 
    transpose convolutional layers."""
    
    if (normalizedWeights and alpha is not None) or zeroMeanWeights:
        weights = weightNormalization(weights,alpha,normalizedWeights,zeroMeanWeights)

    # conv2d
    out = conv2d(pad2D(input,pad,padType),weights,bias = None,stride = 1)
    #clipping of the values before feeding them to the grbf layer
    out = th.clamp(out,lb,ub)
    # gaussian rbf
    out = grbf(out,rbf_weights,rbf_centers,rbf_precision,data_lut)
    # conv_tranpose2d
    out = pad_transpose2D(conv2d_t(out,weights,bias = None,stride = 1),pad,padType)
    # Projection of the result, given the input of the network
    out = l2Prox(input-out,net_input,alpha_prox,stdn)
    
    return out

def residualPreActivation(input, conv1_weights, conv2_weights, \
      prelu1_weights, prelu2_weights, bias1 = None, scale1 = None, \
      dilation1 = 1, bias2 = None, scale2 = None, dilation2 = 1, \
      normalizedWeights = False, zeroMeanWeights = False, shortcut = False,\
      padType = 'symmetric'):
        
    normalizedWeights = formatInput2Tuple(normalizedWeights,bool,2)    
    zeroMeanWeights = formatInput2Tuple(zeroMeanWeights,bool,2)    

    assert(any(prelu1_weights.numel() == i for i in (1,input.size(1)))),\
        "Dimensions mismatch between input and prelu1_weights."
    assert(any(prelu2_weights.numel() == i for i in (1,conv1_weights.size(0)))),\
        "Dimensions mismatch between conv1_weights and prelu2_weights."
    assert(conv1_weights.size(0) == conv2_weights.size(1)), "Dimensions "+\
        "mismatch between conv1_weights and conv2_weights."
    assert(conv2_weights.size(0) == input.size(1)), "Dimensions "+\
        "mismatch between conv2_weights and input."        
    
    if (normalizedWeights[0] and scale1 is not None) or zeroMeanWeights[0]:
        conv1_weights = weightNormalization(conv1_weights,scale1,\
                                    normalizedWeights[0],zeroMeanWeights[0])

    if (normalizedWeights[1] and scale2 is not None) or zeroMeanWeights[1]:
        conv2_weights = weightNormalization(conv2_weights,scale2,\
                                    normalizedWeights[1],zeroMeanWeights[1])
    
    pad1 = getPad2RetainShape(conv1_weights.shape[2:4],dilation1)
    pad2 = getPad2RetainShape(conv2_weights.shape[2:4],dilation2)
    
    out = th.nn.functional.prelu(input,prelu1_weights)
    out = conv2d(pad2D(out,pad1,padType),conv1_weights,bias = bias1, dilation = dilation1)
    out = th.nn.functional.prelu(out,prelu2_weights)
    out = conv2d(pad2D(out,pad2,padType),conv2_weights,bias = bias2, dilation = dilation2)
    
    if shortcut:
        return out.add(input)
    else:
        return out
