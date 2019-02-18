#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:49:14 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import torch as th
from torch import nn
from pydl.nnLayers import cascades
from pydl.nnLayers.functional.functional import imLoss, mseLoss, Pad2D, Crop2D, \
WienerFilter, WeightNormalization, WeightNormalization5D, EdgeTaper
from pydl.nnLayers import init
from pydl.utils import formatInput2Tuple, getPad2RetainShape
from math import log10
#from collections import OrderedDict

#from functools import reduce
#prod = lambda f: reduce(lambda x,y : x*y,f)

class WienerDeconvLayer(nn.Module):
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 numWienerFilters = 4,\
                 sharedWienerFilters = False,\
                 sharedChannels = True,\
                 sharedAlphaChannels = True,\
                 alpha_update = True,\
                 lb = 1e-4,\
                 ub = 1e-2,\
                 pad = True,\
                 padType = 'symmetric',\
                 edgeTaper = True,\
                 scale = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True):
        
        super(WienerDeconvLayer,self).__init__()
        
        self.pad = pad
        self.padType = padType
        self.edgetaper = edgeTaper
        self.sharedWienerFilters = sharedWienerFilters
        self.normalizedWeights = normalizedWeights
        self.zeroMeanWeights = zeroMeanWeights
        
        assert(numWienerFilters > 1),"More than one Wiener filter is expected."
        
        # Initialize conv regularization weights 
        channels = 1 if sharedChannels else input_channels
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)
        
        if sharedWienerFilters:
            shape = (output_features,channels)+kernel_size
        else:
            shape = (numWienerFilters,output_features,channels)+kernel_size
            
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.dctMultiWiener(self.conv_weights)
        
        if scale and normalizedWeights:
            if sharedWienerFilters:
                self.scale = nn.Parameter(th.Tensor(output_features).fill_(0.1))   
            else:
                self.scale = nn.Parameter(th.Tensor(numWienerFilters,output_features).fill_(0.1))   
        else:
            self.register_parameter('scale', None)             
                
        assert(lb > 0 and ub > 0),"Lower (lb) and upper (ub) bounds of the "\
        +"alpha parameter must be positive numbers."
        alpha = th.logspace(log10(lb),log10(ub),numWienerFilters).unsqueeze(-1).log()
        if sharedAlphaChannels:            
            shape = (numWienerFilters,1)
        else:
            alpha = alpha.repeat(1,input_channels)
            shape = (numWienerFilters,input_channels)
        
        if alpha_update:       
            self.alpha = nn.Parameter(th.Tensor(th.Size(shape)))
            self.alpha.data.copy_(alpha)
        else:
            self.alpha = alpha
    
    def forward(self,input,blurKernel,stdn):
        
        if self.pad:
            pad = getPad2RetainShape(blurKernel.shape)
            input = Pad2D.apply(input,pad,self.padType)
        
        if self.edgetaper:
            input = EdgeTaper.apply(input,blurKernel)
    
        
        if self.sharedWienerFilters:        
            conv_weights = WeightNormalization.apply(self.conv_weights,self.scale,\
                                        self.normalizedWeights,self.zeroMeanWeights)
        else:
            conv_weights = WeightNormalization5D.apply(self.conv_weights,self.scale,\
                                        self.normalizedWeights,self.zeroMeanWeights)
                    
        output, cstdn = WienerFilter.apply(input,blurKernel,conv_weights,self.alpha)
        
        # compute the standard deviation of the remaining colored noise in the output
        cstdn = th.sqrt(stdn.type_as(cstdn).unsqueeze(-1).pow(2).mul(cstdn.mean(dim=2)))
        
        if self.pad:
            output = Crop2D.apply(output,pad)
        
        return output, cstdn
        

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size = ' + str(tuple(self.conv_weights.shape[-2:])) \
            + ', input_channels = ' + str(self.conv_weights.size(-3)) \
            + ', output_features = ' + str(self.conv_weights.size(-4)) \
            + ', WienerFilters = ' + str(self.alpha.size(0)) \
            + ', edgeTaper = ' + str(self.edgetaper) + ')'
        
           
class ResidualRBFLayer(nn.Module):
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 rbf_mixtures,\
                 rbf_precision,\
                 pad = 'same',\
                 convWeightSharing = True,\
                 alpha = True,
                 lb = -100,\
                 ub = 100,\
                 padType = 'symmetric',\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True):
        
        super(ResidualRBFLayer, self).__init__()
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)       
        
        if isinstance(pad,str) and pad == 'same':
            pad = getPad2RetainShape(kernel_size)
#            # center of the kernel
#            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
#            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
#                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))
            
        self.pad = formatInput2Tuple(pad,int,4)
        self.padType = padType
        self.normalizedWeights = normalizedWeights
        self.zeroMeanWeights = zeroMeanWeights
        self.convWeightSharing = convWeightSharing
        self.lb = lb
        self.ub = ub
        
        # Initialize conv weights
        shape = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.dct(self.conv_weights)
        
        # Initialize the scaling coefficients for the conv weight normalization
        if scale_f and normalizedWeights:
            self.scale_f = nn.Parameter(th.Tensor(output_features).fill_(0.1))
        else:
            self.register_parameter('scale_f', None)        
        
        if not self.convWeightSharing:
            self.convt_weights = nn.Parameter(th.Tensor(th.Size(shape)))
            init.dct(self.convt_weights)
        
            if scale_t and normalizedWeights:
                self.scale_t = nn.Parameter(th.Tensor(output_features).fill_(0.1))
            else :
                self.register_parameter('scale_t', None)
        
        # Initialize the params for the proxL2
        if alpha :
            self.alpha_prox = nn.Parameter(th.Tensor(1).fill_(0.1))
        else:
            self.register_parameter('alpha_prox', None)
        
        # Initialize the rbf_weights
        self.rbf_weights = nn.Parameter(th.Tensor(output_features,rbf_mixtures).fill_(1e-4))
        self.rbf_centers = th.linspace(lb,ub,rbf_mixtures).type_as(self.rbf_weights)
        self.rbf_precision = rbf_precision
        #self.rbf_data = init.rbf_lut(self.rbf_centers,self.rbf_precision,rbf_start,rbf_end,rbf_step).type_as(self.rbf_weights)
        
    def forward(self,input,stdn,rbf_data,net_input = None):
        if net_input is None:
            # If input is a variable with require_grad = True, then net_input 
            # will be a variable with require_grad = False. 
            # net_input = input.data.clone() 
            net_input = input
        if self.convWeightSharing:
            return cascades.residualDenoise_grbf_sw(input,net_input,\
                    self.conv_weights,self.rbf_weights,self.rbf_centers,\
                    self.rbf_precision,rbf_data,self.alpha_prox,stdn,self.pad,\
                    self.padType,self.scale_f,self.normalizedWeights,\
                    self.zeroMeanWeights,self.lb,self.ub)
        else:
            return cascades.residualDenoise_grbf(input,net_input,self.conv_weights,\
                    self.convt_weights,self.rbf_weights,self.rbf_centers,\
                    self.rbf_precision,rbf_data,self.alpha_prox,stdn,self.pad,\
                    self.padType,self.scale_f,self.scale_t,self.normalizedWeights,\
                    self.zeroMeanWeights,self.lb,self.ub)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size = ' + str(tuple(self.conv_weights.shape[2:])) \
            + ', input_channels = ' + str(self.conv_weights.shape[1]) \
            + ', output_features = ' + str(self.conv_weights.shape[0]) \
            + ', rbf_mixtures = ' + str(self.rbf_centers.numel()) \
            + ', normalizedWeights = ' + str(self.normalizedWeights) \
            + ', zeroMeanWeights = ' + str(self.zeroMeanWeights) \
            + ', convWeightSharing = ' + str(self.convWeightSharing) + ')'

class ResidualRBFLayer_old(nn.Module):
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 rbf_mixtures,\
                 rbf_precision,\
                 pad = 'same',\
                 convWeightSharing = True,\
                 alpha = True,
                 lb = -100,\
                 ub = 100,\
                 padType = 'symmetric',\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True):
        
        super(ResidualRBFLayer, self).__init__()
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)       
        
        if isinstance(pad,str) and pad == 'same':
            pad = getPad2RetainShape(kernel_size)
#            # center of the kernel
#            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
#            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
#                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))
            
        self.pad = formatInput2Tuple(pad,int,4)
        self.padType = padType
        self.normalizedWeights = normalizedWeights
        self.zeroMeanWeights = zeroMeanWeights
        self.lb = lb
        self.ub = ub
        
        # Initialize conv weights
        shape = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.dct(self.conv_weights)
        
        if convWeightSharing:
            self.convt_weights = self.conv_weights
        else:
            self.convt_weights = nn.Parameter(th.Tensor(th.Size(shape)))
            init.dct(self.convt_weights)
        
        # Initialize the scaling coefficients for the conv weight normalization
        if scale_f and normalizedWeights:
            self.scale_f = nn.Parameter(th.Tensor(output_features).fill_(1))
        else:
            self.register_parameter('scale_f', None)
        
        if scale_t and normalizedWeights:
            if convWeightSharing and scale_f:
                self.scale_t = self.scale_f
            elif not convWeightSharing or (convWeightSharing and not scale_f):
                self.scale_t = nn.Parameter(th.Tensor(output_features).fill_(1))
        else :
            self.register_parameter('scale_t', None)
        
        # Initialize the params for the proxL2
        if alpha :
            self.alpha_prox = nn.Parameter(th.Tensor(1).fill_(0))
        else:
            self.register_parameter('alpha_prox', None)
        
        # Initialize the rbf_weights
        self.rbf_weights = nn.Parameter(th.Tensor(output_features,rbf_mixtures).fill_(1e-4))
        self.rbf_centers = th.linspace(lb,ub,rbf_mixtures).type_as(self.rbf_weights)
        self.rbf_precision = rbf_precision
        #self.rbf_data = init.rbf_lut(self.rbf_centers,self.rbf_precision,rbf_start,rbf_end,rbf_step).type_as(self.rbf_weights)
        
    def forward(self,input,stdn,rbf_data,net_input = None):
        if net_input is None:
            # If input is a variable with require_grad = True, then net_input 
            # will be a variable with require_grad = False. 
            # net_input = input.data.clone() 
            net_input = input
        return cascades.residualDenoise_grbf(input,net_input,self.conv_weights,\
            self.convt_weights,self.rbf_weights,self.rbf_centers,\
            self.rbf_precision,rbf_data,self.alpha_prox,stdn,self.pad,\
            self.padType,self.scale_f,self.scale_t,self.normalizedWeights,\
            self.zeroMeanWeights,self.lb,self.ub)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size = ' + str(tuple(self.conv_weights.shape[2:])) \
            + ', input_channels = ' + str(self.conv_weights.shape[1]) \
            + ', output_features = ' + str(self.conv_weights.shape[0]) \
            + ', rbf_mixtures = ' + str(self.rbf_centers.numel()) \
            + ', normalizedWeights = ' + str(self.normalizedWeights) \
            + ', zeroMeanWeights = ' + str(self.zeroMeanWeights) \
            + ', convWeightSharing = ' + str(self.conv_weights is self.convt_weights) + ')'
        
class ResidualPreActivationLayer(nn.Module):
    
    def __init__(self, kernel1_size,\
                 kernel2_size,\
                 input_channels,\
                 output_features,\
                 bias1 = False,\
                 bias2 = False,\
                 dilation1 = 1,\
                 dilation2 = 1,\
                 numparams_prelu1 = 1,\
                 numparams_prelu2 = 1,\
                 prelu_init = 0.1,\
                 padType = 'symmetric',\
                 scale1 = True,\
                 scale2 = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True,\
                 weights_init = 'msra',\
                 shortcut = False):
        
        super(ResidualPreActivationLayer, self).__init__()
        
        self.normalizedWeights = formatInput2Tuple(normalizedWeights,bool,2)
        self.zeroMeanWeights = formatInput2Tuple(zeroMeanWeights,bool,2)
        self.shortcut = shortcut
        self.dilation1 = formatInput2Tuple(dilation1,int,2)
        self.dilation2 = formatInput2Tuple(dilation2,int,2)
        self.padType = padType
        
        kernel1_size = formatInput2Tuple(kernel1_size,int,2)
        kernel2_size = formatInput2Tuple(kernel2_size,int,2)

        # Init of conv1 weights
        shape = (output_features,input_channels)+kernel1_size
        self.conv1_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.msra(self.conv1_weights)
        # Init of conv2 weights
        shape = (input_channels,output_features)+kernel2_size
        self.conv2_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.convWeights(self.conv2_weights,weights_init)
        
        # Initialize the scaling coefficients for the conv weights normalization
        if scale1 and self.normalizedWeights[0]:
            self.scale1 = nn.Parameter(th.Tensor(output_features).fill_(0.1))
        else:
            self.register_parameter('scale1', None)        
        
        if scale2 and self.normalizedWeights[1]:
            self.scale2 = nn.Parameter(th.Tensor(input_channels).fill_(0.1))
        else:
            self.register_parameter('scale2', None)        

        if bias1:
            self.bias1 = nn.Parameter(th.Tensor(output_features).fill_(0))
        else:
            self.register_parameter('bias1', None)    

        if bias2:
            self.bias2 = nn.Parameter(th.Tensor(input_channels).fill_(0))
        else:
            self.register_parameter('bias2', None)               

        # Init of prelu weights
        self.prelu1_weights = nn.Parameter(th.Tensor(numparams_prelu1).fill_(prelu_init))
        self.prelu2_weights = nn.Parameter(th.Tensor(numparams_prelu2).fill_(prelu_init))
    
    def forward(self,input):
        return cascades.residualPreActivation(input,self.conv1_weights,\
                self.conv2_weights,self.prelu1_weights,self.prelu2_weights,\
                self.bias1,self.scale1,self.dilation1,self.bias2,self.scale2,\
                self.dilation2,self.normalizedWeights,self.zeroMeanWeights,\
                self.shortcut,self.padType)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel1_size = ' + str(tuple(self.conv1_weights.shape[2:])) \
            + ', kernel2_size = ' + str(tuple(self.conv2_weights.shape[2:])) \
            + ', input_channels = ' + str(self.conv1_weights.shape[1]) \
            + ', output_features = ' + str(self.conv1_weights.shape[0]) \
            + ', shortcut = ' + str(self.shortcut) \
            + ', normalizedWeights = ' + str(self.normalizedWeights) \
            + ', zeroMeanWeights = ' + str(self.zeroMeanWeights) + ')'

class NConv2D(nn.Module):
    
    def __init__(self,kernel_size,\
                 input_channels,\
                 output_features,\
                 bias=False,\
                 stride=1,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 conv_init = 'dct',\
                 scale = False,\
                 normalizedWeights=False,\
                 zeroMeanWeights=False):
        
        super(NConv2D,self).__init__()
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)
        
        if isinstance(pad,str) and pad == 'same':
            pad = getPad2RetainShape(kernel_size)
#            # center of the kernel
#            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
#            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
#                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))        
        
        self.pad = formatInput2Tuple(pad,int,4)
        self.padType = padType
        self.stride = formatInput2Tuple(stride,int,2)
        
        # Initialize conv weights
        shape = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.convWeights(self.conv_weights,conv_init)
        
        # Initialize the scaling coefficients for the conv weight normalization
        if scale and normalizedWeights:
            self.scale = nn.Parameter(th.Tensor(output_features).fill_(1))
        else:
            self.register_parameter('scale', None)
        
        if bias:
            self.bias = nn.Parameter(th.Tensor(output_features).fill_(0))
        else:
            self.register_parameter('bias', None)           
    
    def forward(self,input):
        
        return cascades.nconv2D(input,self.conv_weights,self.bias,self.stride,\
                                self.pad,self.padType,1,self.scale,\
                                self.normalizedWeights,self.zeroMeanWeights)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size = ' + str(tuple(self.conv_weights.shape[2:])) \
            + ', input_channels = ' + str(self.conv_weights.shape[1]) \
            + ', output_features = ' + str(self.conv_weights.shape[0]) \
            + ', padType = ' + self.padType\
            + ', pad = ' + str(self.pad)\
            + ', normalizedWeights = ' + str(self.normalizedWeights) \
            + ', zeroMeanWeights = ' + str(self.zeroMeanWeights) + ')'

class NConv_transpose2D(nn.Module):
    
    def __init__(self,kernel_size,\
                 input_channels,\
                 output_features,\
                 bias=False,\
                 stride=1,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 conv_init = 'dct',\
                 scale = False,\
                 normalizedWeights=False,\
                 zeroMeanWeights=False):
        
        super(NConv_transpose2D,self).__init__()
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)
        
        if isinstance(pad,str) and pad == 'same':
            # center of the kernel
            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))        
        
        
        self.pad = formatInput2Tuple(pad,int,4)
        self.padType = padType
        self.stride = formatInput2Tuple(stride,int,2)
        
        # Initialize conv weights
        shape = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.convWeights(self.conv_weights,conv_init)
        
        # Initialize the scaling coefficients for the conv weight normalization
        if scale and normalizedWeights:
            self.scale = nn.Parameter(th.Tensor(output_features).fill_(1))
        else:
            self.register_parameter('scale', None)
        
        if bias:
            self.bias = nn.Parameter(th.Tensor(input_channels).fill_(0))
        else:
            self.register_parameter('bias', None)           
    
    def forward(self,input):
        
        return cascades.nconv_transpose2D(input,self.conv_weights,self.bias,\
                             self.stride,self.pad,self.padType,1,self.scale,\
                             self.normalizedWeights,self.zeroMeanWeights)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size = ' + str(tuple(self.conv_weights.shape[2:])) \
            + ', input_channels = ' + str(self.conv_weights.shape[1]) \
            + ', output_features = ' + str(self.conv_weights.shape[0]) \
            + ', padType = ' + self.padType\
            + ', pad = ' + str(self.pad)\
            + ', normalizedWeights = ' + str(self.normalizedWeights) \
            + ', zeroMeanWeights = ' + str(self.zeroMeanWeights) + ')'


class MSELoss(nn.Module):
    def __init__(self,grad=False):
        
        super(MSELoss,self).__init__()
        
        self.grad = grad
        self.mode = "validation"
    
    def forward(self,input,target):
        
        return mseLoss.apply(input,target,self.grad,self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + 'gradMSE = ' + str(self.grad)\
        + ', mode = ' + self.mode + ')'        
   
    
class PSNRLoss(nn.Module):
    
    def __init__(self,peakval=255):
        
        super(PSNRLoss,self).__init__()       
        
        self.peakval = peakval
        self.loss = 'psnr'
        self.mode = 'normal'
        
    def forward(self,input,other):
        
        return imLoss.apply(input,other,self.peakval,self.loss,self.mode) 
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'peakVal = ' + str(self.peakval) + ')'    