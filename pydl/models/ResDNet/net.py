#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:45:54 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""
import torch as th
from torch import nn
from pydl.nnLayers import modules
from pydl.nnLayers import init
from pydl.nnLayers.cascades import nconv2D, nconv_transpose2D
from pydl.nnLayers.functional.functional import L2Proj
from pydl.utils import formatInput2Tuple, getPad2RetainShape

class ResDNet(nn.Module):
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 convWeightSharing = True,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 conv_init = 'dct',\
                 bias_f= True,\
                 bias_t = True,\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True,\
                 alpha = True,\
                 rpa_depth = 5,\
                 rpa_kernel_size1 = (3,3),\
                 rpa_kernel_size2 = (3,3),\
                 rpa_output_features = 64,\
                 rpa_init = 'msra',\
                 rpa_bias1 = True,\
                 rpa_bias2 = True,\
                 rpa_prelu1_mc = True,\
                 rpa_prelu2_mc = True,\
                 prelu_init = 0.1,\
                 rpa_scale1 = True,\
                 rpa_scale2 = True,\
                 rpa_normalizedWeights = True,\
                 rpa_zeroMeanWeights = True,\
                 shortcut = (True,False),\
                 clb = 0,\
                 cub = 255):

        super(ResDNet, self).__init__()
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)
        
        if isinstance(pad,str) and pad == 'same':
            pad = getPad2RetainShape(kernel_size)
#            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
#            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
#                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))              
        
        self.pad = formatInput2Tuple(pad,int,4)
        self.padType = padType
        self.normalizedWeights = normalizedWeights
        self.zeroMeanWeights = zeroMeanWeights
        self.convWeightSharing = convWeightSharing

        # Initialize conv weights
        shape = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.convWeights(self.conv_weights,conv_init)

        # Initialize the scaling coefficients for the conv weight normalization
        if scale_f and normalizedWeights:
            self.scale_f = nn.Parameter(th.Tensor(output_features).fill_(1))
        else:
            self.register_parameter('scale_f', None)       
        
        # Initialize the bias for the conv layer
        if bias_f:
            self.bias_f = nn.Parameter(th.Tensor(output_features).fill_(0))
        else:
            self.register_parameter('bias_f', None)            

        # Initialize the bias for the transpose conv layer
        if bias_t:
            self.bias_t = nn.Parameter(th.Tensor(input_channels).fill_(0))
        else:
            self.register_parameter('bias_t', None)                      

        if not self.convWeightSharing:
            self.convt_weights = nn.Parameter(th.Tensor(th.Size(shape)))
            init.convWeights(self.convt_weights,conv_init)      

            if scale_t and normalizedWeights:
                self.scale_t = nn.Parameter(th.Tensor(output_features).fill_(1))
            else:
                self.register_parameter('scale_t', None)           
        
        
        numparams_prelu1 = output_features if rpa_prelu1_mc else 1
        numparams_prelu2 = rpa_output_features if rpa_prelu2_mc else 1
        
        self.rpa_depth = rpa_depth
        self.shortcut = formatInput2Tuple(shortcut,bool,rpa_depth,strict = False)
        self.resPA = nn.ModuleList([modules.ResidualPreActivationLayer(\
                        rpa_kernel_size1,rpa_kernel_size2,output_features,\
                        rpa_output_features,rpa_bias1,rpa_bias2,1,1,\
                        numparams_prelu1,numparams_prelu2,prelu_init,padType,\
                        rpa_scale1,rpa_scale2,rpa_normalizedWeights,\
                        rpa_zeroMeanWeights,rpa_init,self.shortcut[i]) \
                        for i in range(self.rpa_depth)]) 
        
        self.bbproj = nn.Hardtanh(min_val = clb, max_val = cub)  
        
        # Initialize the parameter for the L2Proj layer
        if alpha:
            self.alpha = nn.Parameter(th.Tensor(1).fill_(0))
        else:
            self.register_parameter('alpha',None)
        
    def forward(self,input,stdn):
        
        output = nconv2D(input,self.conv_weights,bias=self.bias_f,stride=1,\
                     pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_f,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)
        for m in self.resPA:
            output = m(output)
        
        if self.convWeightSharing:
            output = nconv_transpose2D(output,self.conv_weights,bias=self.bias_t,\
                     stride=1,pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_f,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)
        else:
            output = nconv_transpose2D(output,self.convt_weights,bias=self.bias_t,\
                     stride=1,pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_t,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)            
        
        output = L2Proj.apply(output,self.alpha,stdn)
        return self.bbproj(input-output)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'depth = ' + str(self.rpa_depth) \
            + ', convWeightSharing = ' + str(self.convWeightSharing)\
            + ', shortcut = ' + str(self.shortcut) + ')' 

class ResDNet_old(nn.Module):
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 convWeightSharing = True,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 conv_init = 'dct',\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True,\
                 alpha = True,\
                 rpa_depth = 5,\
                 rpa_kernel_size1 = (3,3),\
                 rpa_kernel_size2 = (3,3),\
                 rpa_output_features = 64,\
                 rpa_init = 'msra',\
                 rpa_bias1 = False,\
                 rpa_bias2 = False,\
                 numparams_prelu1 = 1,\
                 numparams_prelu2 = 1,\
                 prelu_init = 0.1,\
                 rpa_scale1 = True,\
                 rpa_scale2 = True,\
                 rpa_normalizedWeights = True,\
                 rpa_zeroMeanWeights = True,\
                 shortcut = (True,False),\
                 clb = 0,\
                 cub = 255):

        super(ResDNet, self).__init__()
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)
        
        if isinstance(pad,str) and pad == 'same':
            pad = getPad2RetainShape(kernel_size)
#            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
#            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
#                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))              
        
        self.pad = formatInput2Tuple(pad,int,4)
        self.padType = padType
        self.normalizedWeights = normalizedWeights
        self.zeroMeanWeights = zeroMeanWeights
        self.convWeightSharing = convWeightSharing

        # Initialize conv weights
        shape = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.convWeights(self.conv_weights,conv_init)

        # Initialize the scaling coefficients for the conv weight normalization
        if scale_f and normalizedWeights:
            self.scale_f = nn.Parameter(th.Tensor(output_features).fill_(1))
        else:
            self.register_parameter('scale_f', None)        

        if not self.convWeightSharing:
            self.convt_weights = nn.Parameter(th.Tensor(th.Size(shape)))
            init.convWeights(self.convt_weights,conv_init)      

            if scale_t and normalizedWeights:
                self.scale_t = nn.Parameter(th.Tensor(output_features).fill_(1))
            else:
                self.register_parameter('scale_t', None)
        
        self.rpa_depth = rpa_depth
        self.shortcut = formatInput2Tuple(shortcut,bool,rpa_depth,strict = False)
        self.resPA = nn.ModuleList([modules.ResidualPreActivationLayer(\
                        rpa_kernel_size1,rpa_kernel_size2,output_features,\
                        rpa_output_features,rpa_bias1,rpa_bias2,1,1,\
                        numparams_prelu1,numparams_prelu2,prelu_init,padType,\
                        rpa_scale1,rpa_scale2,rpa_normalizedWeights,\
                        rpa_zeroMeanWeights,rpa_init,self.shortcut[i]) \
                        for i in range(self.rpa_depth)]) 
        
        self.bbproj = nn.Hardtanh(min_val = clb, max_val = cub)  
        
        # Initialize the parameter for the L2Proj layer
        if alpha:
            self.alpha = nn.Parameter(th.Tensor(1).fill_(0))
        else:
            self.register_parameter('alpha',None)
        
    def forward(self,input,stdn):
        
        output = nconv2D(input,self.conv_weights,bias=None,stride=1,\
                     pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_f,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)
        for m in self.resPA:
            output = m(output)
        
        if self.convWeightSharing:
            output = nconv_transpose2D(output,self.conv_weights,bias=None,\
                     stride=1,pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_f,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)
        else:
            output = nconv_transpose2D(output,self.convt_weights,bias=None,\
                     stride=1,pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_t,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)            
        
        output = L2Proj.apply(output,self.alpha,stdn)
        return self.bbproj(input-output)
        


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'depth = ' + str(self.rpa_depth) \
            + ', convWeightSharing = ' + str(self.convWeightSharing)\
            + ', shortcut = ' + str(self.shortcut) + ')' 


def loadModel(filePath,location='cpu',gpu_device=0):
    """Loads a trained model.
    
    filePath : the path of the file containing the parameters and the architecture
               of the model.
    location : Where to load the model. (Default: cpu)
    """
    if location == 'gpu' and th.cuda.is_available():
        if gpu_device != th.cuda.current_device()\
            and not (gpu_device >= 0 and gpu_device < th.cuda.device_count()):
                gpu_device = th.cuda.current_device()
                
        state = th.load(filePath,map_location=lambda storage,loc:storage.cuda(gpu_device))
    elif location == 'cpu':
        state =  th.load(filePath,map_location=lambda storage,loc:storage)
    else:
        raise Exception("Unknown device to load the model.")
    
    # state['params'] is an Ordered dictionary with the following keys:
    #    odict_keys(['kernel_size', 'input_channels', 'output_features', \
    #   'rbf_mixtures', 'rbf_precision', 'stages', 'pad', 'padType', \
    #  'convWeightSharing', 'scale_f', 'scale_t', 'normalizedWeights', \
    #   'zeroMeanWeights', 'rbf_start', 'rbf_end', 'data_min', 'data_max', \
    #   'data_step', 'alpha', 'clb', 'cub'])
    
    model = ResDNet(*state['params'].values())
    model.load_state_dict(state['model_state_dict'])

    return model

def ResDNet_denoise(y,stdn):
    import os.path
       
    assert(isinstance(stdn,(float,int)) or \
           (th.is_tensor(stdn) and stdn.numel()==y.size(0))),\
           "The second argument must be a tensor or an int or a float."
    
    if not th.is_tensor(stdn):
        stdn = th.Tensor([stdn]).type_as(y)
    
    while y.dim() < 4:
        y = y.unsqueeze(0)
    
    currentPath = os.path.dirname(os.path.realpath(__file__))
    mpath = os.path.join(currentPath,'models','ResDNet_')
    
    batch,channels,H,W = y.shape
    
    if channels == 1:
        mpath += "GD5F64.md"
    elif channels == 3:
        mpath += "CD5F64.md"        
    else: 
        raise ValueError("Input tensor must have either one or three channels.")
    
    model = loadModel(mpath)
    if y.is_cuda:
        model = model.cuda()
    
    with th.no_grad(): out = model(y,stdn)
    
    return out       