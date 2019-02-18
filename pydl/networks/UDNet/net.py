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
from pydl.utils import loadmat

class UDNet(nn.Module):
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 rbf_mixtures,\
                 rbf_precision,\
                 stages = 5,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 convWeightSharing = True,\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True,\
                 rbf_start = -100,\
                 rbf_end = 100,\
                 data_min = -100,\
                 data_max = 100,\
                 data_step = 0.1,\
                 alpha = True,\
                 clb = 0,\
                 cub = 255):
        
        super(UDNet, self).__init__()
        
        rbf_centers = th.linspace(rbf_start,rbf_end,rbf_mixtures).type_as(th.Tensor())       
        self.rbf_data_lut = init.rbf_lut(rbf_centers,rbf_precision,data_min,\
                                         data_max,data_step)
        self.stages = stages        
        
        self.resRBF = nn.ModuleList([modules.ResidualRBFLayer(kernel_size,\
                    input_channels,output_features,rbf_mixtures,\
                    rbf_precision,pad,convWeightSharing,alpha,rbf_start,\
                    rbf_end,padType,scale_f,scale_t,normalizedWeights,\
                    zeroMeanWeights) for i in range(self.stages)])        
        self.bbProj = nn.Hardtanh(min_val = clb, max_val = cub)
        
    def forward(self,input,stdn,net_input = None):
        if net_input is None:
            net_input = input
            
        for m in self.resRBF:
            input = m(input,stdn,self.rbf_data_lut.type_as(input),net_input)
        
        return self.bbProj(input)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'stages = ' + str(self.stages) + ')'    

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
    
    model = UDNet(*state['params'].values())
    model.load_state_dict(state['model_state_dict'])

    return model

def loadModelfromMatlab(filepath,loadParams=False):
    """Creates a pytorch UDNet model from a matlab UDNet model. If loadParams
    is True the model and the parameters to initialize the model are returned.
    Otherwise only the model is returned."""
    
    from collections import OrderedDict
    from argparse import Namespace
    import numpy as np
    
    f = loadmat(filepath)
    d = f['net']['layers']
    step = f['net']['meta']['netParams']['step']
    wshape = np.asarray(d[0]['weights'][0]).shape
    input_channels = 1 if len(wshape) == 3 else wshape[-2]
    
    del f
    
    opt = Namespace(kernel_size = wshape[0:2],\
                    input_channels = input_channels,\
                    output_features = wshape[-1],\
                    rbf_mixtures = len(d[0]['rbf_means']),\
                    rbf_precision = d[0]['rbf_precision'],\
                    stages = len(d)-1,\
                    pad = tuple(d[0]['padSize']),\
                    padType = d[0]['padType'],\
                    convWeightSharing = not bool(d[0]['learningRate'][1]),\
                    scale_f = bool(d[0]['learningRate'][2]),\
                    scale_t = bool(d[0]['learningRate'][3]),\
                    normalizedWeights = bool(d[0]['weightNormalization']),\
                    zeroMeanWeights = bool(d[0]['zeroMeanFilters']),\
                    rbf_start = float(d[0]['rbf_means'][0]),\
                    rbf_end = float(d[0]['rbf_means'][-1]),\
                    data_min = d[0]['lb'],\
                    data_max = d[0]['ub'],\
                    data_step = step,\
                    alpha = bool(d[0]['learningRate'][5]),\
                    clb = d[-1]['lb'],\
                    cub = d[-1]['ub'])
    
    
    params = OrderedDict(kernel_size=opt.kernel_size,input_channels=opt.input_channels,\
         output_features=opt.output_features,rbf_mixtures=opt.rbf_mixtures,\
         rbf_precision=opt.rbf_precision,stages=opt.stages,pad=opt.pad,\
         padType=opt.padType,convWeightSharing=opt.convWeightSharing,\
         scale_f=opt.scale_f,scale_t=opt.scale_t,normalizedWeights=\
         opt.normalizedWeights,zeroMeanWeights=opt.zeroMeanWeights,rbf_start=\
         opt.rbf_start,rbf_end=opt.rbf_end,data_min=opt.data_min,data_max=\
         opt.data_max,data_step=opt.data_step,alpha=opt.alpha,clb=opt.clb,\
         cub=opt.cub)
    
    # Create a pytorch model with the same parameters as the matlab model
    model = UDNet(*params.values())
    state_dict = model.state_dict()
    
    
    for i in range(opt.stages):
        if input_channels == 1:
            state_dict['resRBF.'+str(i)+'.conv_weights'] = \
                th.Tensor(d[i]['weights'][0]).permute(2,0,1).unsqueeze(1)
        else:
            state_dict['resRBF.'+str(i)+'.conv_weights'] = \
                th.Tensor(d[i]['weights'][0]).permute(3,2,0,1)                
        if not opt.convWeightSharing:
            if input_channels == 1:
                state_dict['resRBF.'+str(i)+'.convt_weights'] = \
                    th.Tensor(d[i]['weights'][1]).permute(2,0,1).unsqueeze(1)
            else:
                state_dict['resRBF.'+str(i)+'.convt_weights'] = \
                    th.Tensor(d[i]['weights'][1]).permute(3,2,0,1)        
        
        state_dict['resRBF.'+str(i)+'.scale_f'] = th.Tensor(d[i]['weights'][2]).log()
        if opt.scale_t:
            state_dict['resRBF.'+str(i)+'.scale_t'] = th.Tensor(d[i]['weights'][3]).log()
        
        if opt.alpha:
            state_dict['resRBF.'+str(i)+'.alpha_prox'] = th.Tensor((d[i]['weights'][5],))
        
        state_dict['resRBF.'+str(i)+'.rbf_weights'] = th.Tensor(d[i]['weights'][4]) 
    
    model.load_state_dict(state_dict)
    
    if loadParams:
        return model, params 
    else: 
        return model


def UDNet_denoise(y,stdn,matlab_model = False):
    r"""If matlab_model is set to True then the model trained on MatConvnet
    is loaded instead of the one trained on pytorch. By default the pytorch 
    trained model is used."""
    import os.path
    
    assert(isinstance(stdn,(float,int))),"The second argument must be an int or"\
    + " a float."
    
    stdn = th.Tensor([stdn]).type_as(y)
    
    ext = ".mat" if matlab_model else ".md"
    
    while y.dim() < 4:
        y = y.unsqueeze(0)
    
    currentPath = os.path.dirname(os.path.realpath(__file__))
    mpath = os.path.join(currentPath,'models','UDNet_')
    
    batch,channels,H,W = y.shape
    
    if channels == 1:
        if stdn < 30:
            mpath += "LGJS5"
        else:
            mpath += "HGJS5"
    elif channels == 3:
        if stdn < 30:
            mpath += "LCJS5"
        else:
            mpath += "HCJS5"
    else: 
        raise ValueError("Input tensor must have either one or three channels.")
    
    mpath += ext
    model = loadModelfromMatlab(mpath) if matlab_model else loadModel(mpath)
    if y.is_cuda:
        model = model.cuda()
    
    with th.no_grad(): out = model(y,stdn)
    
    return out       
    