#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:36:58 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""
import torch as th
import os.path

def copyModelParams(state,stages=3):
    params = state['params']
    
    assert(params['stages'] >= stages),"The number of stages in the initial model "\
    +" are less than the number of stages in the the model to be created."
    
    params['stages'] = stages
        
    state_dict = state['model_state_dict']
    
    mkeys = list(state_dict.keys())
    
    new_state_dict = {}
    for stage in range(stages):
        skeys = [key for key in mkeys if key.find('.'+str(stage)+'.') != -1]        
        for i in skeys:
            new_state_dict[i] = state_dict[i]
    
    state = {'model_state_dict':new_state_dict,'params':params}
    return state


def net_init_from_greedyModel(modelpath,savepath="",stages=3):
    
    state = th.load(modelpath,map_location = lambda storage, loc: storage)
    state = copyModelParams(state,stages)
    
    if len(savepath) == 0:
        savepath = os.path.join(os.path.dirname(modelpath),'model_greedy_s{}.pth'.format(stages))
    
    th.save(state,savepath)
    
    return savepath
