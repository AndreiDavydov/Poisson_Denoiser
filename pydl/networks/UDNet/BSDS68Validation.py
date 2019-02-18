#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:13:43 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""
import argparse
import os.path
import torch as th
from pydl.networks.UDNet.net import UDNet_denoise
from pydl.utils import psnr
from pydl.datasets.BSDS import BSDS68
from torch.utils.data import DataLoader

def tupleOfData(s,dtype):   
    if s.find('(',0,1) > -1: # If the first character of the string is '(' then
        # this is a tuple and we keep only the substring with the values 
        # separated by commas, i.e., s[1:-1]. Then we create a list that holds
        # the characters which corresponds to the entries of the tuple by using
        # s[1:-1].split(',')
        s = tuple(dtype(i) for i in s[1:-1].replace(" ","").split(',') if i!="")
    else:
        s = dtype(s)
    return s

tupleOfInts = lambda s: tupleOfData(s,int)
tupleOfFloats = lambda s: tupleOfData(s,float)

parser = argparse.ArgumentParser(description='Validation of ResDNet in BSDS68')
parser.add_argument('--stdn', type = tupleOfFloats, default='(5,10,15,20,25,30,35,40,45,50,55)', help=" Number of noise levels (standard deviation) for which the network will be validated.")
parser.add_argument('--color', action = 'store_true', help="Type of images used to validate the network.")
parser.add_argument('--seed', type = int, default = 20151909, help='random seed to use for generating the noisy images.')
parser.add_argument('--batchSize', type = int, default = 64, help='validation batch size.')
parser.add_argument('--threads', type = int, default = 4, help='number of threads for data loader to use.')
parser.add_argument('--cuda', action = 'store_true', help='use cuda?')
parser.add_argument('--gpu_device', type = int, default = 0, help='which gpu to use?')
parser.add_argument('--matlab_model', action = 'store_true', help='use the matlab trained models?')

opt = parser.parse_args()

print('========= Selected validation parameters =============')
print(opt)
print('======================================================')
print('\n')

if opt.cuda:
    if opt.gpu_device != th.cuda.current_device()\
        and (opt.gpu_device >= 0 and opt.gpu_device < th.cuda.device_count()):
        print("===> Setting GPU device {}".format(opt.gpu_device))
        th.cuda.set_device(opt.gpu_device)

results = dict.fromkeys(opt.stdn)

for stdn in opt.stdn:

    val_tall_set = BSDS68(stdn,random_seed=opt.seed,tall=True,color=opt.color)
    val_wide_set = BSDS68(stdn,random_seed=opt.seed,tall=False,color=opt.color)

    Ntall = len(val_tall_set.img_gt)
    Nwide = len(val_wide_set.img_gt) 

    dataLoader_tall = DataLoader(dataset=val_tall_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    dataLoader_wide = DataLoader(dataset=val_wide_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

    ptable_tall = th.ones(Ntall,2)
    ptable_wide = th.ones(Nwide,2)

    for i, batch in enumerate(dataLoader_tall, 0):
        input, target, sigma = batch[0], batch[1], batch[2]
    
        start = i*opt.batchSize
        end = min((i+1)*opt.batchSize,Ntall)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            sigma = sigma.cuda()
    
        out = UDNet_denoise(input,sigma[0].item(),opt.matlab_model)
        ptable_tall[start:end:1,0]= psnr(input,target)
        ptable_tall[start:end:1,1]= psnr(out,target)
        del out,input,target,sigma

    for i, batch in enumerate(dataLoader_wide, 0):
        input, target, sigma = batch[0], batch[1], batch[2]
    
        start = i*opt.batchSize
        end = min((i+1)*opt.batchSize,Nwide)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            sigma = sigma.cuda()
    
        out = UDNet_denoise(input,sigma[0].item(),opt.matlab_model)
        ptable_wide[start:end:1,0]= psnr(input,target)
        ptable_wide[start:end:1,1]= psnr(out,target)
        del out,input,target,sigma


    ptable = th.cat((ptable_tall,ptable_wide),dim=0)
    del ptable_tall,ptable_wide

    results[stdn] = {"noisy":ptable[:,0],"denoised":ptable[:,1]}

    del ptable

cstr = "color_BSDS68_std:" if opt.color else "gray_BSDS68_std:"
cstr += str(opt.stdn) 

if opt.matlab_model : cstr += "_matlab"
cstr += ".pth"

currentPath = os.path.dirname(os.path.realpath(__file__))
dirPath = os.path.join(currentPath,'Results')
os.makedirs(dirPath,exist_ok = True)
th.save(results,os.path.join(dirPath,cstr))


#results = th.load('pydl/networks/UDNet/Results/color_BSDS68.pth',map_location=lambda storage,loc:storage)
#for std in tuple(results.keys()) :
#    print("==> std: {:>5} ::noisy psnr: {:> 2.2f} -- denoised psnr: {:> 2.2f}".format(std,results[std]['noisy'].mean(),results[std]['denoised'].mean()))
