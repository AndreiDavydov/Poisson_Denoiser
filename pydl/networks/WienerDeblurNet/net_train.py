#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:32:58 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import argparse
from pydl.networks.WienerDeblurNet.net import WienerDeblurNet
from pydl.datasets.BSDS import BSDS_deblur
from pydl.utils import formatInput2Tuple,tic,toc,getSubArrays
from pydl.nnLayers.modules import MSELoss

import os.path
import torch as th
from numpy import array as ndarray
#import torch.nn as nn
import torch.optim as optim
#from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
#from math import log10
from re import match as pattern_match
from collections import OrderedDict

def findLastCheckpoint(expDir):
    """Finds the latest checkpoint."""
    lfiles = os.listdir(expDir)
    lfiles = [i for i in lfiles if pattern_match('model_epoch',i)]
    if len(lfiles) == 0 or (len(lfiles) == 1 and lfiles[0].split('epoch_')[-1].split('.')[0] == 'best'):
        return 0
    else:
        lfiles = [lfiles[i].split('epoch_')[-1].split('.')[0] for i in range(len(lfiles))]
        return max(int(i) for i in lfiles if i != 'best')

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

def tupleOfBools(s):   
    if s.find('(',0,1) > -1: # If the first character of the string is '(' then
        # this is a tuple and we keep only the substring with the values 
        # separated by commas, i.e., s[1:-1]. Then we create a list that holds
        # the characters which corresponds to the entries of the tuple by using
        # s[1:-1].split(',')
        s = tuple(i == 'True' for i in s[1:-1].replace(" ","").split(',') if i!="")
    else:
        s = (s == 'True')
    return s

tupleOfInts = lambda s: tupleOfData(s,int)
tupleOfFloats = lambda s: tupleOfData(s,float)

def tupleOfIntsorString(s):   
    if s == "same":
        return s
    elif s.find('(',0,1) > -1: # If the first character of the string is '(' then
        # this is a tuple and we keep only the substring with the values 
        # separated by commas, i.e., s[1:-1]. Then we create a list that holds
        # the characters which corresponds to the entries of the tuple by using
        # s[1:-1].split(',')
        s = tuple(int(i) for i in s[1:-1].split(','))
    else:
        s = int(s)
    return s


parser = argparse.ArgumentParser(description='Image Debluring with WienerDeblurNet')
# Network parameters

# Debluring sub-network parameters
parser.add_argument('--color', action='store_true', help="Type of images used to train the network.")
parser.add_argument('--wiener_kernel_size', type = tupleOfInts, default = '(5,5)', help="The spatial support of the filters used in the Wiener debluring filter.")
parser.add_argument('--wiener_output_features', type = int, default = 24, help="Number of filters used in the Wiener debluring layer.")
parser.add_argument('--numWienerFilters', type = int, default = 4, help="Number of Wiener debluring layers.")
parser.add_argument('--wienerWeightSharing', action='store_true', help="use shared weights for the Wiener debluring layers?")
parser.add_argument('--wienerChannelSharing', action='store_true', help="use shared weights for the different image channels in the Wiener debluring layers?")
parser.add_argument('--alphaChannelSharing', action='store_true',help="use shared alpha weights for the different image channels in the Wiener debluring layers?")
parser.add_argument('--alpha_update', action='store_true',help="Learn the alpha weights for the Wiener debluring layers?")
parser.add_argument('--lb', type = float, default = 1e-5, help="The minimum value of the alpha parameter.")
parser.add_argument('--ub', type = float, default = 1e-2, help="The maximum value of the alpha parameter.")
parser.add_argument('--wiener_pad', action='store_true', help="Pad the input image before debluring?")
parser.add_argument('--wiener_padType', type = str, default = 'symmetric', help="What padding to use?")
parser.add_argument('--edgeTaper', action='store_true', help="Use edge tapering?")
parser.add_argument('--wiener_scale', action = 'store_true', help="use scaling for the Wiener debluring layer weights?")
parser.add_argument('--wiener_normalizedWeights', action = 'store_true',help="use weightNormalization?")
parser.add_argument('--wiener_zeroMeanWeights', action = 'store_true',help="use zero-mean normalization?")
# Denoising sub-network parameters
parser.add_argument('--kernel_size', type = tupleOfInts, default = '(5,5)', help="The spatial support of the filters in the network.")
parser.add_argument('--num_filters', type = int, default = 32, help="Number of filters used in the convolution layer.")
parser.add_argument('--convWeightSharing', action='store_true',help="use shared weights for the convolution layers?")
parser.add_argument('--pad', type = tupleOfIntsorString, default = 'same', help="amount of padding of the input")
parser.add_argument('--padType', type = str, default = 'symmetric', help="The type of padding used before convolutions.")
parser.add_argument('--conv_init', type = str, default = 'dct', help='type of initialization for the convolutional layers.')
parser.add_argument('--bias_f', action = 'store_true', help="use bias for convolution layer of the model?")
parser.add_argument('--bias_t', action = 'store_true', help="use bias for transpose convolution layer of the model?")
parser.add_argument('--scale_f', action = 'store_true', help="use scaling for the convolution weights?")
parser.add_argument('--scale_t', action = 'store_true', help="use scaling for the transpose convolution weights?")
parser.add_argument('--normalizedWeights', action = 'store_true',help="use weightNormalization?")
parser.add_argument('--zeroMeanWeights', action = 'store_true',help="use zero-mean normalization?")
parser.add_argument('--alpha_proj', action = 'store_true', help="learn a scaling for the projection threshold?")
parser.add_argument('--rpa_depth', type = int, default = 5, help="Number of residual preactivation layers.")
parser.add_argument('--rpa_kernel_size1', type = tupleOfInts, default = '(3,3)', help="The spatial support of the first convolution layer in the RPA layer.")
parser.add_argument('--rpa_kernel_size2', type = tupleOfInts, default = '(3,3)', help="The spatial support of the second convolution layer in the RPA layer.")
parser.add_argument('--rpa_output_features', type = int, default = 64, help="Number of features extracted from the first convolution layer in the RPA layer.")
parser.add_argument('--rpa_init', type = str, default = 'msra', help='type of initialization for the convolutional layers in the RPA layer.')
parser.add_argument('--rpa_bias1', action = 'store_true', help="use bias for the first convolution layer in the RPA layer?")
parser.add_argument('--rpa_bias2', action = 'store_true', help="use bias for the second convolution layer in the RPA layer?")
parser.add_argument('--rpa_prelu1_mc', action = 'store_true', help="Use a single or multiple parameters for the first prelu layer in RPA.")
parser.add_argument('--rpa_prelu2_mc', action = 'store_true', help="Use a single or multiple parameters for the second prelu layer in RPA.")
parser.add_argument('--prelu_init', type = float, default = 0.1, help="Value to initialize the prelu parameters.")
parser.add_argument('--rpa_scale1', action = 'store_true', help="use scaling for the first convolution layer in the RPA layer?")
parser.add_argument('--rpa_scale2', action = 'store_true', help="use scaling for the second convolution layer in the RPA layer?")
parser.add_argument('--rpa_normalizedWeights', action = 'store_true',help="use weightNormalization in the RPA layer?")
parser.add_argument('--rpa_zeroMeanWeights', action = 'store_true',help="use zero-mean normalization in the RPA layer?")
parser.add_argument('--shortcut', type = tupleOfBools, default = '(False,True)', help="Indicates whether a shortcut is used in the corresponding RPA layer.")
parser.add_argument('--clb', type = int, default = 0, help="The minimum valid intensity value of the output of the network.")
parser.add_argument('--cub', type = int, default = 255, help="The maximum valid intensity value of the output of the network.")
# Training parameters
parser.add_argument('--msegrad', action = 'store_true', help='Use gradient of MSE?')
parser.add_argument('--batchSize', type = int, default = 64, help='training batch size.')
parser.add_argument('--testBatchSize', type = int, default = 100, help='testing batch size.')
parser.add_argument('--nEpochs', type = int, default = 100, help='number of epochs to train for.')
parser.add_argument('--lr', type = float, default = 1e-3, help='learning rate. Default=1e-3.')
parser.add_argument('--lr_milestones', type = tupleOfInts, default = 100, help="Scheduler's learning rate milestones. Default=[100].")
parser.add_argument('--lr_gamma', type = float, default = 0.1, help="multiplicative factor of learning rate decay.")
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--gpu_device', type = int, default = 0, help='which gpu to use?')
parser.add_argument('--threads', type = int, default = 4, help='number of threads for data loader to use.')
parser.add_argument('--seed', type = int, default = 123, help='random seed to use. Default=123.')
parser.add_argument('--stdn', type = tupleOfFloats, default='(2.5,4.,5.5,7.,8.5,10.)', help=" Number of noise levels (standard deviation) for which the network will be trained.")
parser.add_argument('--saveFreq', type = int, default = 10, help='Every how many epochs we save the model parameters.')
parser.add_argument('--saveBest', action='store_true', help='save the best model parameters?')
parser.add_argument('--xid', type = str, default = '', help='Identifier for the current experiment')
parser.add_argument('--resume', action='store_true', help='resume training?')
parser.add_argument('--initModelPath', type = str, default = '', help='Initialize the model paramaters from a saved state.')
# DataSet Parameters
parser.add_argument('--imdbPath', type = str, default = '', help='location of the dataset.')
parser.add_argument('--psfTrainPath', type = str, default = '', help='location of the psfs used for training.')
parser.add_argument('--psfTestPath', type = str, default = '', help='location of the psfs used for testing.')
parser.add_argument('--numTrainImagesperPSF', type = int, default = 50, help='How many images to use for training with a particular psf?')
parser.add_argument('--numTestImagesperPSF', type = int, default = 50, help='How many images to use for testing with a particular psf?')
parser.add_argument('--data_seed', type = int, default = 20180102, help='random seed for data generation. Default=20180102')
# Optimizer Parameters
parser.add_argument('--amsgrad', action='store_true', help='Use the fix for Adam?')

opt = parser.parse_args()

print('========= Selected training parameters and model architecture =============')
print(opt)
print('===========================================================================')
print('\n')


# from argparse import Namespace
# opt = Namespace(color=True,wiener_kernel_size=(5,5),wiener_output_features=24,\
# numWienerFilters=3,wienerWeightSharing=True,wienerChannelSharing=True,\
# alphaChannelSharing=True,lb=1e-4,ub=1e-2,wiener_pad=True,wiener_padType='symmetric',\
# edgeTaper=True,wiener_scale=True,wiener_normalizedWeights=True,wiener_zeroMeanWeights=True,\
# kernel_size=(5,5),num_filters=32,convWeightSharing=True,pad='same',padType='symmetric',\
# conv_init='dct',bias_f=True,bias_t=True,scale_f=True,scale_t=False,normalizedWeights=True,\
# zeroMeanWeights=True,alpha_proj=True,rpa_depth=3,rpa_kernel_size1=(3,3),rpa_kernel_size2=(3,3),\
# rpa_output_features=64,rpa_init='msra',rpa_bias1=True,rpa_bias2=True,rpa_prelu1_mc=True,\
# rpa_prelu2_mc=True,prelu_init=0.1,rpa_scale1=True,rpa_scale2=True,rpa_normalizedWeights=True,\
# rpa_zeroMeanWeights=True,shortcut=(True,False,False),clb=0,cub=255,msegrad=True,batchSize=10,\
# testBatchSize=6,nEpochs=10,lr=1e-2,lr_gamma=0.1,lr_milestones=100,cuda=False,gpu_device=0,\
# threads=4,seed=123,stdn=(2.5,4,5.5),saveFreq=5,saveBest=True,xid='',resume=True,\
# initModelPath='',imdbPath='/home/stamatis/Documents/Work/repos/datasets/imdb_256x256_color.npz',\
# psfTrainPath='/home/stamatis/Documents/Work/repos/datasets/MotionBlurKernels/trainKernels20.pt',\
# psfTestPath='/home/stamatis/Documents/Work/repos/datasets/MotionBlurKernels/testKernels.pt',\
# numTrainImagesperPSF=4,numTestImagesperPSF=3,data_seed=20180102,amsgrad=True)



if isinstance(opt.lr_milestones,tuple): 
    opt.lr_milestones = list(opt.lr_milestones)
else:
    opt.lr_milestones = [opt.lr_milestones]

str_cws = '-CWS' if opt.convWeightSharing else ''
str_wws = '-WWS' if opt.wienerWeightSharing else ''
str_wcs = '-WCS' if opt.wienerChannelSharing and opt.color else ''
strc = 'color' if opt.color else 'gray'


opt.wiener_kernel_size = formatInput2Tuple(opt.wiener_kernel_size,int,2)
opt.kernel_size = formatInput2Tuple(opt.kernel_size,int,2)
opt.rpa_kernel_size1 = formatInput2Tuple(opt.rpa_kernel_size1,int,2)
opt.rpa_kernel_size2 = formatInput2Tuple(opt.rpa_kernel_size2,int,2)

if isinstance(opt.stdn,tuple) :
    if len(opt.stdn) == 1:
        stdn = str(opt.stdn[0])
    else:
        stdn = "("+str(opt.stdn[0])+"->"+str(opt.stdn[-1])+")"
else:
    stdn = str(opt.stdn)
    
if opt.xid == '':
    opt.xid = 'WDNet'
else:
    opt.xid = 'WDNet_'+opt.xid

dirname = "{}_{}_wkernel:{}x{}_wfilters:{}_width:{}{}{}_kernel:{}x{}_filters:{}_depth:{}_rpa1:{}x{}_features:{}_rpa2:{}x{}{}_train".format(opt.xid,\
               strc,opt.wiener_kernel_size[0],opt.wiener_kernel_size[1],opt.wiener_output_features,
               opt.numWienerFilters,str_wws,str_wcs,opt.kernel_size[0],opt.kernel_size[1],\
               opt.num_filters,opt.rpa_depth,opt.rpa_kernel_size1[0],opt.rpa_kernel_size1[1],\
               opt.rpa_output_features,opt.rpa_kernel_size2[0],opt.rpa_kernel_size2[1],\
               str_cws)

#currentPath = os.path.dirname(os.path.realpath(__file__))
currentPath = os.path.dirname(os.path.realpath('pydl/networks/WienerDeblurNet/net_train.py'))
dirPath = os.path.join(currentPath,'Results',dirname)
os.makedirs(dirPath,exist_ok = True)

# save the input arguments
th.save(opt,os.path.join(dirPath,"args.pth"))

input_channels = 3 if opt.color else 1
output_features = opt.num_filters

if opt.cuda and not th.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

th.manual_seed(opt.seed)
if opt.cuda:
    if opt.gpu_device != th.cuda.current_device()\
        and (opt.gpu_device >= 0 and opt.gpu_device < th.cuda.device_count()):
        print("===> Setting GPU device {}".format(opt.gpu_device))
        th.cuda.set_device(opt.gpu_device)
        
    th.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

# Load the training and testing kernels
Ktrain = th.load(opt.psfTrainPath)
Ktest = th.load(opt.psfTestPath)

Ntrain, Ntest  = 400, 100

assert(opt.numTrainImagesperPSF <= Ntrain and opt.numTestImagesperPSF <= Ntest),\
 "Invalid values for one or both of numTrainImagesperPSF and numTestImagesperPSF."
train_mask = getSubArrays(0,Ntrain,len(Ktrain),length=opt.numTrainImagesperPSF,dformat=lambda x:ndarray(x))
test_mask = getSubArrays(0,Ntest,len(Ktest),length=opt.numTestImagesperPSF,dformat=lambda x:ndarray(x))  
NS = len(opt.stdn)
train_data_loader = {}
test_data_loader = {}
for k in range(len(Ktrain)):
    train_set = BSDS_deblur(Ktrain[k],opt.stdn,random_seed=opt.data_seed,filepath=opt.imdbPath,train=True,color=opt.color,shape=(256,256),batchSize=50,mask=train_mask[k])
    train_data_loader[k] = DataLoader(dataset = train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
for k in range(len(Ktest)):
    test_set = BSDS_deblur(Ktest[k],opt.stdn,random_seed=opt.data_seed,filepath=opt.imdbPath,train=False,color=opt.color,shape=(256,256),batchSize=50,mask=test_mask[k])
    test_data_loader[k] = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
    
print('===> Building model')

# Parameters that we need to specify in order to initialize our model
params = OrderedDict(input_channels=input_channels,wiener_kernel_size=opt.wiener_kernel_size,\
         wiener_output_features=opt.wiener_output_features,numWienerFilters=opt.numWienerFilters,\
         wienerWeightSharing=opt.wienerWeightSharing,wienerChannelSharing=\
         opt.wienerChannelSharing,alphaChannelSharing=opt.alphaChannelSharing,\
         alpha_update=opt.alpha_update,lb=opt.lb,ub=opt.ub,wiener_pad=opt.wiener_pad,\
         wiener_padType=opt.wiener_padType,edgeTaper=opt.edgeTaper,\
         wiener_scale=opt.wiener_scale,wiener_normalizedWeights=opt.wiener_normalizedWeights,\
         wiener_zeroMeanWeights=opt.wiener_zeroMeanWeights,kernel_size=opt.kernel_size,\
         output_features=output_features,convWeightSharing=opt.convWeightSharing,\
         pad=opt.pad,padType=opt.padType,conv_init=opt.conv_init,bias_f = \
         opt.bias_f,bias_t = opt.bias_t,scale_f=opt.scale_f,scale_t=opt.scale_t,\
         normalizedWeights=opt.normalizedWeights,zeroMeanWeights=opt.zeroMeanWeights,\
         alpha_proj=opt.alpha_proj,rpa_depth=opt.rpa_depth,rpa_kernel_size1=\
         opt.rpa_kernel_size1,rpa_kernel_size2=opt.rpa_kernel_size2,\
         rpa_output_features=opt.rpa_output_features,rpa_init=opt.rpa_init,\
         rpa_bias1=opt.rpa_bias1,rpa_bias2=opt.rpa_bias2,rpa_prelu1_mc=\
         opt.rpa_prelu1_mc,rpa_prelu2_mc=opt.rpa_prelu2_mc,\
         prelu_init=opt.prelu_init,rpa_scale1=opt.rpa_scale1,rpa_scale2=\
         opt.rpa_scale2,rpa_normalizedWeights=opt.rpa_normalizedWeights,\
         rpa_zeroMeanWeights=opt.rpa_zeroMeanWeights,shortcut=opt.shortcut,\
         clb=opt.clb,cub=opt.cub)

model = WienerDeblurNet(*params.values())

if opt.initModelPath != '':
    state = th.load(opt.initModelPath,map_location = lambda storage, loc:storage)
    model.load_state_dict(state['model_state_dict'])
    opt.resume = False

#criterion = nn.MSELoss(size_average=True,reduce=True)
criterion = MSELoss(grad=opt.msegrad)

if opt.cuda :
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999),\
                       eps=1e-04, amsgrad=opt.amsgrad)

start = 0
if opt.resume :
    if opt.saveBest :
        loadPath = os.path.join(dirPath, "model_epoch_best.pth")
        loadPath = loadPath if os.path.isfile(loadPath) else ''
        if loadPath != '':
            msg = "{} :: Resuming by loading the best model.\n".format(dirname)
            print('===>'+msg)
        elif findLastCheckpoint(dirPath) > 0 :
            start = findLastCheckpoint(dirPath)
            msg = "{} :: Resuming by loading epoch {}\n".format(dirname,start)
            print('===>'+msg)
            loadPath = os.path.join(dirPath, "model_epoch_{}.pth".format(start))            
    elif findLastCheckpoint(dirPath) > 0 :
        start = findLastCheckpoint(dirPath)
        msg = "{} :: Resuming by loading epoch {}\n".format(dirname,start)
        print('===>'+msg)
        loadPath = os.path.join(dirPath, "model_epoch_{}.pth".format(start))
    else:
        loadPath = ''
        
    if loadPath == '':
        print('===> No saved model to resume from.\n')
    else:
        if opt.cuda:
            state = th.load(loadPath,map_location=lambda storage,loc:storage.cuda(opt.gpu_device))
        else:
            state = th.load(loadPath,map_location=lambda storage,loc:storage)
    
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])    
        th.set_rng_state(state['rng_state'])
        if start == 0 : start = state['epoch'] 

scheduler = MultiStepLR(optimizer,opt.lr_milestones,gamma=opt.lr_gamma)

def train(epoch):
    scheduler.step()
    epoch_loss = 0
    rper = th.randperm(len(train_data_loader))
    ctr = 0
    diter = len(train_data_loader[0])
    totalIter = len(train_data_loader)*diter
    for k in rper:
        ctr += 1
        for iteration, batch in enumerate(train_data_loader[k], 1):
            input, target, kernel, sigma = batch[0], batch[1], batch[2][0], batch[3]
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
                kernel = kernel.cuda()
                sigma = sigma.cuda()

            optimizer.zero_grad()
            loss = criterion(model(input,kernel,sigma), target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
    
        print("===> train:: Epoch[{}]({}/{}): PSNR: {:.4f} dB".format(epoch,iteration+(ctr-1)*diter,totalIter,loss.item()))
    
    print("===> train:: Epoch[{}] Complete: Avg. PSNR: {:.4f} dB".format(epoch, epoch_loss/totalIter))        
    return epoch_loss


def test():
    avg_psnr = 0
    totalIter = len(test_data_loader)*len(test_data_loader[0])
    for k in range(len(test_data_loader)):
        for batch in test_data_loader[k]:
            input, target, kernel, sigma = batch[0], batch[1], batch[2][0], batch[3]
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
                kernel = kernel.cuda()
                sigma = sigma.cuda()
    
            with th.no_grad(): prediction = model(input,kernel,sigma)
            psnr = criterion(prediction, target)
            avg_psnr += psnr.item()

    print("===> val:: Avg. PSNR: {:.4f} dB".format(avg_psnr/totalIter))        


def save_checkpoint(state):    
    if opt.saveBest:
        savePath = os.path.join(dirPath, "model_epoch_best.pth")
    else:
        savePath = os.path.join(dirPath, "model_epoch_{}.pth".format(state['epoch']))
    th.save(state, savePath)
    print("===> Checkpoint saved to {}".format(savePath))

tic()
epoch_loss = float('inf')
for epoch in range(start+1, opt.nEpochs + 1):
    if opt.saveBest:
        epoch_loss_new = train(epoch)
    else:
        train(epoch)        
    test()
    if not epoch%opt.saveFreq :
        if opt.saveBest and epoch_loss_new < epoch_loss:
            epoch_loss = epoch_loss_new    
            state = {'epoch':epoch, 'model_state_dict':model.state_dict(),\
                 'optimizer_state_dict':optimizer.state_dict(),\
                 'rng_state':th.get_rng_state(),'params':params}
            save_checkpoint(state)
        elif not opt.saveBest:
            state = {'epoch':epoch, 'model_state_dict':model.state_dict(),\
                 'optimizer_state_dict':optimizer.state_dict(),\
                 'rng_state':th.get_rng_state(),'params':params}
            save_checkpoint(state)
    print("******************************************************")
print("\n ============ Training completed in {:.4f} seconds ======================\n".format(toc()))