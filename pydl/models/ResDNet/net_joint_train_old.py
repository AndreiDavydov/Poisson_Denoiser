#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:32:58 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import argparse
from .net import ResDNet_old as ResDNet
from ...datasets.BSDS import BSDS
from pydl.utils import formatInput2Tuple

import os.path
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from math import log10
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


parser = argparse.ArgumentParser(description='Joint-Training of ResDNet')
# Network parameters
parser.add_argument('--kernel_size', type = tupleOfInts, default = '(5,5)', help="The spatial support of the filters in the network.")
parser.add_argument('--color', action='store_true', help="Type of images used to train the network.")
parser.add_argument('--num_filters', type = int, default = 32, help="Number of filters used in the convolution layer.")
parser.add_argument('--convWeightSharing', action='store_true',help="use shared weights for the convolution layers?")
parser.add_argument('--pad', type = tupleOfIntsorString, default = 'same', help="amount of padding of the input")
parser.add_argument('--padType', type = str, default = 'symmetric', help="The type of padding used before convolutions.")
parser.add_argument('--conv_init', type = str, default = 'dct', help='type of initialization for the convolutional layers.')
parser.add_argument('--scale_f', action = 'store_true', help="use scaling for the convolution weights?")
parser.add_argument('--scale_t', action = 'store_true', help="use scaling for the transpose convolution weights?")
parser.add_argument('--normalizedWeights', action = 'store_true',help="use weightNormalization?")
parser.add_argument('--zeroMeanWeights', action = 'store_true',help="use zero-mean normalization?")
parser.add_argument('--alpha', action = 'store_true', help="learn a scaling for the projection threshold?")
parser.add_argument('--rpa_depth', type = int, default = 5, help="Number of residual preactivation layers.")
parser.add_argument('--rpa_kernel_size1', type = tupleOfInts, default = '(3,3)', help="The spatial support of the first convolution layer in the RPA layer.")
parser.add_argument('--rpa_kernel_size2', type = tupleOfInts, default = '(3,3)', help="The spatial support of the second convolution layer in the RPA layer.")
parser.add_argument('--rpa_output_features', type = int, default = 64, help="Number of features extracted from the first convolution layer in the RPA layer.")
parser.add_argument('--rpa_init', type = str, default = 'msra', help='type of initialization for the convolutional layers in the RPA layer.')
parser.add_argument('--rpa_bias1', action = 'store_true', help="use bias for the first convolution layer in the RPA layer?")
parser.add_argument('--rpa_bias2', action = 'store_true', help="use bias for the second convolution layer in the RPA layer?")
parser.add_argument('--numparams_prelu1', type = int, default = 1, help="Number of parameters for the first prelu layer in RPA.")
parser.add_argument('--numparams_prelu2', type = int, default = 1, help="Number of parameters for the second prelu layer in RPA.")
parser.add_argument('--prelu_init', type = float, default = 0.1, help="Value to initialize the prelu parameters.")
parser.add_argument('--rpa_scale1', action = 'store_true', help="use scaling for the first convolution layer in the RPA layer?")
parser.add_argument('--rpa_scale2', action = 'store_true', help="use scaling for the second convolution layer in the RPA layer?")
parser.add_argument('--rpa_normalizedWeights', action = 'store_true',help="use weightNormalization in the RPA layer?")
parser.add_argument('--rpa_zeroMeanWeights', action = 'store_true',help="use zero-mean normalization in the RPA layer?")
parser.add_argument('--shortcut', type = tupleOfBools, default = '(False,True)', help="Indicates whether a shortcut is used in the corresponding RPA layer.")
parser.add_argument('--clb', type = int, default = 0, help="The minimum valid intensity value of the output of the network.")
parser.add_argument('--cub', type = int, default = 255, help="The maximum valid intensity value of the output of the network.")
# Training parameters
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
parser.add_argument('--stdn', type = tupleOfFloats, default='(5,9,13,17,21,25,29)', help=" Number of noise levels (standard deviation) for which the network will be trained.")
parser.add_argument('--saveFreq', type = int, default = 10, help='Every how many epochs we save the model parameters.')
parser.add_argument('--saveBest', action='store_true', help='save the best model parameters?')
parser.add_argument('--xid', type = str, default = '', help='Identifier for the current experiment')
parser.add_argument('--resume', action='store_true', help='resume training?')
parser.add_argument('--initModelPath', type = str, default = '', help='Initialize the model paramaters from a saved state.')
# DataSet Parameters
parser.add_argument('--imdbPath', type = str, default = '', help='location of the dataset.')
parser.add_argument('--data_seed', type = int, default = 20180102, help='random seed for data generation. Default=20180102')

opt = parser.parse_args()

print('========= Selected training parameters and model architecture =============')
print(opt)
print('===========================================================================')
print('\n')

if isinstance(opt.lr_milestones,tuple): 
    opt.lr_milestones = list(opt.lr_milestones)
else:
    opt.lr_milestones = [opt.lr_milestones]
if opt.convWeightSharing:
    str_ws = '-WS'
else:
    str_ws = '-NoWS'

if opt.color:
    strc = 'color'
else:
    strc ='gray'

opt.kernel_size = formatInput2Tuple(opt.kernel_size,int,2)
opt.rpa_kernel_size1 = formatInput2Tuple(opt.rpa_kernel_size1,int,2)
opt.rpa_kernel_size2 = formatInput2Tuple(opt.rpa_kernel_size2,int,2)

if isinstance(opt.stdn,tuple) and len(opt.stdn) == 1:
    stdn = opt.stdn[0]
else:
    stdn = opt.stdn
    
if opt.xid == '':
    opt.xid = 'ResDNet'
else:
    opt.xid = 'ResDNet_'+opt.xid

dirname = "{}_{}_depth:{}_kernel:{}x{}_filters:{}_rpa1:{}x{}_features:{}_rpa2:{}x{}{}_stdn:{}_joint_train".format(opt.xid,\
                 strc,opt.rpa_depth,opt.kernel_size[0],opt.kernel_size[1],\
                 opt.num_filters,opt.rpa_kernel_size1[0],opt.rpa_kernel_size1[1],\
                 opt.rpa_output_features,opt.rpa_kernel_size2[0],opt.rpa_kernel_size2[1],\
                 str_ws,stdn)

currentPath = os.path.dirname(os.path.realpath(__file__))
#currentPath = os.path.dirname(os.path.realpath('pydl/networks/UDNet/net_joint_train.py'))
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
train_set = BSDS(opt.stdn,random_seed=opt.data_seed,filepath=opt.imdbPath,train=True,color=opt.color,shape=(180,180),im2Tensor=True)
test_set = BSDS(opt.stdn,random_seed=opt.data_seed,filepath=opt.imdbPath,train=False,color=opt.color,shape=(180,180),im2Tensor=True)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


print('===> Building model')

# Parameters that we need to specify in order to initialize our model
params = OrderedDict(kernel_size=opt.kernel_size,input_channels=input_channels,\
         output_features=output_features,convWeightSharing=opt.convWeightSharing,\
         pad=opt.pad,padType=opt.padType,conv_init=opt.conv_init,\
         scale_f=opt.scale_f,scale_t=opt.scale_t,normalizedWeights=\
         opt.normalizedWeights,zeroMeanWeights=opt.zeroMeanWeights,\
         alpha=opt.alpha,rpa_depth=opt.rpa_depth,rpa_kernel_size1=\
         opt.rpa_kernel_size1,rpa_kernel_size2=opt.rpa_kernel_size2,\
         rpa_output_features=opt.rpa_output_features,rpa_init=opt.rpa_init,\
         rpa_bias1=opt.rpa_bias1,rpa_bias2=opt.rpa_bias2,numparams_prelu1=\
         opt.numparams_prelu1,numparams_prelu2=opt.numparams_prelu2,\
         prelu_init=opt.prelu_init,rpa_scale1=opt.rpa_scale1,rpa_scale2=\
         opt.rpa_scale2,rpa_normalizedWeights=opt.rpa_normalizedWeights,\
         rpa_zeroMeanWeights=opt.rpa_zeroMeanWeights,shortcut=opt.shortcut,\
         clb=opt.clb,cub=opt.cub)

model = ResDNet(*params.values())

if opt.initModelPath != '':
    state = th.load(opt.initModelPath,map_location = lambda storage, loc:storage)
    model.load_state_dict(state['model_state_dict'])
    opt.resume = False

criterion = nn.MSELoss(size_average=True,reduce=True)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-04)

if opt.cuda :
    model = model.cuda()
    criterion = criterion.cuda()


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
            state = th.load(loadPath)
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
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, sigma = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            sigma = sigma.cuda()

        optimizer.zero_grad()
        loss = criterion(model(input,sigma), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> train:: Epoch[{}]({}/{}): PSNR: {:.4f} dB".format(epoch, iteration, len(training_data_loader),10*log10(opt.cub**2/loss.item())))

    print("===> train:: Epoch[{}] Complete: Avg. PSNR: {:.4f} dB".format(epoch, 10*log10(opt.cub**2/(epoch_loss / len(training_data_loader)))))
    return epoch_loss


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target, sigma = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            sigma = sigma.cuda()

        prediction = model(input,sigma)
        mse = criterion(prediction, target)
        psnr = 10 * log10(opt.cub**2 / mse.item())
        avg_psnr += psnr
    print("===> val:: Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def save_checkpoint(state):    
    if opt.saveBest:
        savePath = os.path.join(dirPath, "model_epoch_best.pth")
    else:
        savePath = os.path.join(dirPath, "model_epoch_{}.pth".format(state['epoch']))
    th.save(state, savePath)
    print("===> Checkpoint saved to {}".format(savePath))


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
            
print("\n ============ Training completed ======================\n")