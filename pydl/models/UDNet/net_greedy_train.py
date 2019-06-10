#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 22:32:58 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import argparse
from pydl.networks.UDNet.net import UDNet
from pydl.datasets.BSDS import BSDS_v2 as BSDS
from pydl.utils import formatInput2Tuple,tic,toc
from pydl.nnLayers.modules import PSNRLoss

import numpy as np
import os.path
import torch as th
import torch.nn as nn
import torch.optim as optim
#from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
#from math import log10
from re import match as pattern_match
from functools import reduce
from collections import OrderedDict

prod = lambda z: reduce(lambda x,y: x*y,z)

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


parser = argparse.ArgumentParser(description='Joint-Training of UDNet')
# Network parameters
parser.add_argument('--kernel_size', type = tupleOfInts, default = '(5,5)', help="The spatial support of the filters in the network.")
parser.add_argument('--color', action='store_true', help="Type of images used to train the network.")
parser.add_argument('--num_filters', type = int, default = 74, help="Number of filters used in the convolution layer.")
parser.add_argument('--rbf_mixtures', type = int, default = 51, help="Number of RBF mixtures.")
parser.add_argument('--rbf_precision', type = int, default = 4, help="The precision for the RBF mixtures.")
parser.add_argument('--stages', type = int, default = 5, help="How many stages the network will consist of.")
parser.add_argument('--pad', type = tupleOfIntsorString, default = 'same', help="amount of padding of the input")
parser.add_argument('--padType', type = str, default = 'symmetric', help="The type of padding used before convolutions.")
parser.add_argument('--convWeightSharing', action='store_true',help="use shared weights for the convolution layers?")
parser.add_argument('--scale_f', action = 'store_true', help="use scaling for the convolution weights?")
parser.add_argument('--scale_t', action = 'store_true', help="use scaling for the transpose convolution weights?")
parser.add_argument('--normalizedWeights', action = 'store_true',help="use weightNormalization?")
parser.add_argument('--zeroMeanWeights', action = 'store_true',help="use zero-mean normalization?")
parser.add_argument('--rbf_start', type = int, default = -100, help="The lower bound of the interval where the RBF centers will be placed.")
parser.add_argument('--rbf_end', type = int, default = 100, help="The upper bound of the interval where the RBF centers will be placed.")
parser.add_argument('--data_min', type = int, default = -100, help="The minimum value of the data used to create the LUT for the computation of the RBF mixture response.")
parser.add_argument('--data_max', type = int, default = 100, help="The maximum value of the data used to create the LUT for the computation of the RBF mixture response.")
parser.add_argument('--data_step', type = float, default = 0.1, help="The step_size to be used for sampling uniformly the data in the range [data_min, data_max].")
parser.add_argument('--alpha', action = 'store_true', help="learn a scaling for the projection threshold?")
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
# DataSet Parameters
parser.add_argument('--imdbPath', type = str, default = '', help='location of the dataset.')
parser.add_argument('--data_seed', type = int, default = 20180102, help='random seed for data generation. Default=20180102')

opt = parser.parse_args()

print('========= Selected training parameters and model architecture =============')
print(opt)
print('===========================================================================')
print('\n')


if opt.stages == 1:
    raise Exception("Use the net_joint_train script for training an 1-stage network.")

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

if isinstance(opt.stdn,tuple) and len(opt.stdn) == 1:
    stdn = opt.stdn[0]
else:
    stdn = opt.stdn
    
if opt.xid == '':
    opt.xid = 'UDNet'
else:
    opt.xid = 'UDNet_'+opt.xid

dirname = "{}_{}_stages:{}_kernel:{}x{}_filters:{}{}_stdn:{}_greedy_train".format(\
                 opt.xid,strc,opt.stages,opt.kernel_size[0],opt.kernel_size[1],\
                 opt.num_filters,str_ws,stdn)

currentPath = os.path.dirname(os.path.realpath(__file__))
#currentPath = os.path.dirname(os.path.realpath('pydl/networks/UDNet/net_greedy_train.py'))
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

Ntrain, Nchannels, H, W  = train_set.train_gt.shape # dimensions of train_gt tensor
Ntest = len(test_set.test_gt) # Number of unique images used in the test_set
NS = len(train_set.stdn) # Number of different noise levels used for training

print('===> Building model')


Lmodel = nn.ModuleList()

params = OrderedDict(kernel_size=opt.kernel_size,input_channels=input_channels,\
         output_features=output_features,rbf_mixtures=opt.rbf_mixtures,\
         rbf_precision=opt.rbf_precision,stages=1,pad=opt.pad,\
         padType=opt.padType,convWeightSharing=opt.convWeightSharing,\
         scale_f=opt.scale_f,scale_t=opt.scale_t,normalizedWeights=\
         opt.normalizedWeights,zeroMeanWeights=opt.zeroMeanWeights,rbf_start=\
         opt.rbf_start,rbf_end=opt.rbf_end,data_min=opt.data_min,data_max=\
         opt.data_max,data_step=opt.data_step,alpha=opt.alpha,clb=opt.clb,\
         cub=opt.cub)

for i in range(opt.stages):
    Lmodel.append(UDNet(*params.values()))

tic()    
for stage in range(opt.stages):
    training_data_loader = DataLoader(dataset=train_set,num_workers=opt.threads,\
                                batch_size=opt.batchSize,shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set,num_workers=opt.threads,\
                                batch_size=opt.testBatchSize,shuffle=False)
    
    #criterion = nn.MSELoss(size_average=True,reduce=True)
    criterion = PSNRLoss(peakval = opt.cub)
    stagePath = os.path.join(dirPath,'stage'+str(stage+1))
    os.makedirs(stagePath,exist_ok = True)
    smodel = Lmodel[stage]
    optimizer = optim.Adam(smodel.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-04)

    if opt.cuda :
        smodel = smodel.cuda()
        criterion = criterion.cuda()

    start = 0
    if opt.resume :
        if opt.saveBest :
            loadPath = os.path.join(stagePath, "model_epoch_best.pth")
            loadPath = loadPath if os.path.isfile(loadPath) else ''
            if loadPath != '':
                msg = "{} :: Resuming by loading the best model.\n".format(dirname)
                print('===>'+msg)
            elif findLastCheckpoint(stagePath) > 0 :
                start = findLastCheckpoint(stagePath)
                msg = "{} :: Resuming by loading epoch {}\n".format(dirname,start)
                print('===>'+msg)
                loadPath = os.path.join(stagePath, "model_epoch_{}.pth".format(start))            
        elif findLastCheckpoint(stagePath) > 0 :
            start = findLastCheckpoint(stagePath)
            msg = "{} :: Resuming by loading epoch {}\n".format(dirname,start)
            print('===>'+msg)
            loadPath = os.path.join(stagePath, "model_epoch_{}.pth".format(start))
        else:
            loadPath = ''
        
        if loadPath == '':
            print('===> No saved model to resume from.\n')
        else:
            if opt.cuda:
                state = th.load(loadPath,map_location=lambda storage,loc:storage.cuda(opt.gpu_device))
            else:
                state = th.load(loadPath,map_location=lambda storage,loc:storage)
    
            smodel.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])    
            th.set_rng_state(state['rng_state'])
            if start == 0 : start = state['epoch'] 

    scheduler = MultiStepLR(optimizer,opt.lr_milestones,gamma=opt.lr_gamma)

    def train(epoch):
        scheduler.step()
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            stage_input, target, sigma, net_input = batch[0], batch[1], batch[2], batch[3]
            if opt.cuda:
                stage_input = stage_input.cuda()
                net_input = net_input.cuda()
                target = target.cuda()
                sigma = sigma.cuda()

            optimizer.zero_grad()
            loss = criterion(smodel(stage_input,sigma,net_input), target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

#            print("===> train:: Stage[{}]: Epoch[{}]({}/{}): PSNR: {:.4f} dB"\
#                  .format(stage+1,epoch,iteration, len(training_data_loader),10*log10(opt.cub**2/loss.item())))
            print("===> train:: Stage[{}]: Epoch[{}]({}/{}): PSNR: {:.4f} dB"\
                  .format(stage+1,epoch,iteration, len(training_data_loader),-loss.item()/len(stage_input)))

#        print("===> train:: Stage[{}]: Epoch[{}] Complete: Avg. PSNR: {:.4f} dB"\
#              .format(stage+1,epoch,10*log10(opt.cub**2/(epoch_loss / len(training_data_loader)))))
        print("===> train:: Stage[{}]: Epoch[{}] Complete: Avg. PSNR: {:.4f} dB"\
              .format(stage+1,epoch,-epoch_loss/(Ntrain*NS)))            
        return epoch_loss


    def test():
        avg_psnr = 0
        for batch in testing_data_loader:
            stage_input, target, sigma, net_input = batch[0], batch[1], batch[2], batch[3]
            if opt.cuda:
                stage_input = stage_input.cuda()
                net_input = net_input.cuda()                    
                target = target.cuda()
                sigma = sigma.cuda()
            
            with th.no_grad(): prediction = smodel(stage_input,sigma,net_input)
#            mse = criterion(prediction, target)
#            psnr = 10 * log10(opt.cub**2 / mse.item())
            psnr = criterion(prediction,target)
            avg_psnr += psnr.item()
            
#        print("===> val:: Stage[{}]: Avg. PSNR: {:.4f} dB".format(stage+1,\
#                                          avg_psnr/len(testing_data_loader)))           
        print("===> val:: Stage[{}]: Avg. PSNR: {:.4f} dB".format(stage+1,\
                                          -avg_psnr/(Ntest*NS)))                   

    def save_checkpoint(state):    
        if opt.saveBest:
            savePath = os.path.join(stagePath, "model_epoch_best.pth")
        else:
            savePath = os.path.join(stagePath, "model_epoch_{}.pth".format(state['epoch']))
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
                state = {'epoch':epoch, 'model_state_dict':smodel.state_dict(),\
                         'optimizer_state_dict':optimizer.state_dict(),\
                         'rng_state':th.get_rng_state(),'params':params}
                save_checkpoint(state)
            elif not opt.saveBest:
                state = {'epoch':epoch, 'model_state_dict':smodel.state_dict(),\
                         'optimizer_state_dict':optimizer.state_dict(),\
                         'rng_state':th.get_rng_state(),'params':params}
                save_checkpoint(state)
        print("******************************************************")
    print("\n ============ Training of stage {} completed in {:.4f} seconds ======================\n".format(stage+1,toc()))
    
    # After the end of training for a particular stage we need to create a new 
    # train_set and test_set that corresponds to the output of this stage. This 
    # new data set will serve as the input to the next stage of the network.
    
    # First we need to create a DataLoader for all the data from both the 
    # train_set and the test_set so that we can compute the output of the 
    # next stage.
    
    if stage+1 < opt.stages:
        full_data = train_set.train_data
        test_data = test_set.test_data
        full_data = np.concatenate((full_data.reshape((NS,\
            full_data.shape[0]//NS)+full_data.shape[1:]),\
            test_data.reshape((NS,test_data.shape[0]//NS)+\
                              test_data.shape[1:])),axis=1)
        del test_data
        full_data = full_data.reshape((prod(full_data.shape[0:2]),)+full_data.shape[2:])
                
        full_obs = train_set.train_obs
        test_obs = test_set.test_obs
        full_obs = np.concatenate((full_obs.reshape((NS,\
            full_obs.shape[0]//NS)+full_obs.shape[1:]),\
            test_obs.reshape((NS,test_obs.shape[0]//NS)+\
                              test_obs.shape[1:])),axis=1)
        del test_obs
        full_obs = full_obs.reshape((prod(full_obs.shape[0:2]),)+full_obs.shape[2:])        
        
        full_gt = np.concatenate((train_set.train_gt,test_set.test_gt),axis = 0)
        
#        if stage == 0:
#            net_input_train = train_set.train_data
#            net_input_test = test_set.test_data
    
        #Instead of creating a new BSDS dataset we use the train_set dataset
        # and change the neccessary class members
        train_set.train_data = full_data
        train_set.train_gt = full_gt
        train_set.train_obs = full_obs
    
        full_data_loader = DataLoader(dataset=train_set,num_workers=opt.threads,\
                                  batch_size = opt.testBatchSize,shuffle=False)    
        
        # We don't apply the box projection on the output of the intermediate
        # stages of the network
        smodel.bbProj.min_val = float('-inf')
        smodel.bbProj.max_val = float('+inf')
        output = np.ndarray(0)
        for batch in full_data_loader:
            stage_input, sigma, net_input = batch[0], batch[2], batch[3]
            if opt.cuda:
                stage_input = stage_input.cuda()
                net_input = net_input.cuda()
                sigma = sigma.cuda()
            
            with th.no_grad(): out = smodel(stage_input,sigma,net_input).cpu().numpy()
            output = np.concatenate((output,out),axis=0) if output.size else out
            
        
	    # Now that we have computed the output of the stage for all images in
        # both the train_set and the test_set we can create the new train_set
        # and test_set, respectively, that will be used to feed the next stage
        # of the network.
        output = output.reshape((NS,output.shape[0]//NS)+output.shape[1:])
        
        full_obs = full_obs.reshape((NS,full_obs.shape[0]//NS)+full_obs.shape[1:])
        
        train_set.train_data = output[:,0:Ntrain,...].reshape((NS*Ntrain,Nchannels,H,W))
        train_set.train_gt = full_gt[0:Ntrain,...]      
        train_set.train_obs = full_obs[:,0:Ntrain,...].reshape((NS*Ntrain,Nchannels,H,W))
        
        test_set.test_data = output[:,Ntrain:,...].reshape((NS*Ntest,Nchannels,H,W))        
        test_set.test_gt = full_gt[Ntrain:,...]
        test_set.test_obs = full_obs[:,Ntrain:,...].reshape((NS*Ntest,Nchannels,H,W))
        
    

    smodel.cpu()
print("\n ============ Training completed ======================\n")


def copyModelParams(model,listModel):
    assert(model.stages == len(listModel)),"stages mismatch between the model "\
    +" and the listModel."
    mkeys = list(model.state_dict().keys())
    
    for stage in range(model.stages):
        skeys = [key for key in mkeys if key.find('.'+str(stage)+'.') != -1]
        lkeys = list(listModel[stage].state_dict().keys())
        for i,j in zip(skeys,lkeys):
            model.state_dict()[i].copy_(listModel[stage].state_dict()[j])

# Create a model of opt.stages that will constist of all the independently 
# trained stages
params['stages'] = opt.stages            
model = UDNet(*params.values())

# Copy the parameters of each 1-stage network to the correct stage of the newly
# created model.
copyModelParams(model,Lmodel)
    
# Save the final model 
state = {'params':params,\
         'model_state_dict':model.state_dict(),\
         'rng_state':th.get_rng_state()}
savePath = os.path.join(dirPath, "model_final.pth")
th.save(state, savePath)
print("===> Final model saved to {}".format(savePath))



    
