import numpy as np
import torch as th

from PoisDenoiser.networks.PoisNet.net import PoisNet
from pydl.networks.UDNet.net import UDNet

from PoisDenoiser.dataset_loader import FMD
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from torch.nn import MSELoss, L1Loss

import os
import sys
from PoisDenoiser.training_procedure import training_procedure, \
                                            initialize_network
from PoisDenoiser.arg_parser import parse_args_joint_train as parse_args

import argparse

opt = parse_args()

th.manual_seed(1234)
th.cuda.manual_seed(1234)

gpu_id = opt.gpu_id 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

stages = opt.stages 
output_features = opt.channels 

prox_param = opt.prox_param
sharing_weights = not opt.no_sharing_weights

print(prox_param)
print(sharing_weights)


model_type = opt.model_type
print(model_type)
if model_type == 'pois':
    model = PoisNet(stages=stages, output_features=output_features,\
        prox_param=prox_param, convWeightSharing=sharing_weights).cuda()
elif model_type == 'l2':
    model = UDNet(stages=stages, output_features=output_features, \
        alpha=prox_param, convWeightSharing=sharing_weights).cuda()

lr0 = opt.lr0
optimizer = Adam(model.parameters(), lr=lr0)

milestones = list(map(int, opt.milestones.strip('[]').split(','))) \
            if opt.milestones != '' else []
    
scheduler = MultiStepLR(optimizer, milestones=milestones, \
                                   gamma=0.1)
loss = opt.loss
if loss == 'MSE':
    criterion = MSELoss().cuda()
elif loss == 'pois':
    criterion = poisLLHLoss
elif loss == 'L1':
    criterion = L1Loss().cuda()
elif loss == 'perceptual_MSE':
    criterion_perceptual = PerceptualLoss(use_gpu=True) 
    criterion = criterion_perceptual
    criterion_ordinary = MSELoss().cuda() ##########
    def compute_perceptual_loss_function(x,y):
        loss1 = criterion_perceptual(x,y)/100
        loss2 = criterion_ordinary(x,y)
        return loss1 + loss2

    criterion = compute_perceptual_loss_function

num_images = opt.num_images
experiment_name = 's{}c{}'.format(stages, output_features)

save_path = './PoisDenoiser/networks/PoisNet/models/fmd/'+opt.save_path_app+'/'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

save_path = save_path + experiment_name + '/'


start_epoch =  opt.start_epoch
initialize_network(model, optimizer, scheduler, save_path, \
                start_epoch=start_epoch)

train_batchsize = opt.train_batchsize
val_batchsize = opt.val_batchsize

# path2dataset = './DATASETS/BSDS500/BSDS500_Pois_crops_PEAK_1/'

# do_VST = opt.do_VST

FMDtrain = FMD(exp=opt.exp, trainorval='train', get_name=False)
FMDtrain_loader = DataLoader(FMDtrain, batch_size=train_batchsize, \
    shuffle=True, num_workers=0)

FMDval = FMD(exp=opt.exp, trainorval='train', get_name=False)
FMDval_loader = DataLoader(FMDval, batch_size=val_batchsize, \
    shuffle=False, num_workers=0)

np.save(save_path+'params.npy', {'num_train_imgs':num_images,\
                                 'train_batchsize':train_batchsize,\
                                 'val_batchsize':val_batchsize,\
                                 'lr0':lr0, 'milestones':milestones})

epochs = opt.num_epochs
save_epoch = opt.save_epoch

if start_epoch is None:
    start_epoch = 1 

training_procedure(start_epoch, epochs, model, model_type,\
                    optimizer, scheduler, criterion, \
                    FMDtrain_loader, FMDval_loader, \
                    save_path, save_epoch)
