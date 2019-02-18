
import numpy as np

import torch as th
from PoisDenoiser.networks.PoisNet.net import PoisNet
from PoisDenoiser.dataset_loader import BSDS500
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pydl.nnLayers.modules import PSNRLoss
from torch.nn import MSELoss

import os

from training_procedure import training_procedure, initialize_network, load_model

th.manual_seed(1234)
th.cuda.manual_seed(1234)

gpu_id=1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

stages=5 ###################################################################
output_features=64 #########################################################
model = PoisNet(stages=stages, output_features=output_features).cuda()
# optimizer = Adam(model.parameters(), lr=1e-2)

optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9)

scheduler = ReduceLROnPlateau(optimizer, 'min', threshold=1e-2, patience=3)
criterion = MSELoss().cuda()

# k = 0.1
# max_val = 1/k
# criterion = PSNRLoss(max_val).cuda()

path2dataset = './DATASETS/BSDS500_Pois_crops/'
experiment_name = 's{}c{}_wtf'.format(stages, output_features)
save_path = './PoisDenoiser/networks/PoisNet/models/'+experiment_name+'/'

start_epoch = None ################################################################

initialize_network(model, optimizer, scheduler, save_path, start_epoch=start_epoch)


np.save(save_path+'PoisProx_output.npy', [])
np.save(save_path+'PoisProx_input.npy', [])
np.save(save_path+'grbf_output.npy', [])
np.save(save_path+'grbf_input.npy', [])


train_batchsize=50 #############################################################
val_batchsize=50   #############################################################

BSDStrain = BSDS500(path2dataset+'train/')
BSDStrain_loader = DataLoader(BSDStrain, batch_size=train_batchsize, \
    shuffle=True, num_workers=0)

BSDSval = BSDS500(path2dataset+'val/')
BSDSval_loader = DataLoader(BSDSval, batch_size=val_batchsize, \
    shuffle=False, num_workers=0)


epochs = 1000 #############################################################
save_epoch = 1 ############################################################

if start_epoch is None:
    start_epoch = 1 

training_procedure(start_epoch, epochs, model, optimizer, scheduler, \
	criterion, BSDStrain_loader, BSDSval_loader, save_path, save_epoch)
        