from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch as th
import os

STDN = 1

def dt():
    return datetime.now().strftime('%H:%M:%S')

def train(model, model_type, optimizer, criterion, data_loader, save_path):

    assert(len(data_loader) > 0),\
    "train data loader must contain data. Check path to dataset."

    print('\n Training...')
    train_losses = list(np.load(save_path+'train_losses.npy'))
    for gt, noisy in tqdm(data_loader):

        gt = gt.cuda()
        noisy = noisy.cuda()

        if model_type == 'pois':
            prediction = model(noisy, noisy)
        elif model_type == 'l2':
            stdn = th.Tensor([STDN]).cuda()
            prediction = model(noisy, stdn, noisy)

        optimizer.zero_grad()
        loss = criterion(prediction, gt)

        cur_loss = loss.data.cpu().numpy().item()
        train_losses.append(cur_loss)
        loss.backward()
        optimizer.step()

        np.save(save_path+'train_losses.npy', train_losses)


def val(model, model_type, criterion, data_loader, save_path):

    assert(len(data_loader) > 0),\
    "val data loader must contain data. Check path to dataset."

    print('\n Validation...')

    val_losses = list(np.load(save_path+'val_losses.npy'))

    val_loss_per_epoch, num_losses_epoch = 0, 0 # for scheduler step

    for gt, noisy in tqdm(data_loader):

        gt = gt.cuda()
        noisy = noisy.cuda()

        with th.no_grad(): 
            if model_type == 'pois':
                prediction = model(noisy, noisy)
            elif model_type == 'l2':
                stdn = th.Tensor([STDN]).cuda()
                prediction = model(noisy, stdn, noisy)
                
        loss = criterion(prediction, gt)
        cur_loss = loss.data.cpu().numpy().item()

        val_loss_per_epoch += cur_loss
        num_losses_epoch += 1

    cur_loss = val_loss_per_epoch / num_losses_epoch

    val_losses.append(cur_loss) 
        # save val_losses in average for all dataset

    np.save(save_path+'val_losses.npy', val_losses)

    return cur_loss


def save_model(epoch, model, optimizer, scheduler, save_path, save_epoch=10):

    if epoch % save_epoch != 0 and epoch != 1:
        return

    state = {'model_state_dict':model.state_dict(),\
             'optimizer_state_dict':optimizer.state_dict(),\
             'scheduler_state_dict':scheduler.state_dict(),\
             'rng_state':th.get_rng_state(),\
             'rng_states_cuda':th.cuda.get_rng_state_all()}   

    np.save(save_path+'epoch.npy', epoch)

    path2file = save_path+'state_'+str(epoch)+'.pth'
    th.save(state, path2file)

    print("\n ===>", dt(), "epoch={}. ".format(epoch))
    print("Model is saved to {}".format(save_path))


def load_model(epoch, model, optimizer, scheduler, save_path):

    path2file = save_path+'state_'+str(epoch)+'.pth'

    state = th.load(path2file,map_location=lambda storage,loc:storage.cuda())
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])  
    scheduler.load_state_dict(state['scheduler_state_dict'])  
    th.set_rng_state(state['rng_state'].cpu()) 

    return model, optimizer, scheduler

from time import sleep

def training_procedure(start_epoch, num_epochs, model, model_type,\
                        optimizer, scheduler, criterion, \
                        train_loader, val_loader, save_path, save_epoch):

    model_name = save_path[len('./PoisDenoiser/networks/PoisNet/models/'):-1]
    # Procedure of training and validating  
    for epoch in range(start_epoch, num_epochs+1):
        print('model {}'.format(model_name))

        # train
        train(model, model_type, optimizer, criterion, train_loader, save_path)
        print('\n', dt(), 'epoch={}/{}. Train is done.'.format(epoch, num_epochs))  
        # val
        val_loss = val(model, model_type, criterion, val_loader, save_path)

        print('val_loss = {:.6f}'.format(float(val_loss)))
        scheduler.step() #val_loss
        # print('opt lr = {:.2e}'.format(optimizer.param_groups[0]['lr']))

        print('\n', dt(), 'epoch={}/{}. Validation is done.'.format(epoch, num_epochs))  

        # save model
        save_model(epoch, model, optimizer, scheduler, save_path, save_epoch)


def initialize_network(model, optimizer, scheduler, save_path, start_epoch=None):
    '''
    start_epoch will load the model saved at this epoch. 
    '''

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        train_losses = []
        val_losses = []
        np.save(save_path+'train_losses.npy', train_losses)
        np.save(save_path+'val_losses.npy', val_losses)
        start_epoch = 1
    else:
        if start_epoch is None:
            train_losses = []
            val_losses = []
            np.save(save_path+'train_losses.npy', train_losses)
            np.save(save_path+'val_losses.npy', val_losses)
            start_epoch = 1
        else:
            train_losses = list(np.load(save_path+'train_losses.npy'))
            val_losses = list(np.load(save_path+'val_losses.npy'))
            start_epoch = int(np.load(save_path+'epoch.npy'))
            model, optimizer, scheduler = load_model(start_epoch, model, \
                                            optimizer, scheduler, save_path)