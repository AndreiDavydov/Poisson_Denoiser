from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch as th
import os

def dt():
    return datetime.now().strftime('%H:%M:%S')

def train(model, optimizer, criterion, data_loader, save_path):

    print('\n Training...')
    train_losses = list(np.load(save_path+'train_losses.npy'))
    for gt, noisy in tqdm(data_loader):

        gt = gt.cuda()
        noisy = noisy.cuda()

        prediction = model(noisy, noisy)

        # if th.isnan(prediction).sum() > 0:
        #     # savings = {'model_state_dict':model.state_dict(),\
        #     #             'optimizer_state_dict':optimizer.state_dict(),\
        #     #             'noisy':noisy.detach()} 

        #     # path2file = save_path+'state_'+'ERROR.pth'
        #     # th.save(savings, path2file)

        #     return True

        optimizer.zero_grad()
        loss = criterion(prediction, gt)

        cur_loss = loss.data.cpu().numpy().item()
        train_losses.append(cur_loss)
        loss.backward()
        optimizer.step()

        np.save(save_path+'train_losses.npy', train_losses)


def val(model, criterion, data_loader, save_path):


    print('\n Validation...')

    val_losses = list(np.load(save_path+'val_losses.npy'))

    val_loss_per_epoch, num_losses_epoch = 0, 0 # for scheduler step

    for gt, noisy in tqdm(data_loader):

        gt = gt.cuda()
        noisy = noisy.cuda()

        with th.no_grad(): prediction = model(noisy, noisy)
        loss = criterion(prediction, gt)
        cur_loss = loss.data.cpu().numpy().item()
        val_losses.append(cur_loss)

        val_loss_per_epoch += cur_loss
        num_losses_epoch += 1

        np.save(save_path+'val_losses.npy', val_losses)

    return val_loss_per_epoch / num_losses_epoch


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
    # th.cuda.set_rng_state(state['rng_states_cuda'][0]) #.cpu()

    return model, optimizer, scheduler


def training_procedure(start_epoch, num_epochs, model, optimizer, scheduler, \
                        criterion, train_loader, val_loader, save_path, save_epoch):

    # Procedure of training and validating  
    for epoch in range(start_epoch, num_epochs+1):

        # train
        train(model, optimizer, criterion, train_loader, save_path)
        print('\n', dt(), 'epoch={}/{}. Train is done.'.format(epoch, num_epochs))  
        # val
        val_loss = val(model, criterion, val_loader, save_path)

        print('{:.4f}'.format(float(val_loss)))
        # scheduler.step(val_loss)

        print('\n', dt(), 'epoch={}/{}. Validation is done.'.format(epoch, num_epochs))  


        # save model
        save_model(epoch, model, optimizer, scheduler, save_path, save_epoch)

        print('lr == {}'.format(float(optimizer.param_groups[0]['lr'])))


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