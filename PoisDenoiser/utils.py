
import torch as th
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from IPython.display import clear_output

from time import sleep

from torch.distributions import Poisson

from PoisDenoiser.networks.PoisNet.net import PoisNet
from pydl.networks.UDNet.net import UDNet

from PoisDenoiser.dataset_loader import BSDS500

from PoisDenoiser.VST import VST_backward_unbiased_exact

filePath = './../Denoising/Diploma/'

def psnr(signal, noise):
    '''
    Computes Peak Signal-to-Noise Ratio.
    
    Assumes that dimensions of inputs coincide.
    '''
    max_val = th.max(signal)
    err = th.mean((signal-noise)**2)
    return 10*th.log10(max_val**2 / err)


def get_poisson_pair(img, k, seed=1):
    '''
    Input image must be in (0,1] range.
    '''
    
    if k == 0:
        return img, img 

    img = th.FloatTensor(img)
    noisy = Poisson(img/k).sample()

    return img/k, noisy


def get_poisson_pair_by_maxval(img, max_val, seed=1):
    '''
    Input image must be in (0,1] range.
    '''
    
    if max_val == 255:
        return img, img 

    img = th.FloatTensor(img)
    noisy = Poisson(img*max_val).sample()

    return img*max_val, noisy


def generate_test_pair(img_type='arange', size=256, k=0.01, seed=1):
    
    if img_type == 'file':

        img = imread(filePath+'DATASETS/BSDS500/BSDS500_256x256/val/ground_truth/78004.jpg').astype(np.float32)
        img = (img[...,0] + img[...,1] + img[...,2])/3

        img[img<1] = 1 # All values in pixels must be more than zero. 
                       # Instead there would be no stochastic process on these values.
        img = img/255
        img = img[:size, :size]

    elif img_type == 'arange':
        img = np.meshgrid(np.arange(1, size+1), np.arange(1, size+1))[0]/size

        img[size//5   :size*2//5, size//5   :size*2//5] = 1
        img[size*3//5 :size*4//5, size*3//5 :size*4//5] = 0

    img, noisy = get_poisson_pair(img=img, k=k, seed=seed)

    return img, noisy


def generate_batch(k=0.1, size=256, seed=1):

    clear1, noisy1 = generate_test_pair(img_type='arange', size=size, k=k, seed=seed)
    clear1.unsqueeze_(0).unsqueeze_(0)
    noisy1.unsqueeze_(0).unsqueeze_(0)

    clear2, noisy2 = generate_test_pair(img_type='file',   size=size, k=k, seed=seed)
    clear2.unsqueeze_(0).unsqueeze_(0)
    noisy2.unsqueeze_(0).unsqueeze_(0)   

    clear_batch = th.cat([clear1, clear2], dim=0)
    noisy_batch = th.cat([noisy1, noisy2], dim=0)

    return clear_batch, noisy_batch


def show_images(images, titles=None, figsize=None, fontsize=None, patches=None):
    '''
    All images are considered to be Tensor and have dimensions of CxHxW or HxW.
    If the number of images to show is more than one, then images is considered to be a list.
    The titles must be either string (1 image for input) or a list of strings.
    '''
    if not isinstance(images, list):
        images = [images]        

    if fontsize is None:
        fontsize = 15

    images_corrected_dims = []
    for i, img in enumerate(images):
        if img.dim() == 4:
            img = img[0]

        img = img[0] if img.size()[0] == 1 else img.permute(1,2,0)
        images_corrected_dims.append(img)

    images = images_corrected_dims

    num_images = len(images)

    if figsize is None:
        figsize = (8+num_images,8*num_images)
        if num_images == 3:
            figsize = (15,5)
        if num_images == 4:
            figsize = (20,5)
        
    fig, ax = plt.subplots(1, num_images, figsize=figsize)
    fig.patch.set_facecolor('white')

    if num_images == 1:
        ax = [ax]

    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].set_axis_off()

    if patches is not None:
        # patches must be a list of patches or one patch
        for i in range(len(ax)):
            if patches[i] == None:
                continue
            ax[i].add_patch(patches[i])

    # Now only titles have to be distributed.
    if titles is not None:
        if isinstance(titles, str): # Title for one image
            ax[0].set_title(titles)

        elif isinstance(titles, list):
            for i, title in enumerate(titles):
                ax[i].set_title(title, fontsize=fontsize)

            if len(titles) < num_images:
                for i in range(len(titles), num_images):
                    ax[i].set_title('', fontsize=fontsize)

    fig.tight_layout()


def showimages4(images, titles, fontsize=40):
# noisy, clear, 
# estimate1, estimate2       

    images_corrected_dims = []
    for i, img in enumerate(images):
        if img.dim() == 4:
            img = img[0]

        img = img[0] if img.size()[0] == 1 else img.permute(1,2,0)
        images_corrected_dims.append(img)

    images = images_corrected_dims
    num_images = len(images)

    figsize = (12,12)
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    for i, img in enumerate(images):
        ax[i//2, i%2].imshow(img, cmap='gray')
        ax[i//2, i%2].set_axis_off()

    # Now only titles have to be distributed.
    if titles is not None:
        if isinstance(titles, list):
            for i, title in enumerate(titles):
                ax[i//2, i%2].set_title(title, fontsize=fontsize)

            if len(titles) < num_images:
                for i in range(len(titles), num_images):
                    ax[i//2, i%2].set_title('', fontsize=fontsize)

    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.25,
                    wspace=0.01)


def showimages_trdpd(images, titles, patches=None, fontsize=40):
# noisy, clear, 
# estimate1, estimate2       

    images_corrected_dims = []
    for i, img in enumerate(images):
        if img.dim() == 4:
            img = img[0]

        img = img[0] if img.size()[0] == 1 else img.permute(1,2,0)
        images_corrected_dims.append(img)

    images = images_corrected_dims
    num_images = len(images)

    figsize = (18,10)
    fig, ax = plt.subplots(2, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    for i, img in enumerate(images):
        ax[i//3, i%3].imshow(img, cmap='gray')
        ax[i//3, i%3].set_axis_off()

    if patches is not None:
        # patches must be a list of patches or one patch
        for i in range(6):
            if patches[i] == None:
                continue
            ax[i//3, i%3].add_patch(patches[i])

    # Now only titles have to be distributed.
    if titles is not None:
        if isinstance(titles, list):
            for i, title in enumerate(titles):
                ax[i//3, i%3].set_title(title, fontsize=fontsize)

            if len(titles) < num_images:
                for i in range(len(titles), num_images):
                    ax[i//3, i%3].set_title('', fontsize=fontsize)

    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.25,
                    wspace=0.01)


def showimages4_lefttext(images, titles, fontsize=40): # GausProx # PoisProx
# noisy, clear, 
# estimate1, estimate2       

    images_corrected_dims = []
    for i, img in enumerate(images):
        if img.dim() == 4:
            img = img[0]

        img = img[0] if img.size()[0] == 1 else img.permute(1,2,0)
        images_corrected_dims.append(img)

    images = images_corrected_dims
    num_images = len(images)

    figsize = (12,12)
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    ax[0, 0].imshow(images[0], cmap='gray')
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    plt.setp(ax[0, 0].get_yticklabels(), visible=False)
    ax[0, 0].tick_params(axis='both', which='both', length=0)
    ax[0, 0].set_ylabel('GausProx', fontsize=fontsize)

    ax[0, 1].imshow(images[1], cmap='gray')
    ax[0, 1].set_axis_off()

    ax[1, 0].imshow(images[2], cmap='gray')
    plt.setp(ax[1,0].get_xticklabels(), visible=False)
    plt.setp(ax[1,0].get_yticklabels(), visible=False)
    ax[1, 0].tick_params(axis='both', which='both', length=0)
    ax[1, 0].set_ylabel('PoisProx', fontsize=fontsize)

    ax[1, 1].imshow(images[3], cmap='gray')
    ax[1, 1].set_axis_off()

    # Now only titles have to be distributed.
    if titles is not None:
        if isinstance(titles, list):
            for i, title in enumerate(titles):
                ax[i//2, i%2].set_title(title, fontsize=fontsize)

            if len(titles) < num_images:
                for i in range(len(titles), num_images):
                    ax[i//2, i%2].set_title('', fontsize=fontsize)

    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.25,
                    wspace=0.01)


def showimages4horizontal(images, titles, maxval=None, fontsize=40):
# noisy, clear, estimate1, estimate2       

    images_corrected_dims = []
    for i, img in enumerate(images):
        if img.dim() == 4:
            img = img[0]

        img = img[0] if img.size()[0] == 1 else img.permute(1,2,0)
        images_corrected_dims.append(img)

    images = images_corrected_dims
    num_images = len(images)

    figsize = (20,6)
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor('white')

    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        if maxval is not None and i == 0:
            plt.setp(ax[i].get_xticklabels(), visible=False)
            plt.setp(ax[i].get_yticklabels(), visible=False)
            ax[i].tick_params(axis='both', which='both', length=0)
            ax[i].set_ylabel('peak {}'.format(maxval), fontsize=fontsize)
        else:
            ax[i].set_axis_off()

    # Now only titles have to be distributed.
    if titles is not None:
        if isinstance(titles, list):
            for i, title in enumerate(titles):
                ax[i].set_title(title, fontsize=fontsize)

            if len(titles) < num_images:
                for i in range(len(titles), num_images):
                    ax[i].set_title('', fontsize=fontsize)
        

    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.25,
                    wspace=0.01)



def show_losses(flag_while=False, exp_name=None, log=False, loglog=False,\
                path2folder='./PoisDenoiser/networks/PoisNet/models/',\
                last_epochs=5, only_val=False):
    
    path2folder += exp_name+'/'

    fontsize = 20
    while True:
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        train = np.load(path2folder+'train_losses.npy')
        val = np.load(path2folder+'val_losses.npy')

        num_epochs = np.load(path2folder+'epoch.npy')
        
        train_x = np.linspace(0, num_epochs, num=len(train))
        val_x = np.linspace(0, num_epochs, num=len(val))
        
        train_x_last = int(len(train)/num_epochs*(num_epochs-last_epochs-1))
        val_x_last = int(len(val)/num_epochs*(num_epochs-last_epochs-1))
        
        if loglog:
            ax[0].loglog(train_x, train)
            ax[0].loglog(val_x, val)

            if not only_val:
                ax[1].loglog(train_x[train_x_last:], train[train_x_last:])
            ax[1].loglog(val_x[val_x_last:], val[val_x_last:])

        if log:
            ax[0].semilogy(train_x, train)
            ax[0].semilogy(val_x, val)

            if not only_val:
                ax[1].semilogy(train_x[train_x_last:], train[train_x_last:])
            ax[1].semilogy(val_x[val_x_last:], val[val_x_last:])

        else:
            ax[0].plot(train_x, train)
            ax[0].plot(val_x, val)

            if not only_val:
                ax[1].plot(train_x[train_x_last:], train[train_x_last:])
            ax[1].plot(val_x[val_x_last:], val[val_x_last:])
            
        # show only last last_epochs epochs
        ax[1].set_xlim((np.maximum(num_epochs-last_epochs, 0), num_epochs))

        if not only_val:
            min_y = np.minimum(train[train_x_last:].min(), val[val_x_last:].min())*0.9
            max_y = np.maximum(train[train_x_last:].max(), val[val_x_last:].max())*1.1
            ax[1].set_ylim(min_y, max_y)

        if min(train) > 0:
            ax[0].set_title('min train= {:.5f} | min val= {:.5f}'\
                    .format(min(train), min(val)), \
                    fontsize=fontsize)
            ax[1].set_title('last train= {:.5f} | last val= {:.5f}'\
                    .format(train[-1], val[-1]), \
                    fontsize=fontsize)
        fig.tight_layout()
        plt.show()

        params = np.load(path2folder+'params.npy').item()
        print('num images = ', params['num_train_imgs'])
        print('train size = {}, val size = {}'\
                .format(params['train_batchsize'], params['val_batchsize']))
        
        if flag_while:
            sleep(2)
            clear_output() 
        else:
            break


def do_inference(s, c, exp_name, saved_epoch, model_type='pois', \
    img_ind=None, img_from_train_dataset=False, clear_ind=0, app=None,\
    path2valdata='./DATASETS/BSDS500/BSDS500_validation_MAXVALs_01_2/', prox_param=False, sharing_weights=True, do_VST=False):

    if model_type == 'pois':
        model = PoisNet(output_features=c, stages=s, prox_param=prox_param, convWeightSharing=sharing_weights)
    elif model_type == 'l2':
        model = UDNet(output_features=c, stages=s, alpha=prox_param, convWeightSharing=sharing_weights)

    path2dataset = path2valdata
    BSDSval = BSDS500(path2dataset+'val/', get_name=True, do_VST_4_visual=do_VST) \
        if not img_from_train_dataset \
        else BSDS500(path2dataset+'train/', get_name=True, do_VST_4_visual=do_VST)
    

    if img_ind is None:
        img_ind = 80 

    if do_VST:
        gt, noisy, noisy_initial, file_name = BSDSval[img_ind]
    else:
        gt, noisy, file_name = BSDSval[img_ind]

    split = file_name.split('_')
    name, maxval = split[0], split[2][7:]
    maxval = (int(maxval[0])*10 + int(maxval[-1]))/10

    gt.unsqueeze_(0)
    noisy.unsqueeze_(0)

    state = th.load('./PoisDenoiser/networks/PoisNet/models/'\
        +exp_name+'/state_{}.pth'.format(saved_epoch),\
                   map_location=lambda storage,loc:storage)

    model.load_state_dict(state['model_state_dict'])

    if model_type == 'pois':
        estim = model(noisy, noisy).detach()
    elif model_type == 'l2':
        stdn = th.Tensor([1])
        estim = model(noisy, stdn, noisy).detach()

    psnr_noisy = psnr(gt, noisy_initial) if do_VST else psnr(gt, noisy)
    psnr_est = psnr(gt, VST_backward_unbiased_exact(estim)) if do_VST else psnr(gt, estim)

    gt_title = 'clear ({} in BSDS val)'.format(img_ind)
    noisy_title = 'noisy (max val={}) \nPSNR: {:.2f} dB'.format(maxval, psnr_noisy)
    if app is not None:  
        estim_title = '{} (epoch={}),\nPSNR: {:.3f} dB'.format(app, saved_epoch, psnr_est)
    else:
        estim_title = 'estim (epoch={}),\nPSNR: {:.3f} dB'.format(saved_epoch, psnr_est)

    if clear_ind == 0:
        show_images([gt, noisy, estim], [gt_title, noisy_title, estim_title])
    else:
        show_images([noisy, gt, estim], [noisy_title, gt_title, estim_title])


def do_inference_return_lists(noisy, gt, s, c, exp_name, saved_epoch, model_type='pois',\
    app=None,  prox_param=False, sharing_weights=True):

    if model_type == 'pois':
        model = PoisNet(output_features=c, stages=s, prox_param=prox_param, convWeightSharing=sharing_weights)
    elif model_type == 'l2':
        model = UDNet(output_features=c, stages=s, alpha=prox_param, convWeightSharing=sharing_weights)

    state = th.load('./PoisDenoiser/networks/PoisNet/models/'\
        +exp_name+'/state_{}.pth'.format(saved_epoch),\
                   map_location=lambda storage,loc:storage)

    model.load_state_dict(state['model_state_dict'])

    if model_type == 'pois':
        estim = model(noisy, noisy).detach()
    elif model_type == 'l2':
        stdn = th.Tensor([1])
        estim = model(noisy, stdn, noisy).detach()

    psnr_est = psnr(gt, estim)

    if app is not None:  
        estim_title = '{} (epoch={}),\nPSNR: {:.3f} dB'.format(app, saved_epoch, psnr_est)
    else:
        estim_title = 'estim (epoch={}),\nPSNR: {:.3f} dB'.format(saved_epoch, psnr_est)

    return estim, estim_title


def compare_pois_l2(s, c, saved_epoch, img_ind, add2exp='_400/',\
    path2valdata='./DATASETS/BSDS500/BSDS500_validation_MAXVALs_01_2/'):

    exp_name_pois = 'pois'+add2exp+'s{}c{}'.format(s,c)
    exp_name_l2 = 'l2'+add2exp+'s{}c{}'.format(s,c)

    model_pois = PoisNet(output_features=c, stages=s)
    model_l2 = UDNet(output_features=c, stages=s)

    path2dataset = path2valdata
    BSDSval = BSDS500(path2dataset+'val/', get_name=True)

    gt, noisy, file_name = BSDSval[img_ind]
    split = file_name.split('_')
    name, maxval = split[0], split[2][7:]
    maxval = (int(maxval[0])*10 + int(maxval[-1]))/10

    gt.unsqueeze_(0)
    noisy.unsqueeze_(0)

    state = th.load('./PoisDenoiser/networks/PoisNet/models/'\
        +exp_name_pois+'/state_{}.pth'.format(saved_epoch),\
                   map_location=lambda storage,loc:storage)
    model_pois.load_state_dict(state['model_state_dict'])
    estim_pois = model_pois(noisy, noisy).detach()

    state = th.load('./PoisDenoiser/networks/PoisNet/models/'\
        +exp_name_l2+'/state_{}.pth'.format(saved_epoch),\
                   map_location=lambda storage,loc:storage)
    model_l2.load_state_dict(state['model_state_dict'])
    stdn = th.Tensor([5])
    estim_l2 = model_l2(noisy, stdn, noisy).detach()

    psnr_noisy = psnr(gt, noisy)
    psnr_est_pois = psnr(gt, estim_pois)
    psnr_est_l2 = psnr(gt, estim_l2)

    gt_title = 'clear ({} in BSDS val)'.format(img_ind)
    noisy_title = 'noisy (max val={}) \nPSNR: {:.2f} dB'.format(maxval, psnr_noisy)
    estim_pois_title = '{} (epoch={}),\nPSNR: {:.3f} dB'\
    .format('pois', saved_epoch, psnr_est_pois)
    estim_l2_title = '{} (epoch={}),\nPSNR: {:.3f} dB'\
    .format('l2', saved_epoch, psnr_est_l2)

    show_images([noisy, gt, estim_pois, estim_l2], \
                [noisy_title, gt_title, estim_pois_title, estim_l2_title])


def compare_pois_l2_pois_w_prox(s, c, saved_epoch, img_ind, add2exp='_400/',\
    path2valdata='./DATASETS/BSDS500/BSDS500_validation_MAXVALs_01_2/'):

    exp_name_pois = 'pois'+add2exp+'s{}c{}'.format(s,c)
    exp_name_poisprox = 'pois_w_prox'+add2exp+'s{}c{}'.format(s,c)
    exp_name_l2 = 'l2'+add2exp+'s{}c{}'.format(s,c)

    model_pois = PoisNet(output_features=c, stages=s)
    model_poisprox = PoisNet(output_features=c, stages=s, prox_param=True)
    model_l2 = UDNet(output_features=c, stages=s)

    path2dataset = path2valdata
    BSDSval = BSDS500(path2dataset+'val/', get_name=True)

    gt, noisy, file_name = BSDSval[img_ind]
    split = file_name.split('_')
    name, maxval = split[0], split[2][7:]
    maxval = (int(maxval[0])*10 + int(maxval[-1]))/10

    gt.unsqueeze_(0)
    noisy.unsqueeze_(0)

    state = th.load('./PoisDenoiser/networks/PoisNet/models/'\
        +exp_name_pois+'/state_{}.pth'.format(saved_epoch),\
                   map_location=lambda storage,loc:storage)
    model_pois.load_state_dict(state['model_state_dict'])
    estim_pois = model_pois(noisy, noisy).detach()

    state = th.load('./PoisDenoiser/networks/PoisNet/models/'\
        +exp_name_poisprox+'/state_{}.pth'.format(saved_epoch),\
                   map_location=lambda storage,loc:storage)
    model_poisprox.load_state_dict(state['model_state_dict'])
    estim_poisprox = model_poisprox(noisy, noisy).detach()

    state = th.load('./PoisDenoiser/networks/PoisNet/models/'\
        +exp_name_l2+'/state_{}.pth'.format(saved_epoch),\
                   map_location=lambda storage,loc:storage)
    model_l2.load_state_dict(state['model_state_dict'])
    stdn = th.Tensor([5])
    estim_l2 = model_l2(noisy, stdn, noisy).detach()

    psnr_noisy = psnr(gt, noisy)
    psnr_est_pois = psnr(gt, estim_pois)
    psnr_est_poisprox = psnr(gt, estim_poisprox)
    psnr_est_l2 = psnr(gt, estim_l2)

    gt_title = 'clear ({} in BSDS val)'.format(img_ind)
    noisy_title = 'noisy (max val={}) \nPSNR: {:.2f} dB'.format(maxval, psnr_noisy)
    estim_pois_title = '{} (epoch={}),\nPSNR: {:.3f} dB'\
    .format('pois', saved_epoch, psnr_est_pois)
    estim_poisprox_title = '{} (epoch={}),\nPSNR: {:.3f} dB'\
    .format('poisprox', saved_epoch, psnr_est_poisprox)
    estim_l2_title = '{} (epoch={}),\nPSNR: {:.3f} dB'\
    .format('l2', saved_epoch, psnr_est_l2)


    images = [noisy, gt, estim_pois, estim_poisprox, estim_l2]
    titles = [noisy_title, gt_title, estim_pois_title, estim_poisprox_title, estim_l2_title]

    fontsize = 15



    images_corrected_dims = []
    for i, img in enumerate(images):
        if img.dim() == 4:
            img = img[0]

        img = img[0] if img.size()[0] == 1 else img.permute(1,2,0)
        images_corrected_dims.append(img)

    images = images_corrected_dims

    figsize = (20,10)
    fig, ax = plt.subplots(2, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    ax[0,0].imshow(images[0], cmap='gray')
    ax[0,0].set_axis_off()
    ax[0,0].set_title(noisy_title, fontsize=fontsize)
    ax[0,1].imshow(images[1], cmap='gray')
    ax[0,1].set_axis_off()
    ax[0,1].set_title(gt_title, fontsize=fontsize)
    ax[1,0].imshow(images[2], cmap='gray')
    ax[1,0].set_axis_off()
    ax[1,0].set_title(estim_pois_title, fontsize=fontsize)
    ax[1,1].imshow(images[3], cmap='gray')
    ax[1,1].set_axis_off()
    ax[1,1].set_title(estim_poisprox_title, fontsize=fontsize)
    ax[1,2].imshow(images[4], cmap='gray')
    ax[1,2].set_axis_off()
    ax[1,2].set_title(estim_l2_title, fontsize=fontsize)
    ax[0,2].remove()

    fig.tight_layout()