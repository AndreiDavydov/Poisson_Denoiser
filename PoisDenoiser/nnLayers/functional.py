import torch as th
import torch.nn as nn
from torch.nn.functional import pad
import torchvision.models as models

TOL = 1e-7

def poisLikelihoodFunc(input, noisy):
    '''
    Computation of the Poisson log-likelihood function. 
     __________________________________________
    |                                          |
    | out = 1^T * input - noisy^T * log(input) |
    |__________________________________________|
    '''
    assert (noisy<0).sum() == 0, \
        'Noisy images values must be nonnegative.'
    assert (input<0).sum() == 0, \
        'Input values must be nonnegative.' 

    input_shape = input.size()
    input = input.view(input.size(0), -1)
    noisy = noisy.view(input.size(0), -1)

    input = th.clamp(input, TOL, float('Inf'))

    return input.sum(dim=1) - noisy.mul(th.log(input)).sum(dim=1)


def condFunc(input, noisy,prox_param):
    '''
    Compute condition function.
     __________________________________________________________________
    |                                                                  |
    | out = pois(input, noisy) - pois(noisy, noisy) - #(noisy > 0) / 2 |
    |__________________________________________________________________|
    '''
    assert (input.dim() == 4 and noisy.dim() == 4), \
        'Input size must be like BxCxHxW.'
    assert(prox_param is None or prox_param.numel() == 1), \
    "prox_param needs to be either None or a tensor of size 1."

    if prox_param is None:
        prox_param = th.Tensor([0]).type_as(input)
        prox_param.requires_grad = False

    m = ( noisy > 0 ).type_as(noisy).view(noisy.size(0),-1).sum(dim=1)
    mu = (poisLikelihoodFunc(noisy,noisy) + m/2)*th.exp(prox_param)

    return poisLikelihoodFunc(input,noisy) - mu


def dCond_dAlphaFunc(input, noisy, alpha):
    '''
    Computes the d projFunc(condFunc(input,noisy,alpha),noisy)/ dAlpha.
    It is needed to compute Newton iteration for alpha approximation.
     _________________________________________________________
    |                                                         |
    | square_root = sqrt( (input - alpha)^2 + 4*alpha*noisy ) |
    |                                                         |
    | tmp = input - alpha + square_root                       |        
    |                                                         |
    | out = -1/2 * ( tmp - 2*noisy )^2 / square_root / tmp    |
    |_________________________________________________________|   
    '''
    assert(input.dim() == 4 and \
           noisy.dim() == 4 and \
           alpha.dim() == 4), \
        "Input, noisy and alpha are expected to be 4-D tensors."
    assert (alpha<0).sum() == 0, \
        'Alpha parameter must be nonnegative.'
    assert (noisy<0).sum() == 0, \
        'Noisy images values must be nonnegative.'
    assert (input<0).sum() == 0, \
        'Input values must be nonnegative.'  

    input = input.view(input.size(0), -1)
    noisy = noisy.view(noisy.size(0), -1)
    alpha = alpha.view(noisy.size(0), -1)
    
    tol = th.Tensor([TOL]).type_as(input)

    sqrt_arg = (input - alpha)**2 + 4*alpha*noisy  
    square_root = th.sqrt(sqrt_arg + tol)
    tmp = input - alpha + square_root + tol
    out = ( tmp - 2*noisy )**2 / square_root / tmp
    
    return -1/2 * out.sum(dim=1)  


def projFunc(input, noisy, alpha):
    '''
    Computes the projection of input, which does not lie in the Condition.
     _____________________________________________________________________
    |                                                                     |
    | out = 1/2*( input-alpha + sqrt( (input-alpha)^2 + 4*alpha*noisy ) ) |
    |_____________________________________________________________________|   
    '''
    assert(input.dim() == 4 and \
           noisy.dim() == 4 and \
           alpha.dim() == 4), \
        "Input, noisy and alpha are expected to be 4-D tensors."
    assert (alpha<0).sum() == 0, \
        'Alpha parameter must be nonnegative.'
    assert (noisy<0).sum() == 0, \
        'Noisy images values must be nonnegative.'
    assert (input<0).sum() == 0, \
        'Input values must be nonnegative.'  

    tol = th.Tensor([TOL]).type_as(input)

    input_proj = th.zeros_like(input)

    # Two following blocks are needed to be separated due to gradient computation.
    # Without first block (alpha == 0) the gradient would 
    # be equal to 1/2 instead of 1 due to tol.

    zero_mask = (alpha <= tol).view(-1)
    if zero_mask.sum() > 0:
        input_proj[zero_mask] = input[zero_mask]

    nnz_mask = (alpha > tol).view(-1)
    if nnz_mask.sum() > 0:
        sqrt_arg = (input[nnz_mask] - alpha[nnz_mask])**2 + 4*alpha[nnz_mask]*noisy[nnz_mask]
        square_root = th.sqrt(sqrt_arg + tol)
        input_proj[nnz_mask] = 1/2 * (input[nnz_mask] - alpha[nnz_mask] + square_root)         

    return input_proj


def alphaFunc(input, noisy, prox_param, num_iters=5):

    assert(input.dim() == 4 and noisy.dim() == 4), \
        "Input, noisy and alpha are expected to be 4-D tensors."
    assert (noisy<0).sum() == 0, \
        'Noisy images values must be nonnegative.'
    assert (input<0).sum() == 0, \
        'Input values must be nonnegative.'
    assert(prox_param is None or prox_param.numel() == 1), \
    "prox_param needs to be either None or a tensor of size 1."

    if prox_param is None:
        prox_param = th.Tensor([0]).type_as(input)
        prox_param.requires_grad = False

    not_in_cond_mask = (condFunc(input,noisy,prox_param) > 0)
    cond_mask = (condFunc(input,noisy,prox_param) <= 0)

    alpha = th.zeros((input.size(0),1,1,1)).type_as(input)

    alpha_in_cond = th.zeros((cond_mask.sum(), 1,1,1)).type_as(input)
    alpha_not_in_cond = th.zeros((not_in_cond_mask.sum(), 1,1,1)).type_as(input)    

    if not_in_cond_mask.sum() > 0: # if there are any NOT IN COND inputs
        for _ in range(num_iters):
            # must be f_val > 0, df_val < 0 !
            f_val = condFunc( projFunc(input[not_in_cond_mask],\
                                       noisy[not_in_cond_mask],\
                                       alpha_not_in_cond),\
                            noisy[not_in_cond_mask],prox_param)

            df_val = dCond_dAlphaFunc(input[not_in_cond_mask],\
                                      noisy[not_in_cond_mask],\
                                      alpha_not_in_cond)

            alpha_not_in_cond -= (f_val/df_val).view(-1, 1,1,1)
            alpha_not_in_cond = th.clamp(alpha_not_in_cond, 0, float('Inf'))

        alpha[not_in_cond_mask] = alpha_not_in_cond


    if cond_mask.sum() > 0: # if there are any IN COND inputs
        num_in_cond = input[cond_mask].size(0)
        alpha_in_cond = input[cond_mask].mul(th.zeros_like(input[cond_mask]))
        alpha_in_cond = alpha_in_cond.view(num_in_cond,-1).sum(dim=1)
        alpha_in_cond = alpha_in_cond.view(-1,1,1,1)

        alpha[cond_mask] = alpha_in_cond

    return alpha   


def poisProx(input, noisy, prox_param=None):
    '''
    X - input (estimate of clear image)
    Y - noisy (noisy version of the ground truth)

    Z = PoisProx(X, Y) computes the proximal map layer for the 
    indicator function:

    Z = prox_IC(Y){X} = 

    = argmin ||X-Z||^2     
        Z
    s.t. 1^T*Z-Y^T*log(Z) <= ( 1^T*Y-Y^T*log(Y) + ( # Y>0 )/2 ) 
                                
    = argmin ||Z-X||^2 + i_C(Y){Z} = projFunc (X,Y, Alpha)
        Z       

    X, Y, Z are all tensors of size B x C x H x W.
    '''

    assert(input.dim() == 4 and noisy.dim() == 4), \
        "Input, noisy and alpha are expected to be 4-D tensors."
    assert (noisy<0).sum() == 0, \
        'Noisy images values must be nonnegative.'
    assert (input<0).sum() == 0, \
        'Input values must be nonnegative.'
    assert(prox_param is None or prox_param.numel() == 1), \
    "prox_param needs to be either None or a tensor of size 1."

    if prox_param is None:
        prox_param = th.Tensor([0]).type_as(input)
        prox_param.requires_grad = False

    alpha = alphaFunc(input, noisy, prox_param)
    input_proj = projFunc(input, noisy, alpha)

    return input_proj



####################################################3

'''
Here are Loss functions.
'''


def poisLLHLoss(input, gt):
    res = poisLikelihoodFunc(input, gt)
    return res.sum()

class PerceptualLoss():
    
    def __init__(self, loss='MSE', layer_inds=[14], use_gpu=True):
        super(PerceptualLoss, self).__init__()

        self.final_size = (256,256) # VGG requires such input size
        if loss == 'L1':
            self.criterion = nn.L1Loss()
        else: # by default - MSELoss
            self.criterion = nn.MSELoss()

        self.layer_inds = layer_inds

        cnn = models.vgg19(pretrained=True).features
        if use_gpu:
            cnn = cnn.cuda()
        self.model = nn.Sequential()
        if use_gpu:
            self.model = self.model.cuda()

        for i, layer in enumerate(list(cnn)):
            self.model.add_module(str(i),layer)
            if i == self.layer_inds[-1]:
                break
            
    def __call__(self, clear, estim):
        # size must be (256,256)
        # Assume that size is smaller.
        if clear.size(1) == 1:
            clear = th.cat((clear, th.zeros_like(clear), th.zeros_like(clear)), \
                dim=1)
            estim = th.cat((estim, th.zeros_like(estim), th.zeros_like(estim)), \
                dim=1)

        padH, padW = 256-clear.size(2), 256-clear.size(3)
        pad_left, pad_right = padW//2, padW-padW//2
        pad_top, pad_bottom = padH//2, padH-padH//2
        

        clear_outs = []
        estim_outs = []
        for i, module in enumerate(list(self.model)):
            clear = module(clear)
            estim = module(estim)
            if i in self.layer_inds:
                clear_outs.append(clear)
                estim_outs.append(estim)

        total_loss = 0
        for clear_out, estim_out in zip(clear_outs, estim_outs):
            clear_out.detach_()
            total_loss += self.criterion(clear_out, estim_out)/clear_out.numel()
        return total_loss

    def do_pad(self, x):
        padH, padW = self.final_size[0]-x.size(2), \
                     self.final_size[1]-x.size(3)
        pad_left, pad_right = padW//2, padW-padW//2
        pad_top, pad_bottom = padH//2, padH-padH//2
        pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)
        return pad(x, pad_tuple)







