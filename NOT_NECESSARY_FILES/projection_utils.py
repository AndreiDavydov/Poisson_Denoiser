from time import time
import torch as th


'''
All functions mimic a functional of "projection.py" module 
but do it in a Pytorch nn mode, when the input size of Tensors 
is B x C x H x W

Hereafter, x - estimate of clear image, y - given noisy version.
'''

def Pois(x, y):
    '''
    Poisson log-likelihood function. 1^T x - y^T log(x).
    
    The necessary point to elaborate on is the case when x=y and y=0 in some points.
    Then the nnz mask must be taken into account.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
        'Input size must be like BxCxHxW.'

    batch_size = x.size(0)

    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    
    assert x.numel() == y.numel()
    
    x = th.clamp(x, th.Tensor([1e-8]).type_as(x), \
                    th.Tensor([float('Inf')]).type_as(x))

    # nnz_mask = (y>0).type_as(x)
    # x_masked = x.mul(nnz_mask)
    # x_masked[x_masked < 1e-8] = 1e-8 # otherwise, log( <=0 ) would give NaNs.
    # return x.sum(dim=1) - y.mul(th.log(x_masked)).mul(nnz_mask).sum(dim=1)

    return x.sum(dim=1) - y.mul(th.log(x)).sum(dim=1)


def cond_func(x, y): # a_cond=None
    '''
    Compute condition function.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
        'Input size must be like BxCxHxW.'

    # if a_cond is None:
    #     a_cond = th.Tensor([0]).type_as(x)

    m = (y>0).type_as(y).view(y.size(0),-1).sum(dim=1)
    mu = Pois(y,y) + m/2 #th.exp(a_cond)

    return Pois(x,y) - mu


def proj_func(x, y, alpha):
    '''
    Projection, is computed in case when x does not lie in the set.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
        'Input size must be like BxCxHxW.'

    assert(alpha.dim() == 4), \
        'Input size of Alpha must be Bx1x1x1.'

    tol = th.Tensor([1e-8]).type_as(x)

    sqrt_arg = (x - alpha)**2 + 4*alpha*y
    root = th.sqrt(sqrt_arg + tol)
    res = 1/2 * (x - alpha + root)

    return res, root # "root" is returned for grad computing


def is_in_C(x, y):
    '''
    The boolean function, which checks whether x lies in the set.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
        'Input size must be like BxCxHxW.' 

    return (cond_func(x,y) <= 0)


def dcond_dalpha(x, y, alpha): 
    '''
    The gradient of cond_func(proj_func(alpha)) by alpha, given x and y.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
        'Input size must be like BxCxHxW.'
    assert (noisy<0).sum() == 0, \
        'Noisy images values must be nonnegative.'
    assert (input<0).sum() == 0, \
        'Input values must be nonnegative.' 
    assert (alpha<0).sum() == 0, \
        'Alpha parameter must be nonnegative.' 

    batch_size = x.size(0)

    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    alpha = alpha.view(batch_size, -1)

    assert x.numel() == y.numel()
    
    tol = th.Tensor([1e-8]).type_as(x)

    sqrt_arg = (x - alpha)**2 + 4*alpha*y  
    root = th.sqrt(sqrt_arg + tol)
    tmp = x - alpha + root + tol
    vec = ( tmp - 2*y )**2 / root / tmp
    
    return -1/2 * vec.sum(dim=1)           


def newton(x,y, cond_mask, a_cond=None, num_iters=2):
    '''
    Finds a root of required equation by Newton method. 
    "df" function is used to compute the derivative.
    Returns optimal ALPHA value. 
    '''

    assert (x.dim() == 4 and y.dim() == 4), \
        'Input size must be like BxCxHxW.'

    if a_cond is None:
        a_cond = th.Tensor([0]).type_as(x)   

    batch_size = x.size(0)
    alpha = th.zeros((batch_size, 1,1,1)).type_as(x)

    f_val = cond_func(proj_func(x,y,alpha=alpha)[0], y, a_cond=a_cond)

    for _ in range(num_iters):
        df_val = dcond_dalpha(x,y,alpha=alpha.clone())
        alpha -= (f_val/df_val).view(-1, 1,1,1) # f_val > 0, df_val < 0 !!!
        f_val = cond_func( proj_func(x,y,alpha=alpha.clone())[0], y, a_cond=a_cond)

    alpha[cond_mask] = 0
        
    return alpha

    
def Projection(x, y, a_cond=None, num_iters=2):
    '''
    Computes the projection of X on the set C.
    Returns X-like object.
    
    If X does not lie in C, then it is needed to find alpha to compute the projection.
    The alpha parameter is computed via Newton iterative method.
    '''

    cond_mask = is_in_C(x,y,a_cond=a_cond).view(-1, 1,1,1)

    alpha = newton(x,y, cond_mask, a_cond=a_cond) # takes into account cond_mask

    x_proj, root = proj_func(x,y,alpha=alpha.clone())               
    # x_proj[th.isnan(x_proj)] = 0

    return x_proj, root # for grad_input computation!