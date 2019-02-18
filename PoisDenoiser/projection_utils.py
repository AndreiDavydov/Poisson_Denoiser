from time import time
import torch as th


'''
All functions mimic a functional of "projection.py" module 
but do it in a Pytorch nn mode, when the input size of Tensors 
is B x C x H x W


All computations were made in Double-precision. Further tests will show whether one needs such accuracy or not.

NB: ALL FUNCTIONS HAVE BEEN REWRITTEN IN SINGLE PRECISION.
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
    
    nnz_mask = (y>0).type_as(x)

    x_masked = x.mul(nnz_mask)
    x_masked[x_masked < 1e-8] = 1 # otherwise, log( <=0 ) would give NaNs.

    return x.sum(dim=1) - y.mul(th.log(x_masked)).mul(nnz_mask).sum(dim=1)


def f(x, y, a_cond=None):
    '''
    Compute condition function.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    if a_cond is None:
        a_cond = th.Tensor([0]).type_as(x)

    m = (y>0).type_as(x).view(x.size(0),-1).sum(dim=1)
    mu = th.exp(a_cond)*(Pois(y,y) + m/2) 

    return Pois(x,y) - mu


def g(x, y, alpha):
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


def is_in_C(x, y, a_cond=None, upper_bound=0):
    '''
    The boolean function, which checks whether x lies in the set.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    if a_cond is None:
        a_cond = th.Tensor([0]).type_as(x)

    return (f(x,y, a_cond=a_cond) <= upper_bound)


def df(x, y, alpha): 
    '''
    The gradient of f(g(alpha)) by alpha, given x and y.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    batch_size = x.size(0)

    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    alpha = alpha.view(batch_size, -1)

    assert x.numel() == y.numel()
    
    tol = th.Tensor([1e-8]).type_as(x)

    root = th.sqrt((x - alpha)**2 + 4*alpha*y + tol)
    tmp = x - alpha + root + tol
    vec = ( tmp - 2*y )**2 / root / tmp
    
    return -1/2 * vec.sum(dim=1)           


def newton(x,y, cond_mask, a_cond=None, num_iters=2): # fixed_iters only!
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

    f_val = f(g(x,y,alpha=alpha)[0], y, a_cond=a_cond)

    for _ in range(num_iters):
        df_val = df(x,y,alpha=alpha.clone())
        alpha -= (f_val/df_val).view(-1, 1,1,1) # f_val > 0, df_val < 0 !!!
        f_val = f( g(x,y,alpha=alpha.clone())[0], y, a_cond=a_cond)

    alpha[cond_mask] = 0

    if th.isnan(alpha).sum() > 0:
        print('NEWTON GIVES NAN ALPHA!!!')
        save_path = './PoisDenoiser/networks/PoisNet/models/s1c8_wtf/'
        savings = {'x': x, 'y':y,\
                    'alpha':alpha}  

        path2file = save_path+'newton_ERROR.pth'
        th.save(savings, path2file)  
        return 
        
    return alpha

    
def Projection(x, y, a_cond=None, num_iters=2):
    '''
    Computes the projection of X on the set C.
    Returns X-like object.
    
    If X does not lie in C, then it is needed to find alpha to compute the projection.
    The alpha parameter is computed via Newton iterative method.
    '''

    cond_mask = is_in_C(x,y,a_cond=a_cond).view(-1, 1,1,1)#.type_as(x)

    alpha = newton(x,y, cond_mask, a_cond=a_cond) # takes into account cond_mask

    x_proj, root = g(x,y,alpha=alpha.clone())               
    # x_proj[th.isnan(x_proj)] = 0

    return x_proj, root # for grad_input computation!