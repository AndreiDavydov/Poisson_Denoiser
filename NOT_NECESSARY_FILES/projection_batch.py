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

    x = x.view(batch_size, -1)#.double() ################################################################################
    y = y.view(batch_size, -1)#.double() ################################################################################
    
    assert x.numel() == y.numel()
    
    nnz_mask = (y>0).type_as(x)

    x_masked = x.mul(nnz_mask)
    x_masked[x_masked < 1e-8] = 1 # otherwise, log( <=0 ) would give NaNs.

    return x.sum(dim=1) - y.mul(th.log(x_masked)).mul(nnz_mask).sum(dim=1)


def f(x, y):
    '''
    Compute condition function.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    m = (y>0).type_as(x).view(x.size(0),-1).sum(dim=1)#.double()  ####################################################
    mu = Pois(y,y) + m/2 
    return Pois(x,y) - mu


def g(x, y, alpha):
    '''
    Projection, is computed in case when x does not lie in the set.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    assert(alpha.dim() == 4), \
    'Input size of Alpha must be Bx1x1x1.'

    root = th.sqrt((x - alpha)**2 + 4*alpha*y)
    res = 1/2 * (x - alpha + root)
    return res


def is_in_C(x, y, upper_bound=0):
    '''
    The boolean function, which checks whether x lies in the set.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    return (f(x,y) <= upper_bound)


def df(x, y, alpha):
    '''
    The gradient of f(g(alpha)) by alpha, given x and y.
    '''
    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    batch_size = x.size(0)

    x = x.view(batch_size, -1)#.double()##########################################################
    y = y.view(batch_size, -1)#.double()##########################################################
    alpha = alpha.view(batch_size, -1)#.double()##########################################################

    assert x.numel() == y.numel()
    
    tol = th.Tensor([1e-8])#.double()###############################################################
    x_tol = x + tol
    y_tol = y + tol

    root = th.sqrt((x_tol - alpha)**2 + 4*alpha*y_tol)
    tmp = x_tol - alpha + root
    
    vec = ( tmp - 2*y_tol )**2 / root / tmp
    
    return -1/2 * vec.sum(dim=1)           


def newton(x,y, eps=1e-2, display=True):
    '''
    Finds a root of required equation by Newton method. "df" function is used to compute the derivative.
    Returns optimal ALPHA value. 
    '''

    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    batch_size = x.size(0)
    alpha = th.zeros((batch_size, 1,1,1))#.double()  ############################################################
    x = x#.double()                                  ############################################################
    y = y#.double()                                  ############################################################
    
    f_val = f(g(x,y,alpha=alpha.clone()), y)

    num_operations = 0

    while (th.abs(f_val) > eps).sum() > 0:
        df_val = df(x,y,alpha=alpha.clone())
        
        if display:
            print('{:.3f}, \t\t{:.3f}, '.format(alpha.view(-1)[0], f_val[0]))

        alpha -= (f_val/df_val).view(-1, 1,1,1)
        f_val = f(g(x,y,alpha=alpha.clone()), y)
        
        num_operations += 1

    if display:
        print('{:.3f}, \t\t{:.3f}, \n\n'.format(alpha.view(-1)[0], f_val[0]))

        print(num_operations)

    return alpha


def newton_fixed_iters(x,y, num_iters=10):
    '''
    Finds a root of required equation by Newton method. "df" function is used to compute the derivative.
    Returns optimal ALPHA value. 
    '''

    assert (x.dim() == 4 and y.dim() == 4), \
    'Input size must be like BxCxHxW.'

    batch_size = x.size(0)
    alpha = th.zeros((batch_size, 1,1,1))#.double()  ############################################################
    x = x#.double()                                  ############################################################
    y = y#.double()                                  ############################################################
    

    f_val = f(g(x,y,alpha=alpha), y)

    for _ in range(num_iters):
        df_val = df(x,y,alpha=alpha.clone())
        alpha -= (f_val/df_val).view(-1, 1,1,1)
        f_val = f(g(x,y,alpha=alpha.clone()), y)
        
    return alpha

    
def Projection(x, y, fixed=True, num_iters=10, eps=1e-2, display=False):
    '''
    Computes the projection of X on the set C.
    Returns X-like object.
    
    If X does not lie in C, then it is needed to find alpha to compute the projection.
    The alpha parameter is computed via Newton iterative method.
    '''
    
    x = x#.double()##########################################################
    y = y#.double()##########################################################

    mask_is_in_C = is_in_C(x,y).view(-1, 1,1,1).type_as(x)#.double() ######################################

    if fixed:
        alpha = newton_fixed_iters(x,y, num_iters=num_iters)
    else:
        alpha = newton(x,y, eps=eps, display=display)

    x_proj = g(x,y,alpha=alpha.clone())
    x_proj[th.isnan(x_proj)] = 0

    return x_proj.mul(1-mask_is_in_C) + x.mul(mask_is_in_C)