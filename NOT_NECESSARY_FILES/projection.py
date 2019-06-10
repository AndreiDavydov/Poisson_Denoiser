from time import time
import torch as th


def Pois(x, y):
    '''
    Poisson log-likelihood function. 1^T x - y^T log(x).
    
    The necessary point to elaborate on is the case when x=y and y=0 in some points.
    Then the nnz mask must be taken into account.
    '''
    
    x = x.view(-1)
    y = y.view(-1)
    
    assert x.numel() == y.numel()
    
    ones = th.ones(x.numel())
    nnz_mask = y>0
    
    return (th.dot(ones, x) - th.dot(y[nnz_mask], th.log(x[nnz_mask])))


def f(x, y):
    '''
    Compute condition function.
    '''
    m = (y>0).sum().type_as(x)

    mu = Pois(y,y) + m/2
    return Pois(x,y) - mu


def g(x, y, alpha=0):
    '''
    Projection, is computed in case when x does not lie in the set.
    '''
    root = th.sqrt((x - alpha)**2 + 4*alpha*y)
    res = 1/2 * (x - alpha + root)
    return res


def is_in_C(x, y, upper_bound=0):
    '''
    The boolean function, which checks whether x lies in the set.
    '''
    return True if f(x,y) <= upper_bound else False


def df(x, y, alpha=0):
    '''
    The gradient of f(g(alpha)) by alpha, given x and y.
    '''
    x = x.view(-1)
    y = y.view(-1)
    assert x.numel() == y.numel()
    
    tol = th.Tensor([1e-8])
    x_tol = x + tol
    y_tol = y + tol
    
    ones = th.ones(x_tol.numel())
    root = th.sqrt((x_tol - alpha)**2 + 4*alpha*y_tol)
    tmp = x_tol - alpha + root
    
    vec = ( tmp - 2*y_tol )**2 / root / tmp
    
    return -1/2 * th.dot(ones, vec)


def newton(x,y, eps=1e-2, display=False):
    '''
    Finds a root of required equation by Newton method. "df" function is used to compute the derivative.
    Returns optimal ALPHA value. 
    '''
    alpha = th.Tensor([0])
    if is_in_C(x,y):
        return alpha
    
    f_val = f(g(x,y,alpha=alpha.clone()), y)
    
    num_operations = 0

    while th.abs(f_val) > eps:
        df_val = df(x,y,alpha=alpha)
        
        if display:
            print('{}, \t\t{}, '.format(alpha.clone().numpy(), f_val))

        alpha -= f_val/df_val
        f_val = f(g(x,y,alpha=alpha.clone()), y)
        
        num_operations += 1

    if display:
        print('{}, \t\t{}, \n\n'.format(alpha.clone().numpy(), f_val))

        print(num_operations)

    return alpha


def newton_fixed_iters(x,y, num_iters=10):
    '''
    Finds a root of required equation by Newton method. "df" function is used to compute the derivative.
    Returns optimal ALPHA value. 
    '''
    alpha = th.Tensor([0])
    if is_in_C(x,y):
        return alpha
    
    f_val = f(g(x,y,alpha=alpha), y)

    for _ in range(num_iters):
        df_val = df(x,y,alpha=alpha.clone())
        alpha -= f_val/df_val
        f_val = f(g(x,y,alpha=alpha.clone()), y)
        
    return alpha

    
def Projection(x, y, fixed=True, num_iters=10, eps=1e-2, display=False):
    '''
    Computes the projection of X on the set C.
    Returns X-like object.
    
    If X does not lie in C, then it is needed to find alpha to compute the projection.
    The alpha parameter is computed via Newton iterative method.
    '''
    
    if is_in_C(x,y):
        return x
    else:
        if fixed:
            alpha = newton_fixed_iters(x,y, num_iters=num_iters)
        else:
            alpha = newton(x,y, eps=eps, display=display)

        x_proj = g(x,y,alpha=alpha.clone())
        return x_proj