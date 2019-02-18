from PoisDenoiser.nnLayers.functional import PoisProx
import numpy as np
import torch as th
from torch.autograd import Variable
from pydl.nnLayers.functional import functional


def gradCheck_poisProx(epsilon=1e-4, dtype='torch.DoubleTensor', GPU=False):
    poisProx = PoisProx.apply

    x = th.rand(4,3,40,40).type(dtype)

    x = x*255
    z = th.randn(4,3,40,40).type(dtype)

    z = z*255    
    alpha  = th.Tensor(np.random.randint(0,3,(1,))).type(dtype)
    stdn = th.Tensor(np.random.randint(5,20,(4,1))).type(dtype)
    

def l2Prox(epsilon=1e-4,dtype='torch.DoubleTensor',GPU=False):
    
    l2ProxF = functional.L2Prox.apply
    
    x = th.randn(4,3,40,40).type(dtype)
    x -= x.view(x.size(0),-1).min().view(-1,1,1,1)
    x /= x.view(x.size(0),-1).max().view(-1,1,1,1) 
    x = x*255
    z = th.randn(4,3,40,40).type(dtype)
    z -= z.view(z.size(0),-1).min().view(-1,1,1,1)
    z /= z.view(z.size(0),-1).max().view(-1,1,1,1) 
    z = z*255    
    alpha  = th.Tensor(np.random.randint(0,3,(1,))).type(dtype)
    stdn = th.Tensor(np.random.randint(5,20,(4,1))).type(dtype)
    
    
    sz_x = x.size()
    grad_output = th.randn_like(x)
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = x_numgrad.clone() 
    cost = lambda input: cost_l2Prox(input,z,alpha,stdn,grad_output)
        
    for k in range(0,x.numel()):
        perturb[k]  = epsilon
        loss1 = cost(x.view(-1).add(perturb).view(sz_x))
        loss2 = cost(x.view(-1).add(-perturb).view(sz_x))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(sz_x)
    
    sz_alpha = alpha.size()
    alpha_numgrad = th.zeros_like(alpha).view(-1)
    perturb = alpha_numgrad.clone()
    cost = lambda input : cost_l2Prox(x,z,input,stdn,grad_output)
    
    for k in range(0,alpha.numel()):
        perturb[k]  = epsilon
        loss1 = cost(alpha.view(-1).add(perturb).view(sz_alpha))
        loss2 = cost(alpha.view(-1).add(-perturb).view(sz_alpha))
        alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0    
    
    alpha_numgrad = alpha_numgrad.view(sz_alpha)
    
    x_var = Variable(x,requires_grad = True)
    alpha_var = Variable(alpha,requires_grad = True)
    
    y = l2ProxF(x_var,z,alpha_var,stdn)
    y.backward(grad_output)
    
    err_x = th.norm(x_var.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x_var.grad.data.view(-1) + x_numgrad.view(-1))
            
    err_a = th.norm(alpha_var.grad.data.view(-1) - alpha_numgrad.view(-1))/\
            th.norm(alpha_var.grad.data.view(-1) + alpha_numgrad.view(-1))            
    
    return err_x, x_var.grad.data,     x_numgrad, \
           err_a, alpha_var.grad.data, alpha_numgrad


def cost_l2Prox(x,z,alpha,stdn,weights):
    F = functional.L2Prox.apply
    out = F(x,z,alpha,stdn)
    return out.mul(weights).sum()



