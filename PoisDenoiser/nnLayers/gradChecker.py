import torch as th
from PoisDenoiser.nnLayers.functional import *

def poisLikelihoodFunc_gradCheck(noise_param=0.01, epsilon=1e-7, dtype='torch.DoubleTensor'):

    x = th.rand((4, 1, 5,5)).type(dtype)
    z = th.rand((4, 1, 5,5)).type(dtype)
    z = th.distributions.Poisson(z/noise_param).sample()

    grad_output = th.randn(x.size(0)).type_as(x)

    # x grad
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = th.zeros_like(x).view(-1)
    cost = lambda input: cost_poisLikelihoodFunc(input, z, grad_output)
        
    for k in range(0,x.numel()):
        cur_x = float(x.view(-1)[k])
        perturb[k]  = epsilon if epsilon < cur_x else cur_x/2

        loss1 = cost(x.view(-1).add( perturb).view(x.size()))
        loss2 = cost(x.view(-1).add(-perturb).view(x.size()))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(x.size())

    # compute errors
    x.requires_grad = True
    z.requires_grad = False
    
    y = poisLikelihoodFunc(x,z)

    y.backward(grad_output)
    
    err_x = th.norm(x.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x.grad.data.view(-1) + x_numgrad.view(-1))
            
    return err_x, x.grad.data, x_numgrad


def condFunc_gradCheck(noise_param=0.01, epsilon=1e-7, dtype='torch.DoubleTensor'):

    x = th.rand((4, 1, 5,5)).type(dtype)
    z = th.rand((4, 1, 5,5)).type(dtype)
    z = th.distributions.Poisson(z/noise_param).sample()
    

    grad_output = th.rand(x.size(0)).type_as(x)

    # x grad
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = th.zeros_like(x).view(-1)
    cost = lambda input: cost_condFunc(input, z, grad_output)
        
    for k in range(0,x.numel()):
        cur_x = x.view(-1)[k]
        perturb[k]  = epsilon if epsilon < cur_x else cur_x/2
        loss1 = cost(x.view(-1).add( perturb).view(x.size()))
        loss2 = cost(x.view(-1).add(-perturb).view(x.size()))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(x.size())

    # compute errors
    x.requires_grad = True
    z.requires_grad = False
    
    y = condFunc(x,z)
    y.backward(grad_output)
    
    err_x = th.norm(x.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x.grad.data.view(-1) + x_numgrad.view(-1))
            
    return err_x, x.grad.data, x_numgrad


def dCond_dAlphaFunc_gradCheck(noise_param=0.01, epsilon=1e-7, dtype='torch.DoubleTensor'):

    x = th.rand((4, 3, 5,5)).type(dtype)
    z = th.rand((4, 3, 5,5)).type(dtype)
    z = th.distributions.Poisson(z/noise_param).sample()

    alpha = th.rand(x.size(0), 1,1,1).type_as(x)*100

    grad_output = th.rand(x.size(0)).type_as(x)

    # x grad
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = th.zeros_like(x).view(-1)
    cost = lambda input: cost_dCond_dAlphaFunc(input, z, alpha, grad_output)
        
    for k in range(0,x.numel()):
        cur_x = x.view(-1)[k]
        perturb[k]  = epsilon if epsilon < cur_x else cur_x/2
        loss1 = cost(x.view(-1).add( perturb).view(x.size()))
        loss2 = cost(x.view(-1).add(-perturb).view(x.size()))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(x.size())

    # alpha grad
    alpha_numgrad = th.zeros_like(alpha).view(-1)
    perturb = th.zeros_like(alpha).view(-1)
    cost = lambda input: cost_dCond_dAlphaFunc(x, z, input, grad_output)
        
    for k in range(0,alpha.numel()):
        cur_alpha = alpha.view(-1)[k]
        perturb[k]  = epsilon if epsilon < cur_alpha else cur_alpha/2
        loss1 = cost(alpha.view(-1).add( perturb).view(alpha.size()))
        loss2 = cost(alpha.view(-1).add(-perturb).view(alpha.size()))
        alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    alpha_numgrad = alpha_numgrad.view(alpha.size())

    # compute errors
    x.requires_grad = True
    z.requires_grad = False
    alpha.requires_grad = True

    y = dCond_dAlphaFunc(x,z,alpha)
    y.backward(grad_output)

    err_x = th.norm(x.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x.grad.data.view(-1) + x_numgrad.view(-1))

    err_alpha = th.norm(alpha.grad.data.view(-1) - alpha_numgrad.view(-1))/\
                th.norm(alpha.grad.data.view(-1) + alpha_numgrad.view(-1))
            
    return err_x, x.grad.data, x_numgrad,\
           err_alpha, alpha.grad.data, alpha_numgrad


def projFunc_gradCheck(noise_param=0.01, epsilon=1e-4, dtype='torch.DoubleTensor'):

    x = th.rand((4, 3, 5,5)).type(dtype)
    z = th.rand((4, 3, 5,5)).type(dtype)
    z = th.distributions.Poisson(z/noise_param).sample()
    
    alpha = th.rand(x.size(0), 1,1,1).type_as(x)*100

    grad_output = th.randn_like(x).type_as(x)

    # x grad
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = th.zeros_like(x).view(-1)
    cost = lambda input: cost_projFunc(input, z, alpha, grad_output)
        
    for k in range(0,x.numel()):
        cur_x = x.view(-1)[k]
        perturb[k]  = epsilon if epsilon < cur_x else cur_x/2
        loss1 = cost(x.view(-1).add( perturb).view(x.size()))
        loss2 = cost(x.view(-1).add(-perturb).view(x.size()))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(x.size())

    # alpha grad
    alpha_numgrad = th.zeros_like(alpha).view(-1)
    perturb = th.zeros_like(alpha).view(-1)
    cost = lambda input: cost_projFunc(x, z, input, grad_output)
        
    for k in range(0,alpha.numel()):
        cur_alpha = alpha.view(-1)[k]
        perturb[k]  = epsilon if epsilon < cur_alpha else cur_alpha/2
        loss1 = cost(alpha.view(-1).add( perturb).view(alpha.size()))
        loss2 = cost(alpha.view(-1).add(-perturb).view(alpha.size()))
        alpha_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    alpha_numgrad = alpha_numgrad.view(alpha.size())

    # compute errors
    x.requires_grad = True
    z.requires_grad = False
    alpha.requires_grad = True
    
    y = projFunc(x,z,alpha)
    y.backward(grad_output)
    
    err_x = th.norm(x.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x.grad.data.view(-1) + x_numgrad.view(-1))
    
    err_alpha = th.norm(alpha.grad.data.view(-1) - alpha_numgrad.view(-1))/\
                th.norm(alpha.grad.data.view(-1) + alpha_numgrad.view(-1))
            
    return err_x, x.grad.data, x_numgrad,\
           err_alpha, alpha.grad.data, alpha_numgrad


def alphaFunc_gradCheck(noise_param=0.01, epsilon=1e-7, dtype='torch.DoubleTensor'):

    x = th.rand((4, 1, 5,5)).type(dtype)
    z = th.rand((4, 1, 5,5)).type(dtype)
    z = th.distributions.Poisson(z/noise_param).sample()

    x = z.clone() + 1e-8
    

    grad_output = th.rand((x.size(0), 1,1,1)).type_as(x)

    # # x grad
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = th.zeros_like(x).view(-1)
    cost = lambda input: cost_alphaFunc(input, z, grad_output)
        
    for k in range(0,x.numel()):
        cur_x = x.view(-1)[k]
        perturb[k]  = epsilon if epsilon < cur_x else cur_x/2
        loss1 = cost(x.view(-1).add( perturb).view(x.size()))
        loss2 = cost(x.view(-1).add(-perturb).view(x.size()))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(x.size())

    # compute errors
    x.requires_grad = True
    z.requires_grad = False

    y = alphaFunc(x,z)
    y.backward(grad_output)

    err_x = th.norm(x.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x.grad.data.view(-1) + x_numgrad.view(-1))
            
    return err_x, x.grad.data, x_numgrad


def poisProx_gradCheck(noise_param=0.01, epsilon=1e-7, dtype='torch.DoubleTensor'):

    x = th.rand((4, 1, 2,2)).type(dtype)
    z = th.rand((4, 1, 2,2)).type(dtype)
    z = th.distributions.Poisson(z/noise_param).sample()

    x[2:4] = z[2:4].clone()

    grad_output = th.ones_like(x).type_as(x)

    # x grad
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = th.zeros_like(x).view(-1)
    cost = lambda input: cost_poisProx(input, z, grad_output)
        
    for k in range(0,x.numel()):
        cur_x = float(x.view(-1)[k])
        perturb[k]  = epsilon if epsilon < cur_x else cur_x/2

        loss1 = cost(x.view(-1).add( perturb).view(x.size()))
        loss2 = cost(x.view(-1).add(-perturb).view(x.size()))
        
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])

        if th.abs(perturb[k]) < 1e-10:
            # both x and z equal to 0. Losses are equal.
            x_numgrad[k] = 1

        perturb[k] = 0

    x_numgrad = x_numgrad.view(x.size())

    # compute errors
    x.requires_grad = True
    z.requires_grad = False
    
    y = poisProx(x,z)

    y.backward(grad_output)

    err_x = th.norm(x.grad.data.view(-1) - x_numgrad.view(-1))/\
           (th.norm(x.grad.data.view(-1) + x_numgrad.view(-1)))
            
    return err_x, x.grad.data, x_numgrad


from PoisDenoiser.networks.PoisNet.net import PoisNet

def poisNet_gradCheck(noise_param=0.01, epsilon=1e-7, dtype='torch.DoubleTensor'):

    x = th.rand((4, 1, 5,5)).type(dtype).cuda(0)
    z = th.rand((4, 1, 5,5)).type(dtype)
    z = th.distributions.Poisson(z/noise_param).sample().cuda(0)

    model = PoisNet(stages=5, output_features=64).double().cuda(0)

    grad_output = th.randn_like(x).type_as(x).cuda(0)

    # x grad
    x_numgrad = th.zeros_like(x).view(-1)
    perturb = th.zeros_like(x).view(-1)
    cost = lambda input: cost_poisNet(input, z, model, grad_output)
        
    for k in range(0,x.numel()):
        cur_x = float(x.view(-1)[k])
        perturb[k]  = epsilon if epsilon < cur_x else cur_x/2

        loss1 = cost(x.view(-1).add( perturb).view(x.size()))
        loss2 = cost(x.view(-1).add(-perturb).view(x.size()))
        x_numgrad[k] = (loss1-loss2)/(2*perturb[k])
        perturb[k] = 0

    x_numgrad = x_numgrad.view(x.size()).cuda(0)

    # compute errors
    x.requires_grad = True
    z.requires_grad = False
    
    y = model(x,z)

    y.backward(grad_output)
    
    err_x = th.norm(x.grad.data.view(-1) - x_numgrad.view(-1))/\
            th.norm(x.grad.data.view(-1) + x_numgrad.view(-1))
            
    return err_x, x.grad.data, x_numgrad


def cost_poisLikelihoodFunc(x,z, weights):
    out = poisLikelihoodFunc(x,z)
    return out.mul(weights).sum()

def cost_condFunc(x,z, weights):
    out = condFunc(x,z)
    return out.mul(weights).sum()

def cost_dCond_dAlphaFunc(x,z,alpha, weights):
    out = dCond_dAlphaFunc(x,z,alpha)
    return out.mul(weights).sum()

def cost_projFunc(x,z,alpha, weights):
    out = projFunc(x,z,alpha)
    return out.mul(weights).sum()

def cost_alphaFunc(x,z, weights):
    out = alphaFunc(x,z)
    return out.mul(weights).sum()

def cost_poisProx(x,z, weights):
    out = poisProx(x,z)
    return out.mul(weights).sum()

def cost_poisNet(x,z,model, weights):
    out = model(x,z)
    return out.mul(weights).sum()
