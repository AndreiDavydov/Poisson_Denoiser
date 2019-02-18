import torch as th
from PoisDenoiser.projection_utils import is_in_C as cond, \
                                            Projection as proj

import numpy as np

class PoisProx(th.autograd.Function):
    '''
    Z = PoisProx(X, Y) computes the proximal map layer for the 
    indicator function:

    Z = prox_IC(Y, a_cond){X} = 

    = argmin ||X-Z||^2     
    s.t. 1^T*Z-Y^T*log(Z) <= exp(a_cond)*( 1^T*Y-Y^T*log(Y) + ( # Y>0 )/2 ) 
                                
    = argmin ||Z-X||^2 + i_C(Y){Z}
        Z       

    X, Y, Z are all tensors of size B x C x H x W. a_cond - scalar tensor.
    '''

    @staticmethod
    def forward(ctx, input, noisy, a_cond):

        assert(input.dim() == 4 and noisy.dim() == 4), \
            "Input and other are expected to be 4-D tensors."
        assert(a_cond is None or a_cond.numel() == 1), \
        "a_cond needs to be either None or a tensor of size 1."

        batch_size = input.size(0)

        if a_cond is None:
            a_cond = th.Tensor([0]).type_as(input)

        num_iters = 2 # num_iters == by default. 
                      # how to change it in the class???

        cond_mask = cond(input, noisy, a_cond=a_cond).view(batch_size, 1,1,1)

        if input.min() < 0:
            print('input has negative vals!!!')

        input_proj, root = proj(input, noisy, a_cond=a_cond, num_iters=num_iters) 

        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(input_proj, cond_mask, root, a_cond)

        return input_proj

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_a_cond = None

        input_proj, cond_mask, root, a_cond = ctx.saved_variables

        if ctx.needs_input_grad[0]:
            grad_input = input_proj / root
            grad_input[cond_mask.expand_as(grad_input)] = 1
            grad_input.mul_(grad_output)

        return grad_input, None, None #, grad_a_cond - too difficult to compute! 
                                      # Then one needs to form "newton" and "f" as autograd.Functions
                                      # with backward there.








