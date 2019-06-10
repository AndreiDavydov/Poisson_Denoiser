#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:33:22 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import torch as th
import numpy as np
import math
from pydl import utils

class WeightNormalization(th.autograd.Function):
    
    @staticmethod
    def forward(ctx,input, alpha = None, normalizedWeights=False,\
                zeroMeanWeights=False):
        
        assert(input.dim() == 4), "A 4D input tensor is expected but instead "\
        +"a tensor of %d dimensions was provided."%input.dim()        
        
        batch = input.size(0)              
        out = input.clone()
                
        if not normalizedWeights or alpha is None:
            alpha = th.ones(1).type_as(input)
            a_shape = None
        else:
            a_shape = alpha.shape
            alpha = alpha.exp().view(-1,1,1,1)
        
        ctx.intermediate_results = zeroMeanWeights, normalizedWeights, a_shape

        assert(alpha.numel() == 1 or alpha.numel() == batch), \
            "The 'alpha' param must be a tensor of either size 1 or size %r." \
            %batch                             
                
        if zeroMeanWeights:            
            # Substract the mean value from each in_channels x kH x kW filter
            out = out.add(-out.view(batch,-1).mean(1).view(-1,1,1,1))
                            
        if normalizedWeights:
            # Compute the l2 norm for each in_channels x kH x kW filter
            w_norm = out.view(batch,-1).norm(p=2,dim=1).view(-1,1,1,1)
        else:
            w_norm = th.ones(1).type(out.type())
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(out,alpha,w_norm)
                
        # Normalize the filters so that each one has an l2-norm equal to alpha
        return out.div(w_norm).mul(alpha)
    
    @staticmethod
    def backward(ctx,grad_output):
        
        out, alpha, w_norm = ctx.saved_variables
        zeroMeanWeights, normalizedWeights, a_shape = ctx.intermediate_results
        batch = out.size(0)
        
        grad_input = grad_alpha = None
        
        grad_output = grad_output.mul(alpha).div(w_norm)
                
        if ctx.needs_input_grad[1] and normalizedWeights:
            grad_alpha = grad_output.mul(out).view(batch,-1).sum(1)
            if alpha.numel() == 1:
                grad_alpha = grad_alpha.sum()      
            grad_alpha = grad_alpha.view(a_shape)
                        
        if ctx.needs_input_grad[0]:
            if not normalizedWeights and zeroMeanWeights:
                grad_input = grad_output - grad_output.view(batch,-1).mean(1).view(-1,1,1,1)
            
            if normalizedWeights and not zeroMeanWeights:
                out = out.div(w_norm)
                ip = grad_output.mul(out).view(batch,-1).sum(1)
                grad_input = grad_output - out.mul(ip.view(-1,1,1,1))          
            
            if normalizedWeights and zeroMeanWeights:
                out = out.div(w_norm)
                ip = grad_output.mul(out).view(batch,-1).sum(1)
                grad_input = grad_output - out.mul(ip.view(-1,1,1,1))                
                grad_input -= grad_input.view(batch,-1).mean(1).view(-1,1,1,1)
            
            if not normalizedWeights and not zeroMeanWeights:
                grad_input = grad_output
        
        return grad_input, grad_alpha, None, None

class WeightNormalization5D(th.autograd.Function):
    
    @staticmethod
    def forward(ctx,input, alpha = None, normalizedWeights=False,\
                zeroMeanWeights=False):
        
        assert(input.dim() == 5), "A 5D input tensor is expected but instead "\
        +"a tensor of %d dimensions was provided."%input.dim()        
        batch = input.size(0)*input.size(1)              
        sz = input.shape[2:]
        out = input.view(batch,*sz).clone()
                
        if not normalizedWeights or alpha is None:
            alpha = th.ones(1).type_as(input)
            a_shape = None
        else:
            a_shape = alpha.shape
            alpha = alpha.exp().view(-1,1,1,1)
        
        ctx.intermediate_results = zeroMeanWeights, normalizedWeights, a_shape

        assert(alpha.numel() in (1,batch)),"The 'alpha' param must be a "\
               +"tensor of either size 1 or size %r."%(batch)
               
        if zeroMeanWeights:
            # Substract the mean value from each in_channels x kH x kW filter
            out = out.add(-out.view(batch,-1).mean(1).view(-1,1,1,1))
                            
        if normalizedWeights:
            # Compute the l2 norm for each in_channels x kH x kW filter
            w_norm = out.view(batch,-1).norm(p=2,dim=1).view(-1,1,1,1)
        else:
            w_norm = th.ones(1).type(out.type())
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(out,alpha,w_norm)
                
        # Normalize the filters so that each one has an l2-norm equal to alpha
        out = out.div(w_norm).mul(alpha).view(*input.shape)
        return out
    
    @staticmethod
    def backward(ctx,grad_output):
        out, alpha, w_norm = ctx.saved_variables
        zeroMeanWeights, normalizedWeights, a_shape = ctx.intermediate_results
        batch = grad_output.size(0)*grad_output.size(1)
        shape = grad_output.shape
        
        grad_input = grad_alpha = None
        
        grad_output = grad_output.view(batch,*shape[2:]).mul(alpha).div(w_norm)
                
        if ctx.needs_input_grad[1] and normalizedWeights:
            grad_alpha = grad_output.mul(out).view(batch,-1).sum(1)
            if alpha.numel() == 1:
                grad_alpha = grad_alpha.sum()      
            grad_alpha = grad_alpha.view(a_shape)
                        
        if ctx.needs_input_grad[0]:
            if not normalizedWeights and zeroMeanWeights:
                grad_input = grad_output - grad_output.view(batch,-1).mean(1).view(-1,1,1,1)
            
            if normalizedWeights and not zeroMeanWeights:
                out = out.div(w_norm)
                ip = grad_output.mul(out).view(batch,-1).sum(1)
                grad_input = grad_output - out.mul(ip.view(-1,1,1,1))          
            
            if normalizedWeights and zeroMeanWeights:
                out = out.div(w_norm)
                ip = grad_output.mul(out).view(batch,-1).sum(1)
                grad_input = grad_output - out.mul(ip.view(-1,1,1,1))                
                grad_input -= grad_input.view(batch,-1).mean(1).view(-1,1,1,1)
            
            if not normalizedWeights and not zeroMeanWeights:
                grad_input = grad_output
        
        return grad_input.view(*shape), grad_alpha, None, None
    
class Shift(th.autograd.Function):
    @staticmethod
    def forward(ctx,input,steps,bc="circular"):
        assert(bc in ("circular","zero","reflexive")),"The valid boundary "\
        +"conditions are 'circular'|'zero'|'reflexive'."
        
        if ctx.needs_input_grad[0]:
            ctx.intermediate = steps,bc
        
        return utils.shift(input,steps,bc)
    
    @staticmethod
    def backward(ctx,grad_output):
        
        steps,bc = ctx.intermediate
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            grad_input = utils.shift_transpose(grad_output,steps,bc)
        
        return grad_input,None,None
        

class Pad2D(th.autograd.Function):
    @staticmethod
    def forward(ctx,input,pad,padType = 'zero'):
        assert(input.dim() >= 2), "The dimensions of the input tensor are "\
        +"expected to be larger or equal to 2."
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0]:            
            ctx.intermediate = pad,padType
            
        return utils.pad2D(input,pad,padType)

    @staticmethod
    def backward(ctx,grad_output):
            
        pad, padType = ctx.intermediate
        grad_input = None
            
        if ctx.needs_input_grad[0]:
            grad_input = utils.pad_transpose2D(grad_output,pad,padType)
            
        return grad_input, None, None    

class Pad_transpose2D(th.autograd.Function):
    @staticmethod
    def forward(ctx,input,pad,padType = 'zero'):
        assert(input.dim() >= 2), "The dimensions of the input tensor are "\
        +"expected to be larger or equal to 2."
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0]:
            ctx.intermediate = pad, padType
            
        return utils.pad_transpose2D(input,pad,padType)

    @staticmethod
    def backward(ctx,grad_output):

        pad, padType = ctx.intermediate
        grad_input = None
            
        if ctx.needs_input_grad[0]:
            grad_input = utils.pad2D(grad_output,pad,padType)

        return grad_input, None, None

class ZeroPad2D(th.autograd.Function):
    
    @staticmethod
    def forward(ctx,input,pad):
        assert(input.dim() >= 2), "The dimensions of the input tensor are "\
        +"expected to be larger or equal to 2."
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0]:
            ctx.intermediate = pad,
            
        return utils.zeroPad2D(input,pad) 
        
    @staticmethod
    def backward(ctx,grad_output):
            
        pad, = ctx.intermediate
        grad_input = None
            
        if ctx.needs_input_grad[0]:
            grad_input = utils.crop2D(grad_output,pad)
            
        return grad_input, None
    
class Crop2D(th.autograd.Function):
    
    @staticmethod
    def forward(ctx,input,crop):
        assert(input.dim() >= 2), "The dimensions of the input tensor are "\
        +"expected to be larger or equal to 2."
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0]:
            ctx.intermediate = crop,
            
        return utils.crop2D(input,crop) 
        
    @staticmethod
    def backward(ctx,grad_output):
            
        crop, = ctx.intermediate
        grad_input = None
            
        if ctx.needs_input_grad[0]:
            grad_input = utils.zeroPad2D(grad_output,crop)
            
        return grad_input, None    

class SymmetricPad2D(th.autograd.Function):
    r"""Pads symmetrically the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""
    
    @staticmethod
    def forward(ctx,input,pad):
        assert(input.dim() >= 2), "The dimensions of the input tensor are "\
        +"expected to be larger or equal to 2."
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0]:
            ctx.intermediate = pad,
            
        return utils.symmetricPad2D(input,pad) 
        
    @staticmethod
    def backward(ctx,grad_output):
            
        pad, = ctx.intermediate
        grad_input = None
            
        if ctx.needs_input_grad[0]:
            grad_input = utils.symmetricPad_transpose2D(grad_output,pad)
            
        return grad_input, None

class SymmetricPad_transpose2D(th.autograd.Function):
    r"""Adjoint of the SymmetricPad2D operation which amounts to a special type
    of cropping. CROP specifies the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT].
    If crop is an integer then each direction is cropped by the same amount. In
    order to achieve a different amount of cropping in each direction of the 
    tensor, crop needs to be a tuple."""             

    @staticmethod
    def forward(ctx,input,crop):
        assert(input.dim() >= 2),  "The dimensions of the input tensor are "\
        +"expected to be larger or equal to 2."
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0]:
            ctx.intermediate = crop,
            
        return utils.symmetricPad_transpose2D(input,crop)

    @staticmethod
    def backward(ctx,grad_output):
        
        crop, = ctx.intermediate

        if ctx.needs_input_grad[0]:
            grad_input = utils.symmetricPad2D(grad_output,crop)
        
        return grad_input, None
        
class L2Prox(th.autograd.Function):
    r""" Y = L2PROX(X,Z,ALPHA,STDN) computes the proximal map layer for the 
   indicator function :
       
   Y = prox_IC(Z,EPSILON){X} = argmin ||Y-X||^2     
                              ||(Y-Z)|| <= EPSILON 
                                
                             = argmin ||Y-X||^2 + i_C(Z,EPSILON){Y}
                                  Y       

                      { 0 if ||Y-Z|| <= EPSILON
   i_C(Z,EPSILON){Y}= {
                      { +inf if ||Y-Z|| > EPSILON

   X, Z and Y are tensors of size N x C x H x W, ALPHA is a scalar tensor, 
   STDN is either a scalar tensor or a tensor with N elements and 
   EPSILON = exp(ALPHA)*V*STDN, where V = sqrt(H*W*C-1).
  
   Y = Z + K* (X-Z) where K = EPSILON / max(||X-Z||,EPSILON);

   DLDX, DLDA = L2PROX.backward(DLDY) computes the derivatives of the block 
   projected onto DLDY. DLDX has the same dimensions as X and DLDA the same 
   dimensions as ALPHA.
    
   DLDX = K ( I - (X-Z)*(X-Z)^T/ max(||X-Z||,EPSILON)^2) * R) * DLDY
   
   where R = (sgn(||X-Z||-epsilon)+1)/2

   DLDA = B*(X-Z)^T*DLDY

   where B = [ EPSILON *{ 2*max(||X-Z||,EPSILON)-
   EPSILON*(1-sgn(||X-Z||-EPSILON)) } ] / [ 2*max(||X-Z||,EPSILON)^2 ]"""    

    @staticmethod
    def forward(ctx,input,other,alpha,stdn):
        assert(input.dim() == 4 and other.dim() == 4), \
            "Input and other are expected to be 4-D tensors."
        assert(alpha is None or alpha.numel() == 1), "alpha needs to be "\
        "either None or a tensor of size 1."
                
        N = math.sqrt(input[0].numel()-1)
        batch = input.size(0)
        
        assert(stdn.numel() == 1 or stdn.numel() == batch), \
            "stdn must be either a tensor of size one or a tensor of size "\
            "equal to the batch number."
        assert(all(stdn.view(-1) > 0)), "The noise standard deviations must be positive."
        
        stdn = stdn.view(-1,1,1,1)
        
        if alpha is None:
            alpha = th.Tensor([0]).type_as(stdn)
            
        epsilon = stdn.mul(alpha.exp())*N
        
        diff = input.add(-other)
        diff_norm = diff.view(batch,-1).norm(p=2,dim = 1).view(batch,1,1,1)
        max_norm = diff_norm.max(epsilon)
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(diff,diff_norm,max_norm,epsilon)
        
        return other + diff.mul(epsilon).div(max_norm)
            
    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_alpha = None
        
        diff,diff_norm,max_norm,epsilon = ctx.saved_variables
        batch = grad_output.size(0)

        if ctx.needs_input_grad[2]:
            k = epsilon.div(diff_norm)
            k[k >= 1] = 0
            grad_alpha = grad_output.mul(diff).mul(k).sum().reshape(1)
            
        if ctx.needs_input_grad[0]:
            r = (utils.signum(diff_norm-epsilon)+1)/2
            r = r.div(max_norm.pow(2))
            grad_input = grad_output.mul(epsilon.div(max_norm))
            ip = grad_input.mul(diff).view(batch,-1).sum(1).view(-1,1,1,1)
            ip = ip.mul(r)
            grad_input -= diff.mul(ip)        
                        
        return grad_input,None,grad_alpha,None

class SVL2Prox(th.autograd.Function):
    r""" Y = SVL2PROX(X,Z,S,ALPHA) computes the proximal map layer for the 
   indicator function that involves the covariance matrix S of spatially 
   variant noise:
       
   Y = prox_IC(Z,S,EPSILON){X} = argmin ||Y-X||^2     
                                ||S(Y-Z)|| <= EPSILON 
                                
                               = argmin ||Y-X||^2 + i_C(Z,S,EPSILON){Y}
                                   Y

                        { 0 if ||S(Y-Z)|| <= EPSILON
   i_C(Z,S,EPSILON){Y}= {
                        { +inf if ||S(Y-Z)|| > EPSILON

   X, Z and Y are tensors of size N x C x H x W, ALPHA is a scalar tensor, 
   S is either a scalar tensor, a tensor with N elements or a tensor with
   the same dimensions as X. In any of these cases its values correspond to the
   standard deviation of the noise for every element of the tensor using the
   broadcasting rules.
   EPSILON = exp(ALPHA)*V, where V = sqrt(H*W*C-1).
  
   Y = Z + K* (X-Z) where K = EPSILON / max(||S(X-Z)||,EPSILON);

   DLDX, DLDA = SVL2PROX.backward(DLDY) computes the derivatives of the block 
   projected onto DLDY. DLDX has the same dimensions as X and DLDA the same 
   dimensions as ALPHA.
    
   DLDX = K ( I - S(S(X-Z)*(X-Z)^T/ max(||S(X-Z)||,EPSILON)^2) * R) * DLDY
   
   where R = (sgn(||S(X-Z)||-epsilon)+1)/2

   DLDA = B*(X-Z)^T*DLDY

   where B = [ EPSILON *{ 2*max(||S(X-Z)||,EPSILON)-
   EPSILON*(1-sgn(||S(X-Z)||-EPSILON)) } ] / [ 2*max(||S(X-Z)||,EPSILON)^2 ]"""    

    @staticmethod
    def forward(ctx,input,other,alpha,stdn):
        assert(input.dim() == 4 and other.dim() == 4), \
            "Input and other are expected to be 4-D tensors."
        assert(alpha is None or alpha.numel() == 1), "alpha needs to be "\
        "either None or a tensor of size 1."
                
        N = math.sqrt(input[0].numel()-1)
        batch = input.size(0)
        
        assert(stdn.numel() == 1 or stdn.numel() == batch or stdn.size() == input.size()), \
            "stdn must be either a tensor of size one or a tensor of size "\
            "equal to the batch number or a tensor of size equal to the size "\
            "of the first input argument."
        assert(all(stdn.view(-1) > 0)), "The noise standard deviations must be positive."
        
        if stdn.size() != input.size():
            stdn = stdn.view(-1,1,1,1)
        
        if alpha is None:
            alpha = th.Tensor([0]).type_as(stdn)
            
        epsilon = alpha.exp().mul(N).view(1,1,1,1)
        
        diff = input.add(-other)
        diff_norm = diff.div(stdn).view(batch,-1).norm(p=2,dim=1).view(batch,1,1,1)
        max_norm = diff_norm.max(epsilon)
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(diff,diff_norm,max_norm,epsilon,stdn)
        
        return other + diff.mul(epsilon).div(max_norm)
            
    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_alpha = None
        
        diff,diff_norm,max_norm,epsilon,stdn = ctx.saved_variables
        batch = grad_output.size(0)

        if ctx.needs_input_grad[2]:
            k = epsilon.div(diff_norm)
            k[k >= 1] = 0
            grad_alpha = grad_output.mul(diff).mul(k).sum()
            
        if ctx.needs_input_grad[0]:
            r = (utils.signum(diff_norm-epsilon)+1)/2
            r = r.div(max_norm.pow(2))
            grad_input = grad_output.mul(epsilon.div(max_norm))
            ip = grad_input.mul(diff).view(batch,-1).sum(1).view(-1,1,1,1)
            ip = ip.mul(r)
            grad_input -= diff.mul(ip).div(stdn.pow(2))         
                        
        return grad_input,None,grad_alpha,None


class L2Proj(th.autograd.Function):
    r""" Y = L2PROJ(X,STDN,ALPHA) computes the projection layer for the 
   indicator function :

   Y = proj(EPSILON){X} = argmin ||Y-X||^2     
                           ||Y|| <= EPSILON 
                                
                          = argmin ||Y-X||^2 + i_C(EPSILON){Y}
                              Y       

                      { 0 if ||Y|| <= EPSILON
   i_C(EPSILON){Y}=   {
                      { +inf if ||Y|| > EPSILON

   X and Y are tensors of size N x C x H x W, ALPHA is a scalar tensor, 
   STDN is either a scalar tensor or a tensor with N elements and 
   EPSILON = exp(ALPHA)*V*STDN, where V = sqrt(H*W*C-1).
  
   Y = K*X where K = EPSILON / max(||X||,EPSILON);

   DLDX,DLDA = L2PROJ.backward(DLDY) computes the derivatives of the block 
   projected onto DLDY. DLDX has the same dimensions as X and DLDA the same
   dimensions as ALPHA.
    
   DLDX = K ( I - X*X^T/ max(||X||,EPSILON)^2) * R) * DLDY
   
   where R = (sgn(||X||-epsilon)+1)/2

   DLDA = B*X^T*DLDY

   where B = [ EPSILON *{ 2*max(||X||,EPSILON)-
   EPSILON*(1-sgn(||X||-EPSILON)) } ] / [ 2*max(||X||,EPSILON)^2 ]"""    

    @staticmethod
    def forward(ctx,input,alpha,stdn):
        assert(input.dim() == 4), "Input is expected to be a 4-D tensor."
        assert(alpha is None or alpha.numel() == 1), "alpha needs to be "\
        "either None or a tensor of size 1."
                
        N = math.sqrt(input[0].numel()-1)
        batch = input.size(0)
               
        assert(stdn.numel() == 1 or stdn.numel() == batch), \
            "stdn must be either a tensor of size one or a tensor of size "\
            "equal to the batch number."
        assert(all(stdn.view(-1) > 0)), "The noise standard deviations must be positive."
        
        stdn = stdn.view(-1,1,1,1)
        
        if alpha is None:
            alpha = th.Tensor([0]).type_as(stdn)
        
        epsilon = stdn.mul(alpha.exp())*N
        
        input_norm = input.contiguous().view(batch,-1).norm(p=2,dim=1).view(batch,1,1,1)
        max_norm = input_norm.max(epsilon)
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(input,input_norm,max_norm,epsilon)
        
        return input.mul(epsilon).div(max_norm)
    
    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_alpha = None
        
        batch = grad_output.size(0)
        input,input_norm,max_norm,epsilon = ctx.saved_variables

        if ctx.needs_input_grad[1]:
            k = epsilon.div(input_norm)
            k[k >= 1] = 0
            grad_alpha = grad_output.mul(input).mul(k).sum()
            
        if ctx.needs_input_grad[0]:
            r = (utils.signum(input_norm-epsilon)+1)/2
            r = r.div(max_norm.pow(2))
            grad_input = grad_output.mul(epsilon.div(max_norm))
            ip = grad_input.mul(input).view(batch,-1).sum(1).view(-1,1,1,1)
            ip = ip.mul(r)
            grad_input -= input.mul(ip)         
                        
        return grad_input,grad_alpha,None


class SVL2Proj(th.autograd.Function):
    r""" Y = SVL2PROJ(X,S,ALPHA) computes the L2 projection that involves the 
    covariance matrix S of spatially variant noise:
       
   Y = proj(S,EPSILON){X} = argmin ||Y-X||^2     
                             ||S(Y)|| <= EPSILON 
                                
                          = argmin ||Y-X||^2 + i_C(S,EPSILON){Y}
                              Y

                        { 0 if ||S(Y)|| <= EPSILON
   i_C(Z,S,EPSILON){Y}= {
                        { +inf if ||S(Y)|| > EPSILON

   X and Y are tensors of size N x C x H x W, ALPHA is a scalar tensor, 
   S is either a scalar tensor, a tensor with N elements or a tensor with
   the same dimensions as X. In any of these cases its values correspond to the
   standard deviation of the noise for every element of the tensor using the
   broadcasting rules.
   EPSILON = exp(ALPHA)*V, where V = sqrt(H*W*C-1).
  
   Y = K*X where K = EPSILON / max(||S(X)||,EPSILON);

   DLDX, DLDA = SVL2PROJ.backward(DLDY) computes the derivatives of the block 
   projected onto DLDY. DLDX has the same dimensions as X and DLDA the same 
   dimensions as ALPHA.
    
   DLDX = K ( I - S(S(X)*(X)^T/ max(||S(X)||,EPSILON)^2) * R) * DLDY
   
   where R = (sgn(||S(X)||-epsilon)+1)/2

   DLDA = B*(X)^T*DLDY

   where B = [ EPSILON *{ 2*max(||S(X)||,EPSILON)-
   EPSILON*(1-sgn(||S(X)||-EPSILON)) } ] / [ 2*max(||S(X)||,EPSILON)^2 ]"""    

    @staticmethod
    def forward(ctx,input,alpha,stdn):
        assert(input.dim() == 4),"Input is expected to be a 4-D tensor."
        assert(alpha is None or alpha.numel() == 1), "alpha needs to be "\
        "either None or a tensor of size 1."
                
        N = math.sqrt(input[0].numel()-1)
        batch = input.size(0)
        
        assert(stdn.numel() == 1 or stdn.numel() == batch or stdn.size() == input.size()), \
            "stdn must be either a tensor of size one or a tensor of size "\
            "equal to the batch number or a tensor of size equal to the size "\
            "of the first input argument."
        assert(all(stdn.view(-1) > 0)), "The noise standard deviations must be positive."
        
        if stdn.size() != input.size():
            stdn = stdn.view(-1,1,1,1)
        
        if alpha is None:
            alpha = th.Tensor([0]).type_as(stdn)
            
        epsilon = alpha.exp().mul(N).view(1,1,1,1)
        
        input_norm = input.div(stdn).view(batch,-1).norm(p=2,dim=1).view(batch,1,1,1)
        max_norm = input_norm.max(epsilon)
        
        # No need to save the variables during inference
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(input,input_norm,max_norm,epsilon,stdn)
        
        return input.mul(epsilon).div(max_norm)
            
    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_alpha = None
        
        input,input_norm,max_norm,epsilon,stdn = ctx.saved_variables
        batch = grad_output.size(0)

        if ctx.needs_input_grad[1]:
            k = epsilon.div(max_norm)
            k[k >= 1] = 0
            grad_alpha = grad_output.mul(input).mul(k).sum()
            
        if ctx.needs_input_grad[0]:
            r = (utils.signum(input_norm-epsilon)+1)/2
            r = r.div(max_norm.pow(2))
            grad_input = grad_output.mul(epsilon.div(max_norm))
            ip = grad_input.mul(input).view(batch,-1).sum(1).view(-1,1,1,1)
            ip = ip.mul(r)
            grad_input -= input.mul(ip).div(stdn.pow(2))         
                        
        return grad_input,grad_alpha,None

class mseLoss(th.autograd.Function):
    @staticmethod
    def forward(ctx,input,target,grad=False,mode="normal"):
        r"""Y = MSELOSS(X, Xgt) computes the loss incurred by the estimated
        images X given the ground-truth images Xgt.

        The estimated images X and the ground-truth images Xgt are organised as 
        tensors of dimensions B x C x H x W. The first dimension is the batch size, 
        C is the number of image channels and H and W, are the spatial dimensions 
        and correspond to the height and width of the image.
        
        If grad = False, then the loss function is defined as the MSE of the 
        image intensities:
            
             B
        L = Sum ||X_b-Xgt_b||^2/(B*C*H*W),
            b=1
        
        while if grad = True, then the loss function is defined as the 
        combination of the MSE of the image intensities and the MSE of the 
        gradient coefficients, i.e.,
        
             B
        L = Sum (||X_b-Xgt_b||^2 + ||G(X_b-Xgt_b)||^2 )/(B*C*H*W),
            b=1
        
        where G denotes the gradient operator. The purpose of the second term 
        is to enforce that the edges of the reconstructed image are close to 
        those of the ground-truth. 
        
        If mode != normal then while the training uses the MSE loss, in screen
        the PSNR is printed.        
        """
        assert(input.shape == target.shape), "The tensor inputs must be "\
        "of the same size."
        
        assert(input.dim() <= 4), "Tensor must be at maximum of 4 dimensions."
        
        while input.dim() < 4:
            input = input.unsqueeze(0)
        
        while target.dim() < 4:
            target = target.unsqueeze(0)        
    
        err = input-target
        
        if ctx.needs_input_grad[0] :
            ctx.save_for_backward(err)
            ctx.intermediate_results = grad, 
        
        if mode == "normal":
            loss = err.norm(p=2).pow(2).div(err.numel())
            if grad:
                loss += utils.imGrad(err,bc='reflexive').norm(p=2).pow(2).div(err.numel())
        else:
            N = err[0].numel()
            batch = err.size(0)
            normE = err.view(batch,-1).norm(p=2,dim=1)
            M = target.view(batch,-1).max(dim=1)[0]
            loss = 20*th.log10(M*math.sqrt(N)/normE)
            loss = loss.mean()
        
        return loss   

    @staticmethod
    def backward(ctx,grad_output):
        
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            err, = ctx.saved_variables
            grad, = ctx.intermediate_results
            grad_input = err
            if grad :
                grad_input += utils.imDivergence(utils.imGrad(err,bc='reflexive'))
            
            grad_input = 2*grad_input.mul(grad_output).div(err.numel())
        
        return grad_input,None,None,None
        

class imLoss(th.autograd.Function) :
    r"""  Y = IMLOSS(X, Xgt) computes the loss incurred by the estimated
    images X given the ground-truth images Xgt.

    The estimated images X and the ground-truth images Xgt are organised as 
    tensors of dimensions B x C x H x W. The first dimension is the batch size, 
    C is the number of image channels and H and W, are the spatial dimensions 
    and correspond to the height and width of the image.

    If loss_type = 'psnr'

    The loss function is defined as the negative psnr:
      B
    -Sum 20*log10(R*sqrt(K)/norm(X_n-Xgt_n)),
     n=1

    If loss_type = 'l1'

    The loss function is defined as:
      B
     Sum abs(X_n-Xgt_n)/K, 
     n=1

    where K=pixels x channels (number of spatial pixels times image
    channels) and R is the maximum pixel intensity level.

    --  Optional arguments --
    
    peakVal:: 255
      Allows to define a different maximum pixel intensity level.

    loss:: 'psnr' | 'l1'
    
    mode:: 'normal' | 'validation'   
      In the validation mode if loss = 'l1' then the  psnr value is printed 
      but the l1-norm is minimized."""
    
    @staticmethod
    def forward(ctx,input,other,peakVal=255,loss='psnr',mode='normal'):
        
        assert(input.shape == other.shape), "The tensor inputs must be "\
        "of the same size."
        
        assert(input.dim() <= 4), "Tensor must be at maximum of 4 dimensions."
        
        while input.dim() < 4:
            input = input.unsqueeze(0)
        
        while other.dim() < 4:
            other = other.unsqueeze(0)
        
        batch = input.size(0)            
        N = input[0].numel()
        err = input - other
        
        if peakVal is None:
            peakVal = other.view(batch,-1).max(dim=1)[0]        
        
        ctx.save_for_backward(err)
        ctx.intermediate_results = peakVal,loss
        
        if loss == 'psnr' or (loss == 'l1' and mode == 'validation'):
            normE = err.view(batch,-1).norm(p=2,dim=1)
            output = (-20/math.log(10))*th.log(peakVal*math.sqrt(N)/normE)
            output = output.sum()
        elif loss == 'l1' and mode == 'normal':
            output = err.view(-1,1).abs().sum().div(N)
        else:
            raise NotImplementedError("Unknown loss function.")
        
        return output
    
    @staticmethod
    def backward(ctx,grad_output):
        
        peakVal, loss = ctx.intermediate_results
        err, = ctx.saved_variables
        
        batch = err.size(0)
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            if loss == 'psnr':
                normE = err.view(batch,-1).norm(p=2,dim=1).view(batch,1,1,1)
                grad_input = grad_output.mul(err.div(normE.pow(2)))
                grad_input *= 20/math.log(10) 
            elif loss == 'l1':
                grad_input = grad_output.mul(th.sign(err)).div(err[0].numel())
            else:
                raise NotImplementedError("Unknown loss function.")
        
        return grad_input, None, None, None, None


class Grbf_lut(th.autograd.Function):
    r""" Interpolation using an RBF-mixture with truncated Gaussian basis
    functions. (To compute the GRBF we use a look-up table and perform linear
    interpolation for the values that do not coincide with the saved ones).
 
    If X is of size B x K x H x W  (K: number of channels, B: number of
    images) then weights is of size K x M where M is the number
    of mixture components in the RBF-mixture, means is of size M x 1 and
    precision is a scalar.

                M-1
    r(z_(n,k)) = S  w_(k,j)*exp(-0.5*(|z_(n,k)-mu_j|/sigma)^2)
                j=0

    where n=0:B*H*W refers to the spatial coordinates of X, k=0:K refers
    to the channels of X. (w=weights, mu=means).
    """
    @staticmethod
    def forward(ctx,input,weights,centers,sigma,origin=-104,step=0.1):
        
        while input.dim() < 4:
            input = input.unsqueeze(0)
        
        assert(weights.size(0) == input.size(1)), "dimensions mismatch "\
        "between weights and input."
        assert(weights.size(1) == centers.numel()), "dimensions mismatch "\
        "between weights and rbf centers."
        
        # *** Prepare the look-up table ***
        centers = centers.type_as(input).squeeze().unsqueeze(0)
        data_samples = th.range(origin,float(centers[0,-1])+sigma,step)
        data_samples = data_samples.type_as(input).unsqueeze(1)
        data_samples = (data_samples - centers)/sigma
        nbins = data_samples.size(0)
        
        L = th.zeros(weights.size(0),nbins).type_as(input)   

        if ctx.needs_input_grad[0]:
            L_der = L.clone()
            
        for k in range(0,weights.size(0)):
            tmp = th.exp(-0.5*data_samples.pow(2)).mul(weights[k,:].unsqueeze(0))
            L[k,:] = tmp.sum(1)
            if ctx.needs_input_grad[0]:
                L_der[k,:] = th.sum(-tmp.mul(data_samples).div(sigma),1)
        # *** End of computation for the look-up table ***

        # Linear interpolation of f(x) for x that lies in [x_k,x_k+1].
        # f(x) = f(x_k) + (f(x_k+1)-f(x_k))*(x-x_k)/(x_k+1-x_k) (1)
        # We use that our interval [x_0, x_N] is split into N bins where the 
        # centers of the bins are equal to x_k = x_0 + k*step and x_0 is the 
        # origin. Based on this, it holds that x_k+1-x_k = step and we write 
        # (1) as : f(x) = f(x_k)*(1-a) + f(x_k+1)*a where a = (x - x0)/step - k
        # According to this equivalent formulation we can find the weights a
        # by subtracting the input value from the origin and normalizing it by 
        # the step, and then subtracting the integer k which indicates that x 
        # lies between the bin centers x_k and x_k+1
        
        ndfo = (input - origin)/step # normalized distance from origin
        
        # The valid values of the bins are in the range [0, nbins-1]. Values 
        # that fall outside of the interval [x_0 x_N] are mapped to the 
        # boundaries, i.e if x < x_0 then x = x_0 and if x > x_N then x = x_N
        
        # The index of the left bin that encloses input
        lbin_idx = th.min(th.max(th.floor(ndfo),th.zeros(1).type_as(ndfo)),\
                          th.ones(1).type_as(ndfo).mul(nbins-1))
        # The index of the right bin that encloses input
        rbin_idx = th.min(th.max(th.ceil(ndfo),th.zeros(1).type_as(ndfo)),\
                          th.Tensor([nbins-1]).type_as(ndfo))

        a = ndfo - lbin_idx

        # b = 0:B, k = 0:K, h = 0:H, w = 0:W, n = 0:N
        # y[b,k,h,w] = L[k,lbin_idx]*(1-a) + L[k,rbin_idx]*a
        index  = th.zeros_like(input).long()
        for i in range(input.shape[1]):
            index[:,i,...] = i*nbins
            
        y = L.take(index+lbin_idx.long()).mul(1-a) + L.take(index+rbin_idx.long()).mul(a)
                
        ctx.intermediate_results = tuple()
        
        if ctx.needs_input_grad[1]:
            ctx.intermediate_results = ctx.intermediate_results + (centers,sigma)
            ctx.save_for_backward(input)
        
        if ctx.needs_input_grad[0]:
            J = L_der.take(index+lbin_idx.long()).mul(1-a) + L_der.take(index+rbin_idx.long()).mul(a)
            ctx.intermediate_results = ctx.intermediate_results + (J,)        
        
        return y
    
    @staticmethod
    def backward(ctx,grad_output):
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            centers, sigma, J = ctx.intermediate_results
        elif ctx.needs_input_grad[0]:
            J, = ctx.intermediate_results
        elif ctx.needs_input_grad[1]:
            centers, sigma = ctx.intermediate_results

        if ctx.needs_input_grad[1]:
            input, = ctx.saved_variables
            
        grad_input = grad_weights = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mul(J)
        
        # Vectorized computation but requires much more memory and it is 
        # significantly slower.
     
#        if ctx.Cneeds_input_grad[1]:
#            K = input.size(1)
#            input = input.permute(1,2,3,0).contiguous().view(K,-1).unsqueeze(2) - centers.unsqueeze(0)
#            input = th.exp(-0.5*input.div(sigma).pow(2))
#            grad_weights = input.mul(grad_output.permute(1,2,3,0).contiguous().view(K,-1).unsqueeze(2)).sum(1)
        
        if ctx.needs_input_grad[1]:
            grad_weights = th.zeros(input.size(1),centers.numel()).type_as(input) 
            for m in range(0,centers.numel()):
                tmp = th.exp(-0.5*(input.add(-centers[0,m]).div(sigma).pow(2)))
                grad_weights[:,m] = tmp.mul(grad_output).sum(dim=0).sum(dim=1).sum(dim=1)
            
        return grad_input, grad_weights, None, None, None, None


class grbf_LUT(th.autograd.Function):
    r""" Interpolation using an RBF-mixture with truncated Gaussian basis
    functions. (To compute the GRBF we use a look-up table and perform linear
    interpolation for the values that do not coincide with the saved ones).
 
    If X is of size B x K x H x W  (K: number of channels, B: number of
    images) then weights is of size K x M where M is the number
    of mixture components in the RBF-mixture, means is of size M x 1 and
    precision is a scalar.

                M-1
    r(z_(n,k)) = S  w_(k,j)*exp(-0.5*(|z_(n,k)-mu_j|/sigma)^2)
                j=0

    where n=0:B*H*W refers to the spatial coordinates of X, k=0:K refers
    to the channels of X. (w=weights, mu=means).
    """
    @staticmethod
    def forward(ctx,input,weights,centers,sigma,data_lut):
        
        while input.dim() < 4:
            input = input.unsqueeze(0)
        
        assert(weights.size(0) == input.size(1)), "dimensions mismatch "\
        "between weights and input."
        assert(weights.size(1) == centers.numel()), "dimensions mismatch "\
        "between weights and rbf centers."
        
        assert(data_lut.size(1) == centers.numel()), "dimensions mismatch "\
        "between data_lut and rbf centers."
        
        
        # *** Prepare the look-up table ***
        
        # data_lut is a 2D tensor of nbins x centers. Suppose that we are 
        # interested in the interval [start end] where we sample it in a uniform 
        # way so that data = th.range(start,end,step). Then data_lut is 
        # created as data_lut[:,k] = (data-centers[k])/sigma for k = 0:M
        # where M is the total number of rbf centers.        
        centers = centers.type_as(input).squeeze().unsqueeze(0)
        step = float(sigma*(data_lut[1,0]-data_lut[0,0]))
        origin = float(data_lut[0,0]*sigma+centers[0,0])
        nbins = data_lut.size(0)
        
        L = th.zeros(weights.size(0),nbins).type_as(input)   

        if ctx.needs_input_grad[0]:
            L_der = L.clone()
            
        for k in range(0,weights.size(0)):
            tmp = th.exp(-0.5*data_lut.pow(2)).mul(weights[k,:].unsqueeze(0))
            L[k,:] = tmp.sum(1)
            if ctx.needs_input_grad[0]:
                L_der[k,:] = th.sum(-tmp.mul(data_lut).div(sigma),1)
        # *** End of computation for the look-up table ***

        # Linear interpolation of f(x) for x that lies in [x_k,x_k+1].
        # f(x) = f(x_k) + (f(x_k+1)-f(x_k))*(x-x_k)/(x_k+1-x_k) (1)
        # We use that our interval [x_0, x_N] is split into N bins where the 
        # centers of the bins are equal to x_k = x_0 + k*step and x_0 is the 
        # origin. Based on this, it holds that x_k+1-x_k = step and we write 
        # (1) as : f(x) = f(x_k)*(1-a) + f(x_k+1)*a where a = (x - x0)/step - k
        # According to this equivalent formulation we can find the weights a
        # by subtracting the input value from the origin and normalizing it by 
        # the step, and then subtracting the integer k which indicates that x 
        # lies between the bin centers x_k and x_k+1
        
        ndfo = (input - origin)/step # normalized distance from origin
        
        # The valid values of the bins are in the range [0, nbins-1]. Values 
        # that fall outside of the interval [x_0 x_N] are mapped to the 
        # boundaries, i.e if x < x_0 then x = x_0 and if x > x_N then x = x_N
        
        # The index of the left bin that encloses input
        lbin_idx = th.min(th.max(th.floor(ndfo),th.zeros(1).type_as(ndfo)),\
                          th.ones(1).type_as(ndfo).mul(nbins-1))
        # The index of the right bin that encloses input
        rbin_idx = th.min(th.max(th.ceil(ndfo),th.zeros(1).type_as(ndfo)),\
                          th.Tensor([nbins-1]).type_as(ndfo))

        a = ndfo - lbin_idx

        # b = 0:B, k = 0:K, h = 0:H, w = 0:W, n = 0:N
        # y[b,k,h,w] = L[k,lbin_idx]*(1-a) + L[k,rbin_idx]*a
        index  = th.zeros_like(input).long()
        for i in range(input.shape[1]):
            index[:,i,...] = i*nbins
            
        y = L.take(index+lbin_idx.long()).mul(1-a) + L.take(index+rbin_idx.long()).mul(a)
                
        ctx.intermediate_results = tuple()
        
        if ctx.needs_input_grad[1]:
            ctx.intermediate_results = ctx.intermediate_results + (centers,sigma)
            ctx.save_for_backward(input)
        
        if ctx.needs_input_grad[0]:
            J = L_der.take(index+lbin_idx.long()).mul(1-a) + L_der.take(index+rbin_idx.long()).mul(a)
            ctx.intermediate_results = ctx.intermediate_results + (J,)        
        
        return y

    @staticmethod
    def backward(ctx,grad_output):

        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            centers, sigma, J = ctx.intermediate_results
        elif ctx.needs_input_grad[0]:
            J, = ctx.intermediate_results
        elif ctx.needs_input_grad[1]:
            centers, sigma = ctx.intermediate_results

        if ctx.needs_input_grad[1]:
            input, = ctx.saved_variables
            
        grad_input = grad_weights = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mul(J)
        
        # Vectorized computation but requires much more memory and it is 
        # significantly slower.
     
#        if ctx.needs_input_grad[1]:
#            K = input.size(1)
#            input = input.permute(1,2,3,0).contiguous().view(K,-1).unsqueeze(2) - centers.unsqueeze(0)
#            input = th.exp(-0.5*input.div(sigma).pow(2))
#            grad_weights = input.mul(grad_output.permute(1,2,3,0).contiguous().view(K,-1).unsqueeze(2)).sum(1)
        
        if ctx.needs_input_grad[1]:
            grad_weights = th.zeros(input.size(1),centers.numel()).type_as(input) 
            for m in range(0,centers.numel()):
                tmp = th.exp(-0.5*(input.add(-centers[0,m]).div(sigma).pow(2)))
                grad_weights[:,m] = tmp.mul(grad_output).sum(dim=0).sum(dim=1).sum(dim=1) 

        return grad_input, grad_weights, None, None, None

class WienerFilter(th.autograd.Function):
    
    @staticmethod
    def forward(ctx,input,blurKernel,weights,alpha):
        r"""Multi Multichannel Deconvolution Wiener Filter for a batch of input
        images. (Filtering is taking place in the Frequency domain under the 
                 assumption of periodic boundary conditions for the input image.)
    
        input :: tensor of size batch x channels x height x width.
        blurKernel :: tensor of size [1 | batch] x [1 | channels] x b_height x b_width
        weights :: tensor of size [1 | N] x D x [1 | channels] x w_height x w_width
        alpha :: tensor of size N x [1 | channels].
        
        N is the number of different Wiener filters applied on the input
        
        For the weights and alpha parameters the notation [1 | C] means that
        the specified dimension of the corresponding tensor can either have a 
        single or C elements. In case that a single element is used then this 
        element is shared across the dimension. For the input parameters where 
        their leading dimensions can have 1 element, these dimensions can be 
        omitted.
        
        WienerFilter returns the output and a normalization constant (c) to be 
        used for computing the standard deviation of the colored noise.
        
        output : batch x N x channels x height x width
            
            output = F^H (B^H*F(input)/(|B|^2+exp(alpha)*|W|^2))
            
        c : batch x N 
        
            c = trace(G^T*G)/N where G = F^H (B^H/(|B|^2+exp(alpha)*|W|^2)) F
        
        If y = Hx+n then output = Wy = WHx + Wn  where W is the linear Wiener 
        filter. In this case we need to compute the variance of the noise 
        n_color = Wn.
        
        var(n_color)=E(n^T*W^T*W*n) = E(n^T*n)*Trace(W^T*W)/N where N is the 
        total number of spatial elements in y. Then we have that 
        var(n_color) = var(n)*c where c = Trace(W^T*W)/N        
        
        """        
        from pydl.cOps import cmul, cabs, conj      
        
        assert(input.dim() < 5),"The input must be at most a 4D tensor."    
        while input.dim() < 4:
            input = input.unsqueeze(0)

        batch = input.size(0)
        channels = input.size(1)

        assert(blurKernel.dim() < 5),"The blurring kernel must be at most a 4D tensor."
        while blurKernel.dim() < 4:
            blurKernel = blurKernel.unsqueeze(0)
    
        bshape = tuple(blurKernel.shape)
        assert(bshape[0] in (1,batch) and bshape[1] in (1,channels)),"Invalid blurring kernel dimensions."
        
        N = alpha.size(0) # Number of employed Wiener filters.
        assert(alpha.dim() == 2 and alpha.size(-1) in (1,channels)),\
        "Invalid dimensions for the alpha parameter. The expected shape of the "\
        +"tensor is {} x [{}|{}]".format(N,1,channels)
        alpha = alpha.exp()        
        
        assert(weights.dim() > 3 and weights.dim() < 6),"The regularization "\
        +"kernel must be a 4D or 5D tensor."    

        if weights.dim() < 5:
            weights = weights.unsqueeze(0)    
    
        wshape = tuple(weights.shape)
        assert(wshape[0] in (1,N) and wshape[2] in (1,channels)),\
        "Invalid regularization kernel dimensions."
                
        # Zero-padding of the blur kernel to match the input size
        B = th.zeros(bshape[0],bshape[1],input.size(2),input.size(3)).type_as(blurKernel)
        B[...,0:bshape[2],0:bshape[3]] = blurKernel
        del blurKernel
        # Circular shift of the zero-padded blur kernel
        bs = tuple(int(i) for i in -(np.asarray(bshape[-2:])//2))
        bs = (0,0) + bs
        B = utils.shift(B,bs,bc='circular')            
        # FFT of B
        B = th.rfft(B,2) # tensor of size batch x channels x height x width x 2
        
        # Zero-padding of the spatial dimensions of the weights to match the input size    
        G = th.zeros(wshape[0],wshape[1],wshape[2],input.size(2),input.size(3)).type_as(weights)
        G[...,0:wshape[3],0:wshape[4]] = weights
        del weights
        # circular shift of the zero-padded weights
        ws = tuple(int(i) for i in -(np.asarray(wshape[-2:])//2))
        ws = (0,0,0) + ws
        G = utils.shift(G,ws,bc='circular')    
        # FFT of G
        G = th.rfft(G,2) # N x D x channels x height x width x 2
        
        Y = cmul(conj(B),th.rfft(input,2)).unsqueeze(1) # batch x 1 x channels x height x width x 2
        
        ctx.intermediate_results = tuple()
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            ctx.intermediate_results += (alpha,B,G,Y,wshape)
        elif ctx.needs_input_grad[0]:
            ctx.intermediate_results += (alpha,B,G)
        
        B = cabs(B).unsqueeze(-1) # batch x channels x height x width x 1
        G = cabs(G).pow(2).sum(dim=1) # N x channels x height x width
        G = G.mul(alpha.unsqueeze(-1).unsqueeze(-1)).unsqueeze(0).unsqueeze(-1) # 1 x N x channels x height x width x 1
        G += B.pow(2).unsqueeze(1) # batch x N x channels x height x width x 1
        S = (B.unsqueeze(1).div(G)).pow(2).squeeze(-1)
        # Sinc S is one-sided Fourier Transform we need to compute the sum of 
        # the Fourier coefficients taking into account the symmetry.
        c = 2*S.sum(dim=-1).sum(dim=-1)-S[...,0].sum(dim=-1)
        if not input.size(-1)%2:
            c -= S[...,-1].sum(dim=-1)
        c = c/(input.size(-1)*input.size(-2))
        return th.irfft(Y.div(G),2,signal_sizes = input.shape[-2:]), c
        # output 1: batch x N x channels x height x width
        # output 2: batch x N x channels 
        
        # Note: If sigma (of size batch) is the noise std of the input image, 
        # then the noise std of the output images will be equal to 
        # th.sqrt(sigma.unsqueeze(-1).pow(2).mul(c.mean(dim=2)))
    
    @staticmethod
    def backward(ctx,grad_output,grad_c = None):                
        from pydl.cOps import cmul, cabs, conj
                
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            alpha,B,G,Y,wshape = ctx.intermediate_results
            channels = Y.size(2)
        elif ctx.needs_input_grad[0]:
            alpha,B,G = ctx.intermediate_results
                
        grad_input = grad_weights = grad_alpha = None        
        
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3] :
            D = cabs(B).pow(2).unsqueeze(1) # batch x 1 x channels x height x width 
            T = cabs(G).pow(2).sum(dim=1).unsqueeze(0) # 1 x N x channels x height x width 
            T = T.mul(alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) # batch x N x channels x height x width             
            D = D + T # batch x N x channels x height x width 
            del T
            D = D.unsqueeze(-1) # batch x N x channels x height x width x 1
            
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            Z = th.rfft(grad_output,2) # batch x N x channels x height x width x 2
                    
        if ctx.needs_input_grad[0]:
            grad_input = th.irfft(cmul(B.unsqueeze(1),Z).div(D),2,\
                                  signal_sizes=grad_output.shape[-2:])
            grad_input = grad_input.sum(dim=1)
        
        if 'B' in locals(): del B        
        if ctx.needs_input_grad[2]:
            ws = tuple(int(i) for i in -(np.asarray(wshape[-2:])//2))
            ws = (0,0,0,0) + ws
            U = cmul(conj(Z),Y.div(D.pow(2))) # batch x N x channels x height x width x 2
            U = U[...,0].unsqueeze(-1).unsqueeze(2) # batch x N x D x channels x height x width x 1
            U = U.mul(G.unsqueeze(0)) # batch x N x D x channels x height x width x 2
            U = th.irfft(U,2,signal_sizes=grad_output.shape[-2:]) # batch x N x D x channels x height x width            
            U = utils.shift_transpose(U,ws,bc='circular')
            U = U[...,0:wshape[3],0:wshape[4]] # batch x N x D x channels x height x width
            grad_weights = -2*U.mul(alpha.unsqueeze(0).unsqueeze(2).unsqueeze(-1).unsqueeze(-1))
            del U
            grad_weights = grad_weights.sum(dim=0)
            if wshape[2] == 1:
                grad_weights = grad_weights.sum(dim=2,keepdim=True)
            if wshape[0] == 1 and alpha.size(0) != 1:
                grad_weights = grad_weights.sum(dim=0)                
        
        if 'Z' in locals(): del Z
        if ctx.needs_input_grad[3]:
            Y = Y.mul(cabs(G).pow(2).sum(dim=1).unsqueeze(0).unsqueeze(-1)) # batch x 1 x channels x height x width x 2
            Y = Y.div(D.pow(2))
            Y = th.irfft(Y,2,signal_sizes=grad_output.shape[-2:]) # batch x 1 x channels x height x width 
            Y = Y.mul(-alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            Y = Y.mul(grad_output)
            grad_alpha = Y.sum(dim=4).sum(dim=3).sum(dim=0)
            if channels != 1 and alpha.size(-1) == 1:
                grad_alpha = grad_alpha.sum(dim=-1,keepdim=True)
        
        return grad_input,None,grad_weights,grad_alpha

class EdgeTaper(th.autograd.Function):
    
    @staticmethod
    def forward(ctx,input,psf):        
        from pydl.cOps import cmul, conj

        assert(input.dim() < 5), "The input is expected to be at most a 4D tensor."
        while input.dim() < 4 :
            input = input.unsqueeze(0)
        
        assert(psf.dim()==2),"Only 2D psfs are accepted."
        
        beta = {}
        
        if psf.size(0) != 1:
            psfProj = psf.sum(dim=1)
            z = th.zeros(input.size(-2)-1).type_as(psf)
            z[0:psf.size(0)] = psfProj
            z = th.rfft(z,1,onesided=True)
            z = th.irfft(cmul(z,conj(z)),1,onesided=True,signal_sizes=(input.size(-2)-1,))
            z = th.cat((z,z[0:1]),dim=0).div(z.max())
            beta['dim0'] = z.unsqueeze(-1)
        
        if psf.size(1) != 1:
            psfProj = psf.sum(dim=0)
            z = th.zeros(input.size(-1)-1).type_as(psf)
            z[0:psf.size(1)] = psfProj
            z = th.rfft(z,1,onesided=True)
            z = th.irfft(cmul(z,conj(z)),1,onesided=True,signal_sizes=(input.size(-1)-1,))
            z = th.cat((z,z[0:1]),dim=0).div(z.max())
            beta['dim1'] = z.unsqueeze(0)
    
        if len(beta.keys()) == 1:
            alpha = 1 - beta[list(beta.keys())[0]]
        else:
            alpha = (1-beta['dim0'])*(1-beta['dim1'])
        
        while alpha.dim() < input.dim():
            alpha = alpha.unsqueeze(0)
                
        otf = utils.psf2otf(psf,input.shape)
        
        blurred_input = th.irfft(cmul(th.rfft(input,2),otf),2,\
                                 signal_sizes = input.shape[-2:])
        
        output = alpha*input + (1-alpha)*blurred_input
                        
        if ctx.needs_input_grad[0]:
            # mask = ((output >= input.min())+(output <= input.max())).eq(2)
            mask = th.__and__(output >= input.min(),output <= input.max())
            ctx.intermediate_results = alpha,otf,mask
        
        return output.clamp(input.min(),input.max())
    
    @staticmethod
    def backward(ctx,grad_output):
        from pydl.cOps import cmul,conj
        
        alpha,otf,mask = ctx.intermediate_results        
        
        grad_input = mask.type_as(grad_output)*grad_output

        grad_input = alpha*grad_input + th.irfft(cmul(th.rfft((1-alpha)\
                *grad_input,2),conj(otf)),2,signal_sizes = grad_input.shape[-2:])
        
        return grad_input, None