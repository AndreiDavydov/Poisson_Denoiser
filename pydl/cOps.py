#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:46:43 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import torch as th

def cmul(input,other):
    r"""Returns the pointwise product of the elements of two complex tensors."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2 and other.size(-1) == 2), "Inputs must be "\
    +"complex tensors (their last dimension should be equal to two)."
    
    #assert(input.shape == other.shape), "Dimensions mismatch between inputs."
    
    real = input[...,0].mul(other[...,0])-input[...,1].mul(other[...,1])
    imag = input[...,0].mul(other[...,1])+input[...,1].mul(other[...,0])
    
    return th.cat((real.unsqueeze(-1),imag.unsqueeze(-1)),dim=-1)

def crmul(input,other):
    r"""Returns the pointwise product of the elements of a complex and a 
    real tensor."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2), "The first input must be a complex tensor "\
    "(its last dimension should be equal to 2) and the second input must be a "\
    "real tensor."
        
    return input.mul(other.unsqueeze(-1).expand(*input.shape))

def cabs(input):
    r"""Returns the pointwise magnitude of the elements of the input complex tensor."""
    
    assert(th.is_tensor(input) and input.size(-1) == 2),"Inputs is expected "\
    +"to be a complex tensor."    
    
    return input.pow(2).sum(dim=-1).sqrt()

def cinv(input):
    r"""Returns the pointwise inverse of the elements of the input complex tensor."""
    
    assert(th.is_tensor(input) and input.size(-1) == 2),"Inputs is expected "\
    +"to be a complex tensor."    
    
    out = crdiv(conj(input),cabs(input).pow(2))
    
    return out

def cdiv(input,other):
    r"""Returns the pointwise division of the elements of two complex tensors."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2 and other.size(-1) == 2), "Inputs must be "\
    +"complex tensors (their last dimension should be equal to two)."
    
    #assert(input.shape == other.shape), "Dimensions mismatch between inputs."
    
    return cmul(input,conj(other)).div((cabs(other)**2).unsqueeze(-1))

def crdiv(input,other):
    r"""Returns the pointwise division of the elements of a complex and a 
    real tensor."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2), "The first input must be a complex tensor "\
    "(its last dimension should be equal to 2) and the second input must be a "\
    "real tensor."
        
    return input.div(other.unsqueeze(-1).expand(*input.shape))

def cradd(input,other):
    r"""Returns the pointwise addition of the elements of a complex and a 
    real tensor."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2), "The first input must be a complex tensor "\
    "(its last dimension should be equal to 2) and the second input must be a "\
    "real tensor."
    
    out = input.clone()
    out[...,0] += other
    return out

def conj(input):
    r"""Returns the complex conjugate of the input complex tensor."""
    
    assert(th.is_tensor(input) and input.size(-1) == 2),"Input is expected "\
    +"to be a complex tensor."
    
    out = input.clone()
    out[...,1] = -out[...,1]
    return out

def conj_(input):
    r"""Returns the complex conjugate of the input complex tensor (inplace operation)."""
    
    assert(th.is_tensor(input) and input.size(-1) == 2),"Input is expected "\
    +"to be a complex tensor."
    
    input[...,1] = - input[...,1]
    
def real(input):
    r"""Returns the real part of the input complex tensor."""
    
    assert(th.is_tensor(input)),"Input is expected to be a tensor."
    
    if input.size(-1) == 2:
        out = input[...,0]
    else:
        out = input.clone()
    
    return out

def imag(input):
    r"""Returns the imaginary part of the input complex tensor."""
    
    assert(th.is_tensor(input)),"Input is expected to be a tensor."
    
    if input.size(-1) == 2:
        out = input[...,1]
    else:
        out = th.zeros_like(input)
    
    return out

def power(input,p):
    mod = cabs(input)
    theta = th.atan2(imag(input),real(input))
    r = mod.pow(p)*th.cos(p*theta)
    i = mod.pow(p)*th.sin(p*theta)
    return th.cat((r.unsqueeze(-1),i.unsqueeze(-1)),dim=-1)


def norm(input,p=2,dim=None):

    assert(th.is_tensor(input) and input.size(-1) == 2),"Input is expected "\
    +"to be a complex tensor."
    
    out = cabs(input)
    if dim is not None:
        return out.norm(p=2,dim=dim)
    else:
        return out.norm(p=2)
    

def complex(real,imag = None):
    
    if imag is not None:
        assert(real.shape == imag.shape),"Dimensions mismatch between real "\
        +"and imaginary input tensors."
    else:
        imag = th.zeros_like(real)
        
    return th.cat((real.unsqueeze(-1),imag.unsqueeze(-1)),dim = -1)



class Complex(object):
    
    def __init__(self,real,imag = None):
        
        assert(th.is_tensor(real) and th.is_tensor(imag)),"Init values for "\
        +"the complex tensor must be pytorch real tensors."        
        
        if imag is not None:
            assert(real.shape == imag.shape),"Dimensions mismatch between real "\
            +"and imaginary input tensors."
        else:
            imag = th.zeros_like(real)
        
        self.data  = th.cat((real.unsqueeze(-1),imag.unsqueeze(-1)),dim = -1)
        self.shape = self.data.shape[0:-1]
        
    def real(self):
        return self.data[...,0]
    
    def imag(self):
        return self.data[...,1]
    
    def conj(self):
        return Complex(self.data[...,0],-self.data[...,1])
    
    def conj_(self):
        
        self.data[...,1] = -self.data[...,1]
        return self

    def size(self,ind=None):
        assert(ind is None or (ind >= 0 and ind < len(self.shape))),\
        "dimension out of range."
        return self.data.size(ind)
    
    def ndim(self):
        return len(self.shape)
    
    def abs(self):
        return self.data.pow(2).sum(dim=-1).sqrt()
    
    def norm(self,p=2,dim=None):
        out = self.abs()
        if dim is None:
            return out.norm(p)
        else:
            return out.norm(p,dim)
    
    def add(self,other):
        if isinstance(other,Complex):
            out = self.data + other.data
        else:
            out = self.data.clone()
            out[...,0] += other
            
        return Complex(out[...,0],out[...,1])
    
    def __add__(self,other):
        return self.add(other)
    
    def __radd__(self,other):
        return self.add(other)
    
    def __iadd__(self,other):
        if isinstance(other,Complex):
            self.data += other.data
        else:
            self.data[...,0] += other    
        
        return self

    def sub(self,other):
        if isinstance(other,Complex):
            out = self.data - other.data
        else:
            out = self.data.clone()
            out[...,0] -= other
            
        return Complex(out[...,0],out[...,1])    
    
    def __sub__(self,other):
        return self.sub(other)
    
    def __rsub__(self,other):
        return self.sub(other)
    
    def __isub__(self,other):
        if isinstance(other,Complex):
            self.data -= other.data
        else:
            self.data[...,0] -= other    
        
        return self    

    def mul(self,other):
        if isinstance(other,Complex):
            real = self.data[...,0].mul(other.data[...,0])-self.data[...,1].mul(other.data[...,1])
            imag = self.data[...,0].mul(other.data[...,1])+self.data[...,1].mul(other.data[...,0])
            out = Complex(real,imag)
        else:
            out = self.data * other
            out = Complex(out[...,0],out[...,1])
        
        return out

    def __mul__(self,other):
        return self.mul(other)
    
    def __rmul__(self,other):
        return self.mul(other)
    
    def __imul__(self,other):
        if isinstance(other,Complex):
            real = self.data[...,0].mul(other.data[...,0])-self.data[...,1].mul(other.data[...,1])
            imag = self.data[...,0].mul(other.data[...,1])+self.data[...,1].mul(other.data[...,0])
            self.data[...,0] = real
            self.data[...,1] = imag
        else:
            self.data *= other
            
        return self
    
    def div(self,other):
        if isinstance(other,Complex):
            num = self.mul(other.conj())
            denom = (other.abs().data**2).unsqueeze(-1)
            out = num.data/denom
        else:
            out = self.data/other
            
        return Complex(out.data[...,0],out.data[...,1])
    
    def __truediv__(self,other):
        return self.div(other)
    
    def __rtruediv__(self,other):
        if isinstance(other,Complex):
            return other.div(self)
        else:
            denom = (self.abs()**2).unsqueeze(-1)
            out = self.conj().mul(other)
            out = out.div(denom)
            return out
        
    def __itruediv__(self,other):
        if isinstance(other,Complex):
            num = self.mul(other.conj())
            denom = (other.abs().data**2).unsqueeze(-1)
            self.data = num.data/denom
        else:
            self.data /= other
        
        return self
    
    def inv(self):
        return self.conj().div((self.abs()**2).unsqueeze(-1))
    
    def pow(self,power):
        mod = self.abs()
        theta = th.atan2(self.imag(),self.real())
        real = mod.pow(power)*th.cos(power*theta)
        imag = mod.pow(power)*th.sin(power*theta)
        return Complex(real,imag)
    
    def __pow__(self,power):
        return self.pow(power)

    def __ipow__(self,power):
        mod = self.abs()
        theta = th.atan2(self.imag(),self.real())
        self.data[...,0] = mod.pow(power)*th.cos(power*theta)
        self.data[...,1] = mod.pow(power)*th.sin(power*theta)
        return self
    
    def sqrt(self):
        return self.pow(0.5)
    
    def __repr__(self):
        return repr(self.data)