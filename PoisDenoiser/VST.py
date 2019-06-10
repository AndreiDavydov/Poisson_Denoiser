import torch as th
from math import sqrt

def VST_forward(x):
    return 2*th.sqrt(x + 3/8)

def VST_backward_alg(x):
    return (x/2)**2 - 3/8

def VST_backward_unbiased(x):
    return (x/2)**2 - 3/8 + 1/4

def VST_backward_unbiased_exact(x):
    return (x/2)**2 - 3/8 + 1/4 + 1/4*sqrt(3/2)/x - 11/8/x/x + 5/8*sqrt(3/2)/x/x/x