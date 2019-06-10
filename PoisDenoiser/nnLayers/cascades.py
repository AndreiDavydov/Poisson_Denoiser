import torch as th
from pydl.nnLayers.functional import functional as F
from PoisDenoiser.nnLayers.functional import poisProx

pad2D = F.Pad2D.apply
pad_transpose2D = F.Pad_transpose2D.apply
weightNormalization = F.WeightNormalization.apply
grbf = F.grbf_LUT.apply
conv2d = th.nn.functional.conv2d
conv2d_t = th.nn.functional.conv_transpose2d


def resRBFPois_f(input, noisy,\
                  weights, weights_t,\
                  rbf_weights, rbf_centers, rbf_precision, data_lut,\
                  pad=0, padType='zero',\
                  alpha=None, alpha_t=None,\
                  normalizedWeights=False, zeroMeanWeights=False,\
                  lb=-100, ub=100,\
                  prox_param=None):
    r""" residual layer with independent weights between the convolutional and 
    transpose convolutional layers."""
    
    if (normalizedWeights and alpha is not None) or zeroMeanWeights:
        weights = weightNormalization(weights, alpha,\
            normalizedWeights, zeroMeanWeights)
    if (normalizedWeights and alpha_t is not None) or zeroMeanWeights:
        weights_t = weightNormalization(weights_t, alpha_t,\
            normalizedWeights, zeroMeanWeights)

    # conv2d
    out = conv2d(pad2D(input,pad,padType),weights,bias=None,stride=1)
    #clipping of the values before feeding them to the grbf layer
    out = th.clamp(out,lb,ub)
    # gaussian rbf
    out = grbf(out,rbf_weights,rbf_centers,rbf_precision,data_lut)
    # conv_tranpose2d
    out = pad_transpose2D(conv2d_t(out,weights_t,bias = None,stride = 1),pad,padType)
    # Projection of the result, given the input of the network
    out = input - out
    out = th.clamp(out, 1e-8, float('Inf'))
    out = poisProx(out, noisy, prox_param)

    return out

def resRBFPois_f_sw(input, noisy,\
            weights,\
            rbf_weights, rbf_centers, rbf_precision, data_lut,\
            pad=0,padType='zero',\
            alpha=None, \
            normalizedWeights=False, zeroMeanWeights=False,\
            lb=-100,ub=100,\
            prox_param=None):
    r""" residual layer with shared weights between the convolutional and 
    transpose convolutional layers."""
    
    if (normalizedWeights and alpha is not None) or zeroMeanWeights:
        weights = weightNormalization(weights,alpha,normalizedWeights,zeroMeanWeights)

    # conv2d
    out = conv2d(pad2D(input,pad,padType),weights,bias = None,stride = 1)
    #clipping of the values before feeding them to the grbf layer
    out = th.clamp(out,lb,ub)
    # gaussian rbf
    out = grbf(out,rbf_weights,rbf_centers,rbf_precision,data_lut)
    # conv_tranpose2d
    out = pad_transpose2D(conv2d_t(out,weights,bias = None,stride = 1),pad,padType)
    # Projection of the result, given the input of the network
    out = input - out
    out = th.clamp(out, 1e-8, float('Inf'))
    out = poisProx(out, noisy, prox_param)

    return out