import torch as th
from torch import nn
from PoisDenoiser.nnLayers import cascades
from pydl.nnLayers.functional.functional import imLoss, mseLoss
from pydl.nnLayers import init
from pydl.utils import formatInput2Tuple, getPad2RetainShape

def initialize_conv_weights(weights):
    weights.data = th.randn(weights.size())


class ResRBFPoisLayer(nn.Module): # this class is almost fully based on ResidualRBFLayer.  
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 rbf_mixtures,\
                 rbf_precision,\
                 pad='same',\
                 convWeightSharing=True,\
                 lb=-100,\
                 ub=100,\
                 padType='symmetric',\
                 scale_f=True,\
                 scale_t=True,\
                 normalizedWeights=True,\
                 zeroMeanWeights=True,\
                 prox_param=True):
        
        super(ResRBFPoisLayer, self).__init__()
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)       
        
        if isinstance(pad,str) and pad == 'same':
            pad = getPad2RetainShape(kernel_size)
            
        self.pad = formatInput2Tuple(pad,int,4)
        self.padType = padType
        self.normalizedWeights = normalizedWeights
        self.zeroMeanWeights = zeroMeanWeights
        self.convWeightSharing = convWeightSharing
        self.lb = lb
        self.ub = ub
        
        # Initialize conv weights
        shape = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))

        # initialize_conv_weights(self.conv_weights)
        init.dct(self.conv_weights)
        
        # Initialize the scaling coefficients for the conv weight normalization
        if scale_f and normalizedWeights:
            self.scale_f = nn.Parameter(th.Tensor(output_features).fill_(0.1))
        else:
            self.register_parameter('scale_f', None)        
        
        if not self.convWeightSharing:
            self.convt_weights = nn.Parameter(th.Tensor(th.Size(shape)))
            init.dct(self.convt_weights)
        
            if scale_t and normalizedWeights:
                self.scale_t = nn.Parameter(th.Tensor(output_features).fill_(0.1))
            else :
                self.register_parameter('scale_t', None)

        # Initialize the params for the PoisProx.
        # Projection condition multiplier.
        if prox_param:
            self.prox_param = nn.Parameter(th.Tensor(1).fill_(0.1))
        else:
            self.register_parameter('prox_param', None)
        
        # Initialize the rbf_weights
        self.rbf_weights = nn.Parameter(th.Tensor(output_features,rbf_mixtures).fill_(1e-4))
        self.rbf_centers = th.linspace(lb,ub,rbf_mixtures).type_as(self.rbf_weights)
        self.rbf_precision = rbf_precision
        
    def forward(self, input, noisy, rbf_data):

        if self.convWeightSharing:
            return cascades.resRBFPois_f_sw(input, noisy,\
                            self.conv_weights,\
                            self.rbf_weights, self.rbf_centers, self.rbf_precision, rbf_data,\
                            self.pad, self.padType,\
                            self.scale_f, \
                            self.normalizedWeights, self.zeroMeanWeights,\
                            self.lb, self.ub,\
                            self.prox_param)
        else:
            return cascades.resRBFPois_f(input, noisy,\
                            self.conv_weights, self.convt_weights,\
                            self.rbf_weights, self.rbf_centers, self.rbf_precision, rbf_data,\
                            self.pad,self.padType,\
                            self.scale_f, self.scale_t, \
                            self.normalizedWeights, self.zeroMeanWeights,\
                            self.lb, self.ub,\
                            self.prox_param)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size = ' + str(tuple(self.conv_weights.shape[2:])) \
            + ', input_channels = ' + str(self.conv_weights.shape[1]) \
            + ', output_features = ' + str(self.conv_weights.shape[0]) \
            + ', rbf_mixtures = ' + str(self.rbf_centers.numel()) \
            + ', normalizedWeights = ' + str(self.normalizedWeights) \
            + ', zeroMeanWeights = ' + str(self.zeroMeanWeights) \
            + ', convWeightSharing = ' + str(self.convWeightSharing) + ')'  
