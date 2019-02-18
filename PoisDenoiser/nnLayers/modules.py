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
                 pad = 'same',\
                 convWeightSharing = True,\
                 lb = -100,\
                 ub = 100,\
                 padType = 'symmetric',\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True):
        
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

        initialize_conv_weights(self.conv_weights)
        #init.dct(self.conv_weights) # wtf on 8 channels???
        
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
        
        # Initialize the rbf_weights
        self.rbf_weights = nn.Parameter(th.Tensor(output_features,rbf_mixtures).fill_(1e-4))
        self.rbf_centers = th.linspace(lb,ub,rbf_mixtures).type_as(self.rbf_weights)
        self.rbf_precision = rbf_precision
        
    def forward(self, input, noisy, a_cond, rbf_data):

        if self.convWeightSharing:
            return cascades.resRBFPois_f_sw(input, noisy, a_cond,\
                            self.conv_weights,\
                            self.rbf_weights, self.rbf_centers, self.rbf_precision, rbf_data,\
                            self.pad, self.padType,\
                            self.scale_f, \
                            self.normalizedWeights, self.zeroMeanWeights,\
                            self.lb, self.ub)
        else:
            return cascades.resRBFPois_f(input, noisy, a_cond,\
                            self.conv_weights, self.convt_weights,\
                            self.rbf_weights, self.rbf_centers, self.rbf_precision, rbf_data,\
                            self.pad,self.padType,\
                            self.scale_f, self.scale_t, \
                            self.normalizedWeights, self.zeroMeanWeights,\
                            self.lb, self.ub)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size = ' + str(tuple(self.conv_weights.shape[2:])) \
            + ', input_channels = ' + str(self.conv_weights.shape[1]) \
            + ', output_features = ' + str(self.conv_weights.shape[0]) \
            + ', rbf_mixtures = ' + str(self.rbf_centers.numel()) \
            + ', normalizedWeights = ' + str(self.normalizedWeights) \
            + ', zeroMeanWeights = ' + str(self.zeroMeanWeights) \
            + ', convWeightSharing = ' + str(self.convWeightSharing) + ')'


# class MSELoss(nn.Module):
#     def __init__(self,grad=False):
        
#         super(MSELoss,self).__init__()
        
#         self.grad = grad
#         self.mode = "validation"
    
#     def forward(self,input,target):
        
#         return mseLoss.apply(input,target,self.grad,self.mode)

#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#         + 'gradMSE = ' + str(self.grad)\
#         + ', mode = ' + self.mode + ')'        
   
    
# class PSNRLoss(nn.Module):
    
#     def __init__(self,peakval=255):
        
#         super(PSNRLoss,self).__init__()       
        
#         self.peakval = peakval
#         self.loss = 'psnr'
#         self.mode = 'normal'
        
#     def forward(self,input,other):
        
#         return imLoss.apply(input,other,self.peakval,self.loss,self.mode) 
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'peakVal = ' + str(self.peakval) + ')'    
