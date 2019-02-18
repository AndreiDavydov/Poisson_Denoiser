import torch as th
from torch import nn

from PoisDenoiser.nnLayers import modules
from pydl.nnLayers import init

class PoisNet(nn.Module): # this class is almost fully based on UDNet class from pydl.nnLayers.modules
    
    def __init__(self, kernel_size=(7,7),\
                 input_channels=1,\
                 output_features=48,\
                 rbf_mixtures=51,\
                 rbf_precision=4,\
                 stages = 5,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 convWeightSharing = True,\
                 scale_f = True,\
                 scale_t = False,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True,\
                 rbf_start = -100,\
                 rbf_end = 100,\
                 data_min = -100,\
                 data_max = 100,\
                 data_step = 0.1,\
                 clb = 0,\
                 cub = 255):

        super(PoisNet, self).__init__()
        
        rbf_centers = th.linspace(rbf_start,rbf_end,rbf_mixtures).type_as(th.Tensor())       
        self.rbf_data_lut = init.rbf_lut(rbf_centers,rbf_precision,\
                                        data_min,data_max,data_step)
        self.stages = stages   

        self.a_cond = None # Projection condition multiplier. Equal to 0 (exp(a_cond)) by default. 
        
        self.resRBFPois = nn.ModuleList([modules.ResRBFPoisLayer(kernel_size,\
                                      input_channels,\
                                      output_features,\
                                      rbf_mixtures,\
                                      rbf_precision,\
                                      pad,\
                                      convWeightSharing,\
                                      rbf_start,\
                                      rbf_end,\
                                      padType,\
                                      scale_f,\
                                      scale_t,\
                                      normalizedWeights,\
                                      zeroMeanWeights) for i in range(self.stages)])      

        self.bbProj = nn.Hardtanh(min_val=clb, max_val=cub)
        
    def forward(self, input, noisy):

        for m in self.resRBFPois:
            input = m(input, noisy, self.a_cond, self.rbf_data_lut.type_as(input))
        
        return self.bbProj(input)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'stages = ' + str(self.stages) + ')'   
