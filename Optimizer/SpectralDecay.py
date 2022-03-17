# encoding: utf-8
"""
Spectral Decay Regularizer.
Author: Jason.Fang
Update time: 16/03/2022
"""

import math
import torch
import torch.nn.init as init
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules import conv
from torch.nn.utils import spectral_norm

class SpecConv2d(conv._ConvNd):
    r"""
    Convolution with spectral decay as regularizer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(1)
        groups = 1
        bias = False
        super(SpecConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, 'zeros')

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    #approximated SVD
    #https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
    def _power_iteration(self, W, eps=1e-10, Ip=2):
        """
        power iteration for max_singular_value
        """
        v = torch.FloatTensor(W.size(1), 1).normal_(0, 1).cuda()
        W_s = torch.matmul(W.T, W)
        #while True:
        for _ in range(Ip):
            v_t = v
            v = torch.matmul(W_s, v_t)
            v = v/torch.norm(v)
            #if abs(torch.dot(v.squeeze(), v_t.squeeze())) > 1 - eps: #converged
            #    break

        u = torch.matmul(W, v)
        s = torch.norm(u)
        u = u/s
        #return left vector, sigma, right vector
        return u, s, v

    def _specgrad(self):
        
        spec_module = spectral_norm(self)
        spec_weight = torch.matmul(spec_module.weight_u.unsqueeze(1), spec_module.weight_v.unsqueeze(0))

        wgrad = spec_weight.view(self.weight.shape)

        return wgrad

    def updategrad(self, coef=1E-4):
        
        Wgrad = self._specgrad()
        if self.weight.grad is None:
            self.weight.grad = coef * Wgrad
        else:
            self.weight.grad += coef * Wgrad

#backward
def weightdecay(model, coef=1E-4, skiplist=[]):

    def _apply_weightdecay(name, module, skiplist=[]):
        if hasattr(module, 'updategrad'):
            return not any(name[-len(entry):] == entry for entry in skiplist)
        return False

    module_list = enumerate(module for name, module in model.named_modules() if _apply_weightdecay(name, module, skiplist=skiplist))
    for i, module in module_list:
        module.updategrad(coef=coef)

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 3, 16, 16).cuda()
    sconv = SpecConv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2).cuda()
    out = sconv(x)
    print(out.shape)
    weightdecay(sconv)

