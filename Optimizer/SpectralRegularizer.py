# encoding: utf-8
"""
Spectral Decay Regularizer.
Author: Jason.Fang
Update time: 16/03/2022
Utility: Solving the defect that the torch.optim optimizer can only achieve L2 regularization and penalize all parameters in the network.
Novelty: Applying spectral norm to achieve weight decay due to that it is difficult to give good priors for the coefficient in L2 regularize.
         Our spectral norm-based weight decay is non-sensitive to the coefficient set.
Reference: 
        https://github.com/PanJinquan/pytorch-learning-notes/blob/master/image_classification/train_resNet.py
        https://zhuanlan.zhihu.com/p/62393636
"""
import os
import math
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import torch.nn.init as init
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules import conv
from torch.nn.utils import spectral_norm
from resnet import resnet50

class SpectralRegularizer(torch.nn.Module):
    def __init__(self, model, coef=1e-4, p=2, is_spec=True):
        '''
        :param model 
        :param coef: coefficient of regularizer
        :param p: The value of the power exponent in the norm.
                  if p=1, achieveing L1 regularizaiton OR Nuclear norm-based weight sparse
                  if p=2, achieveing L2 regularization OR Frobenius norm-based weight decay
        :param is_spec: adding spectra_norm before L1 or L2 regularization
        '''
        super(SpectralRegularizer, self).__init__()

        self.coef=coef
        self.p=p
        self.is_spec=is_spec
        self._weight_info(self._get_weight(model)) #static weights

    def _get_weight(self,model):
        '''
        :param model:
        :return: weights of model
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def _weight_info(self,weight_list):
        '''
        print the weight information.
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

    def forward(self, model):
        weight_list=self._get_weight(model) #dynamic weights
        reg_loss = self.regularization_loss(model, weight_list)
        return reg_loss

    def regularization_loss(self, model, weight_list):
        '''
        :param weight_list: updated weights
        :return:
        '''
        reg_loss=0
        if self.is_spec: #spectral norm-based regularizer
            for m_name, module in model.named_children(): #.named_modules()
                if hasattr(module, 'weight'):
                    spec_module = spectral_norm(module)
                    reg_loss += torch.norm(spec_module.weight, p=self.p) #L1 torch.sum(abs(w))
            reg_loss = self.coef*reg_loss
        else: #no spectral norm
            for w_name, w in weight_list:
                reg_loss += torch.norm(w, p=self.p) #L1 torch.sum(abs(w))
            reg_loss = self.coef*reg_loss
       
        return reg_loss

if __name__ == "__main__":
    #for debug  
    model = resnet50(pretrained=True, num_classes=1000).cuda()
    spec_reg =SpectralRegularizer(model, coef=1e-4, p=2, is_spec=True).cuda()
    print(spec_reg(model))
 
