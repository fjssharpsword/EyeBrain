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
#https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html#torch-nn-utils-spectral-norm
#https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/spectral_norm.py
from torch.nn.utils import spectral_norm
#https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html
#https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/parametrizations.py
#from torch.nn.utils.parametrizations import spectral_norm
from resnet import resnet50
from vit import ViT

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
        self.m_name = [] 
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
            self.m_name.append(name.replace('.weight', ''))
        print("---------------------------------------------------")

    def forward(self, model):
        weight_list=self._get_weight(model) #dynamic weights
        reg_loss = self.regularization_loss(model, weight_list)
        return reg_loss

    #approximated SVD
    #https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
    def _power_iteration(self, W, eps=1e-10, Ip=1):
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

    def regularization_loss(self, model, weight_list):
        '''
        :param weight_list: updated weights
        :return:
        '''
        reg_loss=0
        if self.is_spec: #spectral norm-based regularizer
            #for m_name, module in model.named_children(): #resnet
            #for idx, module in enumerate(model.modules()):# vit
                #if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear))and hasattr(module, 'weight'): 
            for m_name, module in model.named_modules():
                if m_name in self.m_name:

                    weight_mat = module.weight.view(module.weight.shape[0],-1)
                    u, s, v = self._power_iteration(weight_mat)
                    reg_loss += torch.norm(module.weight/s, p=self.p)

                    """
                    sn_mod = spectral_norm(module)
                    reg_loss += torch.norm(sn_mod.weight, p=self.p)
                    """
                    """
                    weight_mat = module.weight.view(module.weight.shape[0],-1)
                    u = sn_mod.weight_u#.unsqueeze(0)
                    v = sn_mod.weight_v#.unsqueeze(1)
                    sigma = torch.dot(u, torch.mv(weight_mat, v))
                    reg_loss += torch.norm(module.weight/sigma, p=self.p) #L1 torch.sum(abs(w))
                    """
                    """
                    max_weight = torch.matmul(sn_mod.weight_u.unsqueeze(1), sn_mod.weight_v.unsqueeze(0))
                    max_weight = max_weight.view(module.weight.shape)
                    reg_loss += torch.norm(module.weight-max_weight, p=self.p)
                    """
            reg_loss = self.coef*reg_loss
        else: #no spectral norm
            for w_name, w in weight_list:
                reg_loss += torch.norm(w, p=self.p) #L1 torch.sum(abs(w))
            reg_loss = self.coef*reg_loss
       
        return reg_loss

if __name__ == "__main__":
    #for debug  
    #model = resnet50(pretrained=True, num_classes=1000).cuda()
    model = ViT(image_size = 224, patch_size = 32, num_classes = 1000, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048).cuda()
    spec_reg =SpectralRegularizer(model, coef=1e-4, p=2, is_spec=True).cuda()
    print(spec_reg(model))
 
