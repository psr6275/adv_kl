#from __future__ import absolute_import
import numpy as np
import time, os, sys,json, tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys
sys.path.append("../")
from utils import TripleDataSet

__all__ = ['WhiteBox']

def pgd_attack(target_model,X,y,params):
    niter = params['niter']
    alpha = params['alpha']
    eps = params['eps']
    normlz = params['normalize']

    X_pgd = Variable(X.data,requires_grad = True)
    opt = optim.Adam([X_pgd], lr=1.)
    for i in range(niter):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(target_model(X_pgd),y)
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data+eta,requires_grad=True)

        eta = torch.clamp(X_pgd.data-X.data,-eps,eps)
        if normlz is None:
            X_pgd = Variable(X.data + eta, requires_grad=True)
        elif normlz =='0/1':
            X_pgd = torch.clamp(X.data+eta,0,1)
            X_pgd = Variable(X_pgd,requires_grad=True)

    return X_pgd



class WhiteBox:
    def __init__(self, target_model,dataloader,attacker = 'pgd',params = {'eps':8/255,'niter':20, 'alpha':2/255,'normalize':'0/1'}):
        self.target_model = target_model.eval()
        self.dataloader = dataloader
        self.params = params
        print("<<<",attacker ," Attack>>>")
        if attacker is "pgd":
            self.attacker = pgd_attack
        self.tripleDataSet = self.attack_model()

    def attack_model(self):
        adv_list =[]
        input_list = []
        target_list = []
        num_batch = len(self.dataloader)
        for i, (inputs, targets) in enumerate(self.dataloader):
            inputs = Variable(inputs).cuda()
            targets = Variable(targets).cuda()
            adv_inputs = self.attacker(self.target_model,inputs,targets,self.params)

            input_list.append(inputs)
            target_list.append(targets)
            adv_list.append(adv_inputs)
            print("attacking batch [%.d/%.d]"%(i,num_batch),end='\r')
        print("\n")
        adv_tensor = torch.cat(adv_list,0)
        input_tensor = torch.cat(input_list,0)
        target_tensor = torch.cat(target_list,0)
        print(target_list)
        print(target_tensor)
        print("complete attacks!")
        return TripleDataSet(input_tensor,adv_tensor,target_tensor)





