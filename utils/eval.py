from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

__all__ = ['accuracy','kl_loss','custom_DataLoader','noise_DataLoader','triple_DataLoader','CustomDataSet','TripleDataSet','GaussianDataSet']

def accuracy(output, target, topk=(1.)):
    '''Compute the top1 and top k error'''
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res

def kl_loss(output, target,reverse = False):
    '''after dividing T!
    target is predicted output
    '''
    output_prob = nn.Softmax(-1)(output)
    target_prob = nn.Softmax(-1)(target)
    if reverse:
        loss = -torch.sum(output_prob * torch.log(target_prob / output_prob))
    else:
        loss = -torch.sum(target_prob*torch.log(output_prob/target_prob))

    return loss

class CustomDataSet(Dataset):
    def __init__(self,inputs_list, targets_list):
        """

        :param inputs_list: it is a list of input batches
        :param targets_list: it is a list of target batches
        """
        super(CustomDataSet, self).__init__()
        self.inputs_list = inputs_list
        self.targets_list = targets_list
        self.concat_items()

    def concat_items(self):
        self.inputs = torch.cat(self.inputs_list,0)
        self.targets = torch.cat(self.targets_list,0)
        self.len = len(self.targets)

    def __getitem__(self, item):
        img = self.inputs[item]
        target = self.targets[item]
        return img, target

    def __len__(self):
        
        return self.len

class GaussianDataSet(Dataset):
    def __init__(self,inputs_tensor, targets_tensor,noise_std = 0.1):
        super(GaussianDataLoader, self).__init__()
        self.inputs_tensor = inputs_tensor
        self.targets_tensor=targets_tensor
        self.noise_std = noise_std
        self.len = len(self.targets_tensor)
        self.generate_random()

    def generate_random(self):
        self.noise_tensor = noise(self.inputs_tensor,self.noise_std)

    def __getitem__(self, item):
        img = self.inputs_tensor[item]
        noise_img = self.noise_tensor[item]
        target = self.targets_tensor[item]
        return img, noise_img,target

    def __len__(self):
        return self.len
class UniformDataSet(Dataset):
    def __init__(self,inputs_tensor, targets_tensor, epsilon=4./255., lb = 0.0,ub = 1.0):
        super(UniformDataSet, self).__init__()
        self.inputs_tensor = inputs_tensor
        self.targets_tensor = targets_tensor
        self.epsilon = epsilon
        self.lb = lb
        self.ub = ub
        self.generate_random()

    def generate_random(self):
        self.noise_tensor = self.inputs_tensor + self.epsilon*(torch.rand_like(self.inputs_tensor)*2-1)
        self.noise_tensor = torch.clamp(self.noise_tensor,self.low,self.high)
    def __getitem__(self,item):
        img = self.inputs_tensor[item]
        noise_img = self.noise_tensor[item]
        target = self.targets_tensor[item]
        return img, noise_img, target





class TripleDataSet(Dataset):
    def __init__(self,inputs_tensor1, inputs_tensor2,targets_tensor):
        super(TripleDataSet, self).__init__()
        self.inputs1 = inputs_tensor1
        self.inputs2 = inputs_tensor2
        self.targets = targets_tensor
        self.len = len(targets_tensor)

    def __getitem__(self, item):
        img1 = self.inputs1[item]
        img2 = self.inputs2[item]
        target = self.targets[item]
        return img1, img2, target
    def __len__(self):
        return self.len

def noise(X, noise_std, low=0, high=1):
    X_noise = X + noise_std * torch.randn_like(X)
    return torch.clamp(X_noise, low, high)

def custom_DataLoader(inputs_list,targets_list,batch_size = 10,shuffle=True,num_workers=16):
    dataset = CustomDataSet(inputs_list,targets_list)

    return DataLoader(dataset,batch_size = batch_size, shuffle=shuffle,num_workers=num_workers)

def noise_DataLoader(inputs_tensor, targets_tensor,batch_size=10,shuffle=True, num_workers = 16):
    dataset = GaussianDataSet(inputs_tensor,targets_tensor)
    return DataLoader(dataset, batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

def triple_DataLoader(inputs_tensor1,inputs_tensor2,targets_tensor,batch_size=10,shuffle=True, num_workers = 16):
    dataset = TripleDataSet(inputs_tensor1,inputs_tensor2,targets_tensor)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle, num_workers = num_workers)
