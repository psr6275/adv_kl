from __future__ import absolute_import
import torch
import torch.nn as nn

__all__ = ['accuracy','kl_loss','custom_DataLoader']

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

class CustomDataSet(torch.utils.data.Dataset):
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

    def __getitem__(self, item):
        img = self.inputs[item]
        target = self.targets[item]
        return img, target

    def __len__(self):
        n = self.targets.size[0]
        return n

def custom_DataLoader(inputs_list,targets_list,batch_size = 10,shuffle=True,num_workers=16):
    dataset = CustomDataSet(inputs_list,targets_list)

    return torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle=shuffle,num_workers=num_workers)
