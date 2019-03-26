from __future__ import absolute_import
import torch
import torch.nn as nn

__all__ = ['accuracy','kl_loss']

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