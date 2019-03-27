import argparse
import os
import numpy as np
import time, pickle,sys, json, seaborn, PIL, tempfile, warnings
import shutil

import torch
import torch.nn as nn
import torch.optm as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data as data

import matplotlib.pyplot as plt

# User-defined parts
from utils import mkdir_p,accuracy,AverageMeter, Logger, savefig, kl_loss
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()

# datasets
parser.add_argument('--data',type=str,default='mnist', help='mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--workers', default=16, type=int, metavar='N',help='number of data loading workers (default:16)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')

# classifier
parser.add_argument('--load_clf',type=str,defaut=None, help = 'Specify the load path of classifiers. {None:train}')
parser.add_argument('--tansform',type=bool,default=False,help='add transformation for (clf or dae)')

# Learning Parameters
parser.add_argument('--lr','--learning-rate',default=0.001,type=float,metavar='LR',help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay','--wd',default=5e-4,type=float, metavar='W',help='weight decay (default:5e-4)')
parser.add_argument('--epochs',default=30, type=int, help='number of epochs to run')

# PGD attacks
parser.add_argument('--eps',type=float,default=8/255. help='pgd bound')
parser.add_argument('--niters',type=int,default=20, help='pgd steps')
parser.add_argument('--alpha',type=float,default=2/255,help='pgd steps')

# DAE part
parser.add_argument('--dae-loss',type=str,default='KL',help='KL, L2, L1')

# Checkpoints
parser.add_argument('-c','--checkpoint',default='checkpoint',type=str,metavar='PATH',
                    help='path to save checkpoint {default:checkpoint}')

args = parser.parse_args()

def main():
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print('==> preparing dataset %s',args.dataset)

    if args.dataset =='mnist':
        dataloader = datasets.MNIST
    elif args.dataset =='fashion_mnist':
        dataloader = datasets.FashionMNIST
    elif args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
    elif args.dataset =='cifar100':
        dataloader = datasets.CIFAR100

    if args.transform and args.dataset in ['cifar10','cifar100']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
        ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])



    trainset = dataloader(root = './data',train=True, download=True,transform = transform_train)
    trainloader = data.Dataloader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=True,transform=transform_test)
    testloader = data.Dataloader(testset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    ## construct classifier model
    if args.dataset == 'mnist':
        model_clf,model_dae = models.__dict__[args.dataset](

        )

    elif args.dataset == 'fashion_mnist':
        model_clf,model_dae = models.__dict__[args.dataset](

        )
    elif args.dataset == 'cifar10':
        model_clf, model_dae = models.__dict__[args.dataset](

        )
    elif args.dataset == 'cifar100':
        model_clf,model_dae = models.__dict__[args.dataset](

        )

    model_clf = nn.DataParallel(model_clf).cuda()

    # Model Load or Train!
    if args.load_clf:
        print('==> Load the trained classifier..')
        assert os.path.isfile(args.load_clf), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.load_clf)
        checkpoint = torch.load(args.load_clf)
        model_clf.load_state_dict(checkpoint['state_dict'])

    else:
# Train the classifier
        train_clf(model_clf,trainloader)

def load_data(dataset = args.dataset,transform=args.transform , train_batch=args.train_batch,
              test_batch = args.test_batch, workers=args.workers):
    '''

    :param dataset should be in ['mnist', 'fashion_mnist', 'cifar10','cifar100']:
    :param transform:
    :return:
    '''
    print('==> preparing dataset %s', dataset)

    if dataset == 'mnist':
        dataloader = datasets.MNIST
    elif dataset == 'fashion_mnist':
        dataloader = datasets.FashionMNIST
    elif dataset == 'cifar10':
        dataloader = datasets.CIFAR10
    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100

    if transform and dataset in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.Dataloader(trainset, batch_size=train_batch, shuffle=True, num_workers=workers)

    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.Dataloader(testset, batch_size=test_batch, shuffle=True, num_workers=workers)

    return trainloader, testloader

def load_model(dataset = args.dataset,checkpoint = args.checkpoint, load_clf=args.load_clf):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    assert dataset in model_names, 'Error: should choice models among '+model_names

    if dataset == 'mnist':
        model_clf,model_dae = models.__dict__[dataset](

        )

    elif dataset == 'fashion_mnist':
        model_clf,model_dae = models.__dict__[dataset](

        )
    elif dataset == 'cifar10':
        model_clf, model_dae = models.__dict__[dataset](

        )
    elif dataset == 'cifar100':
        model_clf,model_dae = models.__dict__[dataset](

        )

    model_clf = nn.DataParallel(model_clf).cuda()

    # Model Load or Train!
    if load_clf:
        print('==> Load the trained classifier..')
        assert os.path.isfile(load_clf), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(load_clf)
        checkpoint = torch.load(load_clf)
        model_clf.load_state_dict(checkpoint['state_dict'])

    return model_clf, model_dae



def train_clf(model,trainloader,testloader,criterion = nn.CrossEntropyLoss(), lr = args.lr,epochs = args.apochs, checkpoint = args.checkpoint, dataset = args.dataset):
    #optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    best_acc = 0
    logger = Logger(os.path.join(checkpoint,'log.txt'),title = dataset)
    logger.set_names(['Train Loss','Valid Loss', 'Train Acc', 'Valid Acc'])

    for epoch in range(epochs):
        # Switch train mode
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        print('\nEpoch: [%d |%d] ' % (epoch+1,epochs))
        for batch_idx, (inputs,targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs),Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # print intermediate top1 and top3 error
            prec1, prec3 = accuracy(outputs.data, targets.data,topk=(1,3))
            if (batch_idx)% 10 == 1:
                print("Training [Acc] top1: %.2f , top3: %.2f "%(100*prec1, 100*prec3))
            losses.update(loss.data[0],inputs.size(0))
            top1.update(prec1[0],inputs.size(0))
            top3.update(prec3[0],inputs.size(0))


            # Compute gradient and SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate Test error!
        test_loss, test_acc, test_acc3 = test_clf(model, testloader, criterion)
        print('Training/Test [loss] %.4f / %.4f, [top1] %.2f / %.2f, [top3] %.2f / %.2f' %
              (losses.avg,test_loss, 100*top1.avg, 100*test_acc, 100*top3.avg, 100*test_acc3))
        #print('Test [loss] %.4f, [top1]')
        logger.append([losses.avg,top1.avg,top3.avg])
        # save_model
        is_best = top1.avg > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'acc':top1.avg,
            'best_acc':best_acc,
            'optimizer': optimizer.state_dict(),

        }, is_best, checkpoint = checkpoint)

    logger.close()
    logger.plot()
    savefig()
    print('Best acc:')
    print(best_acc)

def train_clf_step(model,trainloader,epoch,epochs,criterion,):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    print('\nEpoch: [%d |%d] ' % (epoch + 1, epochs))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # print intermediate top1 and top3 error
        prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1, 3))
        if (batch_idx) % 10 == 1:
            print("Training [Acc] top1: %.2f , top3: %.2f " % (100 * prec1, 100 * prec3))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        # Compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_dae(model_dae, model_clf, data_loader,dae_loss = args.dae_loss, lr = args.lr,
              epochs = 50, tempr = 10, std = 0.1, checkpoint='checkpoint'):
    assert dae_loss in ['KL, L2, L1','KL_reverse'], 'Error dae_loss should be in [KL, L2, L1, KL_reverse]'
    # Use classifier only for evaluation
    model_clf.eval()
    optimizer = optim.Adam(model_dae.parameters(),lr=lr)
    if dae_loss == 'L1':
        criterion = nn.L1Loss()
        checkpoint = checkpoint+'/dae_l1'
    elif dae_loss =='L2':
        criterion = nn.L2Loss()
        checkpoint = checkpoint+'/dae_l2'
    else:
        criterion = kl_loss
        checkpoint = checkpoint+'/dae_kl'

    for epoch in range(epochs):
        print('\nEpoch: [%d |%d] ' % (epoch + 1, epochs))

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            noise_inputs = noise(inputs)
            noise_inputs,inputs, targets =noise_inputs.cuda(), inputs.cuda(), targets.cuda()
            noise_inputs,inputs, targets = Variable(noise_inputs), Variable(inputs), Variable(targets)

            outputs = model_clf(inputs)
            noise_ = model_dae(noise_inputs)
            denoise_inputs = torch.clamp(noise_inputs + noise_,0,1)
            outputs_ = model_clf(denoise_inputs)

            if dae_loss == 'KL':
                loss = criterion(outputs_/tempr,outputs/tempr)
            elif dae_loss =='KL_reverse':
                loss = criterion(outputs_/tmpr,outputs/tempr,True)
            else:
                loss = criterion(outputs,outputs_)


            # Training the models
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Test accuracy!




"""
class train_dae():
    def __init__(self, model_dae, model_clf, data_loader, dae_loss= args.dae_loss, lr = args.lr,
                 epochs = args.epochs, tempr = 10):
        self.model_dae = model_dae
        self.mode_clf = model_clf
        self.data_loader = data_loader
        self.dae_loss = dae_loss
        self.lr = lr
        self.epochs = epochs
        self.tempr = tempr

    def kl_loss(self, output, target):
"""

def test_clf(model, testloader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile = True), Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec3 = accurac(outputs.data, targets.data, topk=(1,3))
        losses.update(loss.data[0],inputs.size(0))
        top1.update(prec1[0],inputs.size(0))
        top3.update(prec3[0],inputs.size(0))


    return losses.avg, top1.avg, top3.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint',filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint,filename)
    torch.save(state,filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,'model_best.pth.tar'))


def noise(X, noise_std):
    X_noise = X+noise_std*torch.randn_like(X)
    return torch.clamp(X_noise, 0,1)


if __name__ =='__main__':
    main()



