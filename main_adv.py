import argparse
import os
import numpy as np
import time, pickle,sys, json, seaborn, PIL, tempfile, warnings
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data as data 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# User-defined parts
from utils import mkdir_p,accuracy,AverageMeter, Logger, savefig, kl_loss, custom_DataLoader
import models

__all__=['load_data','load_model','train_clf','test_clf','test_clf3','train_dae','test_dae','noise','CombinedModel']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

"""
parser = argparse.ArgumentParser()

# datasets
parser.add_argument('--data',type=str,default='mnist', help='mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--workers', default=16, type=int, metavar='N',help='number of data loading workers (default:16)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')

# classifier
parser.add_argument('--load_clf',type=str,default=None, help = 'Specify the load path of classifiers. {None:train}')
parser.add_argument('--tansform',type=bool,default=False,help='add transformation for (clf or dae)')

# Learning Parameters
parser.add_argument('--lr','--learning-rate',default=0.001,type=float,metavar='LR',help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay','--wd',default=5e-4,type=float, metavar='W',help='weight decay (default:5e-4)')
parser.add_argument('--epochs',default=30, type=int, help='number of epochs to run')

# PGD attacks
parser.add_argument('--eps',type=float,default=8/255, help='pgd bound')
parser.add_argument('--niters',type=int,default=20, help='pgd steps')
parser.add_argument('--alpha',type=float,default=2/255,help='pgd steps')

# DAE part
parser.add_argument('--dae-loss',type=str,default='KL',help='KL, L2, L1')

# Checkpoints
parser.add_argument('-c','--checkpoint',default='checkpoint',type=str,metavar='PATH',
                    help='path to save checkpoint {default:checkpoint}')

args = parser.parse_args()
"""
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
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=True,transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

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

class CombinedModel:
    def __init__(self,dataset='mnist',transform=None,train_batch=128, test_batch = 100, workers=16):
        self.dataset = dataset
        self.transform = transform
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.workers = workers

        # Load Dataset
        self.load_data()


    def load_data(self):
        self.trainloader, self.testloader = load_data(self.dataset, self.transform, self.train_batch,
                                                  self.test_batch, self.workers)

    def construct_model(self,checkpoint = 'checkpoint', dae_type='recon',noise_level = 8/255):
        self.checkpoint = checkpoint
        self.dae_type = dae_type
        self.noise_level= noise_level
        self.model_clf, self.model_dae, self.model_comb = load_model(self.dataset,checkpoint,load_clf = False,dae_type = dae_type,noise_level = noise_level)

    def load_model(self,checkpoint, model_type):
        # should implement
        print("Not Implemented Yet")

    def train_clf(self, criterion = nn.CrossEntropyLoss(), lr = 0.001, epochs = 30):
        self.clf_criterion = criterion
        train_clf(self.model_clf,self.trainloader,self.testloader,self.clf_criterion,
                  lr,epochs,self.checkpoint,self.dataset)

    def test_clf(self,testloader=None, criterion = None):
        if testloader is None:
            testloader = self.testloader
        if criterion is None:
            criterion = self.clf_criterion
        print(test_clf(self.model_clf,testloader,criterion))

    def train_dae(self, criterion = None, dae_loss = "KL",lr=0.001,tempr=10,std=0.1, epochs = 10,add_clf_loss = False,params = None):
        #assert dae_loss in ['KL','L2','L1','KL_reverse'], 'Error dae_loss should be seleted within [KL,L2,L1,KL_reverse]'
        if criterion is None:
            if dae_loss == 'L1':
                self.dae_criterion = nn.L1Loss()
            elif dae_loss =='L2':
                self.dae_criterion = nn.L2Loss()
            else:
                self.dae_criterion = kl_loss
            self.dae_checkpoint = self.checkpoint+'/'+dae_loss
            self.dae_loss = dae_loss
        else:
            self.dae_criterion = criterion
            self.dae_loss = dae_loss
            self.tempr = tempr
        if self.dae_type =="recon":
            train_dae(self.model_dae, self.model_clf, self.model_comb, self.trainloader,self.testloader, self.dataset,
                  self.dae_criterion, dae_loss,lr,epochs,tempr,std,self.checkpoint,add_clf_loss)
        else:
            #else means dae_type is denoi!
            self.whitebox_attack(target_model = self.model_clf,dataloader =self.trainloader,batch_size=self.train_batch,params = params)
            #train_adv_noise(self.model_dae,self.model_comb)
            train_dae(self.model_dae, self.model_clf, self.model_comb, self.advloader,self.testloader, self.dataset,
                  self.dae_criterion, dae_loss,lr,epochs,tempr,std,self.checkpoint,add_clf_loss,True)

    def test_dae(self, testloader =None, noise_std=0, tempr=None):
        if tempr is None:
            tempr = self.tempr
        if testloader is None:
            testlaoder = self.testloader
        return test_dae(self.model_clf,self.model_comb,testloader, self.dae_criterion, self.dae_loss,noise_std,tempr)

    def whitebox_attack(self,target_model = None, dataloader = None, batch_size = None, target_object = 'clf',
                        attacker = 'pgd', shuffle = True,
                        params = {'eps':8/255,'niter':20, 'alpha':2/255,'normalize':'0/1'}):
        if params is None:
            params = {'eps': 8 / 255, 'niter': 20, 'alpha': 2 / 255, 'normalize': '0/1'}
        self.params = params
        if target_model is None:
            if target_object == 'clf':
                target_model = self.model_clf
            elif target_object == 'dae':
                target_model = self.model_comb
        if dataloader is None:
            dataloader = self.testloader
        if batch_size is None:
            batch_size = self.test_batch
        print("Start to attack: ", target_object)
        self.advModel = models.WhiteBox(target_model,dataloader,attacker,params)
        self.advloader = DataLoader(self.advModel.advDataSet,batch_size, shuffle=shuffle, num_workers = 0)
        return self.advloader

#def train_adv_noise():

def load_data(dataset = "mnist",transform=None, train_batch=128,
              test_batch = 100, workers=16):
    '''

    :param dataset should be in ['mnist', 'fashion_mnist', 'cifar10','cifar100']:
    :param transform:
    :return:
    '''
    datapath = './data/'+dataset
    print('==> preparing dataset %s', dataset)

    if dataset == 'mnist':
        dataloader = datasets.MNIST
    elif dataset == 'fashion_mnist':
        dataloader = datasets.FashionMNIST
    elif dataset == 'cifar10':
        dataloader = datasets.CIFAR10
    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100

    if transform is not None and dataset in ['cifar10', 'cifar100']:
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

    trainset = dataloader(root=datapath, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=workers)

    testset = dataloader(root=datapath, train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=True, num_workers=workers)

    return trainloader, testloader

def load_model(dataset = "mnist",checkpoint = 'checkpoint', load_clf=None,dae_type = 'recon',noise_level = 8/255):
    checkpoint = checkpoint+'/'+dataset

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    assert dataset in model_names, 'Error: should choice models among '+model_names
    print(models.__dict__[dataset])
    if dataset == 'mnist':
        model_clf,model_dae = models.__dict__[dataset](
            dae_type
        )

    elif dataset == 'fashion_mnist':
        model_clf,model_dae = models.__dict__[dataset](
            dae_type
        )
    elif dataset == 'cifar10':
        model_clf, model_dae = models.__dict__[dataset](
            dae_type
        )
    elif dataset == 'cifar100':
        model_clf,model_dae = models.__dict__[dataset](
            dae_type
        )

    #model_clf = nn.DataParallel(model_clf).cuda()
    #model_dae = nn.DataParallel(model_dae).cuda()

    # Model Load or Train!
    if load_clf:
        print('==> Load the trained classifier..')
        assert os.path.isfile(load_clf), 'Error: no checkpoint directory found!'
        checkpoint = os.path.dirname(load_clf)
        checkpoint = torch.load(load_clf)
        model_clf.load_state_dict(checkpoint['state_dict'])
    model_comb = models.combine_model(model_dae=model_dae, model_clf = model_clf,dae_type=dae_type,e=noise_level)
    model_clf = nn.DataParallel(model_clf).cuda()
    model_dae = nn.DataParallel(model_dae).cuda()
    model_comb = nn.DataParallel(model_comb).cuda()
    return model_clf.cuda(), model_dae.cuda(), model_comb.cuda()



def train_clf(model,trainloader,testloader,criterion = nn.CrossEntropyLoss(), lr = 0.001,epochs = 30, checkpoint = 'checkpoint', dataset = "mnist"):
    checkpoint = checkpoint+'/'+dataset
    #optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    best_acc = 0
    logger = Logger(os.path.join(checkpoint,'log.txt'),title = dataset)
    logger.set_names(['Train Loss', 'Train Top1', 'Train Top3', 'Valid Loss', 'Valid Top1', 'Valid Top3'])

    for epoch in range(epochs):

        # Train classifier through one epoch
        losses, top1, top3 = train_clf_step(model,trainloader, epoch, epochs, criterion, optimizer)

        # Calculate Test error!
        test_loss, test_acc, test_acc3 = test_clf(model, testloader, criterion)
        print('Training/Test [loss] %.4f / %.4f, [top1] %.2f / %.2f, [top3] %.2f / %.2f' %
              (losses.avg,test_loss, top1.avg, test_acc, top3.avg, test_acc3))
        #print('Test [loss] %.4f, [top1]')
        logger.append([losses.avg,top1.avg,top3.avg, test_loss, test_acc,test_acc3])
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
    savefig(checkpoint+'/'+dataset)
    print('Best acc:')
    print(best_acc)

def train_clf_step(model,trainloader,epoch,epochs,criterion,optimizer):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    model.train()

    print('\nEpoch: [%d |%d] ' % (epoch + 1, epochs))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # print intermediate top1 and top3 error
        prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1, 3))
        if (batch_idx) % 10 == 1:
            print("Training [Acc] top1: %.2f , top3: %.2f " % ( prec1,  prec3))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        # Compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses, top1, top3

def train_dae(model_dae, model_clf, model_comb, trainloader,testloader,dataset = "mnist",dae_criterion = None,dae_loss="KL", lr = 0.001,
              epochs = 50, tempr = 10, std = 0.1, checkpoint='checkpoint',add_clf_loss = False, adv_expl = False):

    assert dae_criterion is not None, 'Error dae_criterion should be specified ex. nn.L1Loss()'
    #checkpoint = checkpoint+'/'+dataset

    # Use classifier only for evaluation
    model_clf.eval()

    best_acc = 0
    
    ## Training dae part only!
    optimizer = optim.Adam(model_dae.parameters(),lr=lr)
    """
    if dae_loss == 'L1':
        criterion = nn.L1Loss()
        checkpoint = checkpoint+'/dae_l1'
    elif dae_loss =='L2':
        criterion = nn.L2Loss()
        checkpoint = checkpoint+'/dae_l2'
    else:
        criterion = kl_loss
        checkpoint = checkpoint+'/dae_kl'
    """
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    logger = Logger(os.path.join(checkpoint, 'log.txt'), title=dataset)
    logger.set_names(['Train Loss', 'Train Top1', 'Train Top3', 'Valid Loss','Valid Top1', 'Valid Top3'])

    for epoch in range(epochs):
        model_dae.train()
        print('\nEpoch: [%d |%d] ' % (epoch + 1, epochs))
        if adv_expl:
            #Here, trainloader is advloader!!
            losses, top1, top3 = train_adv_step(trainloader, model_clf,model_comb, dae_criterion, optimizer,dae_loss,tempr,add_clf_loss)
        else:
            losses, top1, top3 = train_dae_step(trainloader, model_clf,model_comb, dae_criterion, optimizer,dae_loss,tempr,std,add_clf_loss)

        # Test models
        # test error for clean images and reconstructed images
        test_loss, test_acc, test_acc3 = test_dae(model_clf,model_comb, testloader, dae_criterion,dae_loss, False,tempr,add_clf_loss)
        #test error for Ganssian corrupted images
        test_loss_n, test_acc_n, test_acc3_n = test_dae(model_clf,model_comb, testloader,dae_criterion,dae_loss, std, tempr,add_clf_loss)

        print('Training/cleanTest/noiseTest [loss] %.4f / %.4f / %.4f, [top1] %.2f / %.2f / %.2f, [top3] %.2f / %.2f / %.2f' %
              (losses.avg, test_loss,test_loss_n,  top1.avg,  test_acc, test_acc_n, \
                top3.avg,  test_acc3, test_acc3_n))
        # print('Test [loss] %.4f, [top1]')
        logger.append([losses.avg, top1.avg, top3.avg, test_loss_n, test_acc_n, test_acc3_n])
        # save_model
        is_best = top1.avg > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_comb.state_dict(),
            'acc': top1.avg,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),

        }, is_best, checkpoint=checkpoint)
    logger.close()
    logger.plot()
    savefig(checkpoint+'/'+dataset)
    print('Best acc:')
    print(best_acc)
            # Test accuracy!

def train_dae_step(data_loader, model_clf,model_comb,criterion, optimizer,dae_loss,tempr,std,add_clf_loss = False):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    if add_clf_loss:
        criterion_clf = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        noise_inputs = noise(inputs,std)
        noise_inputs, inputs, targets = noise_inputs.cuda(), inputs.cuda(), targets.cuda()
        noise_inputs, inputs, targets = Variable(noise_inputs), Variable(inputs), Variable(targets)

        outputs = model_clf(inputs)
        #noise_ = model_dae(noise_inputs)
        #denoise_inputs = torch.clamp(noise_inputs + noise_, 0, 1)
        #denoise_inputs = model_dae(noise_inputs)
        _,outputs_ = model_comb(noise_inputs)
        

        if dae_loss == 'KL':
            loss = criterion(outputs_ / tempr, outputs / tempr)
        elif dae_loss == 'KL_reverse':
            loss = criterion(outputs_ / tempr, outputs / tempr, True)
        elif dae_loss == 'tripleKL':
            _,outputs_c = model_comb(inputs)
            loss1 = criterion(outputs_/tempr, outputs/tempr)
            loss2 = criterion(outputs_c/tempr,outputs/tempr)
            loss = 0.5*loss1+0.5*loss2
        elif dae_loss == 'tripleKL_reverse':
            _,outputs_c = model_comb(inputs)
            loss1 = criterion(outputs/tempr, outputs_/tempr)
            loss2 = criterion(outputs/tempr,outputs_c/tempr)
            loss = 0.5*loss1+0.5*loss2
        else:
            loss = criterion(outputs_,outputs.detach())

        if add_clf_loss:
            loss = 0.5*loss + 0.5*criterion_clf(outputs_, targets)

        prec1, prec3 = accuracy(outputs_.data, targets.data, topk=(1, 3))
        if (batch_idx) % 10 == 1:
            print("Training [loss] %.4f [Acc] top1: %.2f , top3: %.2f " % (loss,  prec1,  prec3))

        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        # Training the models
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses, top1, top3

def train_adv_step(advloader,model_clf,model_comb,criterion,optimizer,dae_loss,tempr,add_clf_loss = False):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    if add_clf_loss:
        criterion_clf = nn.CrossEntropyLoss()
    for batch_idx, (inputs,adv_imgs, targets) in enumerate(advloader):
        #noise_inputs = noise(inputs, std)
        adv_imgs, inputs, targets = adv_imgs.cuda(), inputs.cuda(), targets.cuda()
        adv_imgs, inputs, targets = Variable(adv_imgs), Variable(inputs), Variable(targets)

        outputs = model_clf(inputs)
        # noise_ = model_dae(noise_inputs)
        # denoise_inputs = torch.clamp(noise_inputs + noise_, 0, 1)
        # denoise_inputs = model_dae(noise_inputs)
        _, outputs_ = model_comb(adv_imgs)

        if dae_loss == 'KL':
            loss = criterion(outputs_ / tempr, outputs / tempr)
        elif dae_loss == 'KL_reverse':
            loss = criterion(outputs_ / tempr, outputs / tempr, True)
        elif dae_loss == 'tripleKL':
            _, outputs_c = model_comb(inputs)
            loss1 = criterion(outputs_ / tempr, outputs / tempr)
            loss2 = criterion(outputs_c / tempr, outputs / tempr)
            loss = 0.5 * loss1 + 0.5 * loss2
        elif dae_loss == 'tripleKL_reverse':
            _, outputs_c = model_comb(inputs)
            loss1 = criterion(outputs / tempr, outputs_ / tempr)
            loss2 = criterion(outputs / tempr, outputs_c / tempr)
            loss = 0.5 * loss1 + 0.5 * loss2
        else:
            loss = criterion(outputs_, outputs.detach())

        if add_clf_loss:
            loss = 0.5 * loss + 0.5 * criterion_clf(outputs_, targets)

        prec1, prec3 = accuracy(outputs_.data, targets.data, topk=(1, 3))
        if (batch_idx) % 10 == 1:
            print("Training [loss] %.4f [Acc] top1: %.2f , top3: %.2f " % (loss, prec1, prec3))

        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        # Training the models
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses, top1, top3
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
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1,3))
        losses.update(loss.data[0],inputs.size(0))
        top1.update(prec1[0],inputs.size(0))
        top3.update(prec3[0],inputs.size(0))



    return losses.avg, top1.avg, top3.avg

def test_clf3(model, tripleloader, criterion):
    testloader = tripleloader
    losses = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()
    top12 = AverageMeter()
    top3 = AverageMeter()
    top32 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs,inputs2, targets) in enumerate(testloader):
        inputs,inputs2, targets = inputs.cuda(),inputs2.cuda(), targets.cuda()
        inputs,inputs2, targets = Variable(inputs),Variable(inputs2), Variable(targets)

        outputs = model(inputs)
        outputs2 = model(inputs2)
        loss = criterion(outputs, targets)
        loss2 = criterion(outputs2,targets)
        prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1,3))
        prec12,prec32 = accuracy(outputs2.data,targets.data,topk=(1,3))
        losses.update(loss.data[0],inputs.size(0))
        top1.update(prec1[0],inputs.size(0))
        top3.update(prec3[0],inputs.size(0))
        losses2.update(loss2.data[0],inputs2.size(0))
        top12.update(prec12[0],inputs2.size(0))
        top32.update(prec32[0],inputs2.size(0))



    return (losses.avg, top1.avg, top3.avg),(losses2.avg,top12.avg,top32.avg)


def test_dae(model_clf, model_comb, testloader, criterion,dae_loss, noise_std=False, tempr=1,add_clf_loss = False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    criterion_clf = nn.CrossEntropyLoss()
    model_clf.eval()
    model_comb.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if noise_std>0:
            noise_inputs = noise(inputs, noise_std)
        else:
            noise_inputs = inputs
        noise_inputs, inputs, targets = noise_inputs.cuda(), inputs.cuda(), targets.cuda()
        noise_inputs, inputs, targets = Variable(noise_inputs), Variable(inputs), Variable(targets)

        outputs = model_clf(inputs)
        _,outputs_ = model_comb(noise_inputs)

        if dae_loss == 'KL':
            loss = criterion(outputs_ / tempr, outputs / tempr)
        elif dae_loss == 'KL_reverse':
            loss = criterion(outputs_ / tempr, outputs / tempr, True)
        elif dae_loss == 'tripleKL':
            _, outputs_c = model_comb(inputs)
            loss1 = criterion(outputs_ / tempr, outputs / tempr)
            loss2 = criterion(outputs_c / tempr, outputs / tempr)
            loss = 0.5 * loss1 + 0.5 * loss2
        elif dae_loss == 'tripleKL_reverse':
            _, outputs_c = model_comb(inputs)
            loss1 = criterion(outputs / tempr, outputs_ / tempr)
            loss2 = criterion(outputs / tempr, outputs_c / tempr)
            loss = 0.5 * loss1 + 0.5 * loss2
        else:
            loss = criterion(outputs_, outputs.detach())

        if add_clf_loss:
            loss = 0.5 * loss + 0.5 * criterion_clf(outputs_, targets)

        """
        if dae_loss == 'KL':
            loss = criterion(outputs_ / tempr, outputs / tempr)
        elif dae_loss == 'KL_reverse':
            loss = criterion(outputs_ / tempr, outputs / tempr, True)
        else:
            loss = criterion(outputs, outputs_)
        """
        prec1, prec3 = accuracy(outputs_.data, targets.data, topk=(1, 3))
        losses.update(loss.data[0],inputs.size(0))
        top1.update(prec1[0],inputs.size(0))
        top3.update(prec3[0],inputs.size(0))

    return losses.avg, top1.avg, top3.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint',filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint,filename)
    torch.save(state,filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,'model_best.pth.tar'))


def noise(X, noise_std,low=0, high=1):
    X_noise = X+noise_std*torch.randn_like(X)
    return torch.clamp(X_noise, low,high)


if __name__ =='__main__':
    main()



