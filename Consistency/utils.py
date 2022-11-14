"""
Created on Fri Sep 30 2022

@author: fanghenshao
"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import random

import os
import sys

sys.path.append("model")
import cifar_resnet
import cifar_resnet_mhead
import imgnet_resnet_mhead

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.models as tmodels
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from typing import List

# -------- fix random seed 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

########################################################################################################
########################################################################################################
########################################################################################################

def cifar10_dataloaders(data_dir, batch_size=128):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, test_loader

def imagenet_dataloaders(data_dir, batch_size=128):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_path = os.path.join(data_dir, 'ILSVRC2012_img_train')
    test_path = os.path.join(data_dir, 'ILSVRC2012_img_val')

    train_set = ImageFolder(train_path, transform=train_transform)
    test_set = ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=False)

    return train_loader, test_loader

def get_datasets(args):
    if args.dataset == 'CIFAR10':
        return cifar10_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    elif args.dataset == 'ImageNet':
        return imagenet_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    else:
        assert False, "Unknown dataset : {}".format(args.dataset)


########################################################################################################
########################################################################################################
########################################################################################################

def get_model(args):
    if args.dataset == 'CIFAR10':
        args.num_classes = 10
    elif args.dataset == 'ImageNet':
        args.num_classes = 1000
    else:
        assert False, "Unknown dataset : {}".format(args.dataset)
    
    normalize_layer = get_normalize_layer(args.dataset)

    if args.arch == 'R110' and args.dataset == 'CIFAR10':
        net = cifar_resnet.resnet(depth=110,num_classes=args.num_classes)
    elif args.arch == 'R50' and args.dataset == 'ImageNet':
        net = torch.nn.DataParallel(tmodels.resnet50(pretrained=False))
        cudnn.benchmark=True

    else:
        assert False, "Unknown model : {}".format(args.arch)

    return torch.nn.Sequential(normalize_layer, net)

def get_model_mhead(args):
    if args.dataset == 'CIFAR10':
        args.num_classes = 10
    elif args.dataset == 'ImageNet':
        args.num_classes = 1000
    else:
        assert False, "Unknown dataset : {}".format(args.dataset)
    
    normalize_layer = get_normalize_layer(args.dataset)

    if args.arch == 'R110' and args.dataset == 'CIFAR10':
        net = cifar_resnet_mhead.__dict__['resnet'](depth=110,num_classes=args.num_classes,num_heads=args.num_heads)
    elif args.arch == 'R20' and args.dataset == 'CIFAR10':
        net = cifar_resnet_mhead.__dict__['resnet'](depth=20,num_classes=args.num_classes,num_heads=args.num_heads)
    elif args.arch == 'R50' and args.dataset == 'ImageNet':
        net = imgnet_resnet_mhead.__dict__['resnet50'](num_classes=args.num_classes,num_heads=args.num_heads)
        net = torch.nn.DataParallel(net)
        cudnn.benchmark=True
    else:
        assert False, "Unknown model : {}".format(args.arch)

    return torch.nn.Sequential(normalize_layer, net)
########################################################################################################
########################################################################################################
########################################################################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

########################################################################################################
########################################################################################################
########################################################################################################

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

########################################################################################################
########################################################################################################
########################################################################################################

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
        and dividing by the dataset standard deviation.

        In order to certify radii in original coordinates rather than standardized coordinates, we
        add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
        layer of the classifier rather than as a part of preprocessing as is typical.
        """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds

def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "ImageNet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "CIFAR10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    else:
        assert False, "Unknown dataset : {}".format(dataset)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

########################################################################################################
########################################################################################################
########################################################################################################
