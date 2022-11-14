"""
Created on Fri Sep 30 2022

@author: fanghenshao
"""

from __future__ import print_function

import torch

import numpy as np
import pandas as pd

import os
import sys
import time
import argparse
import datetime

from utils import setup_seed
from utils import get_datasets, get_model, get_model_mhead
from utils import Logger
from utils import AverageMeter, accuracy

from core_mhead import Smooth

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Certify DIO ensemble')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='folder to store logs')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=128,help='batch size for training (default: 256)')
parser.add_argument('--noise_sd',default=0.0,type=float,help="standard deviation of Gaussian noise for data augmentation")
# -------- certify --------
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--num_heads',type=int,default=10,help='number of orthogonal paths')
args = parser.parse_args()

# ======== log writer init. ========
datanoise='noise-'+str(args.noise_sd)
hyperparam=os.path.split(os.path.split(args.model_path)[-2])[-1]
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,datanoise,'certify-mhead')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,datanoise,'certify-mhead'))
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,datanoise,'certify-mhead',hyperparam+'-certify-skip-%d.log'%(args.skip))


# -------- main function
def main():

    # ======== fix random seed ========
    setup_seed(666)
    
    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== load network ========
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    net = get_model_mhead(args).cuda()
    net.load_state_dict(checkpoint['state_dict'])
    print('-------- MODEL INFORMATION --------')
    print('---- arch.: '+args.arch)
    print('---- num_heads: '+str(args.num_heads))
    print('---- saved path  : '+args.model_path)

    smoothed_classifier = Smooth(net, args.num_classes, args.noise_sd)
    f = open(args.logs_path, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    for i in range(len(testloader.dataset)):
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = testloader.dataset[i]

        before_time = time.time()
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        after_time = time.time()
        correct = int(prediction == label)
    
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
    f.close()

    # ======== get ACR results ========
    ApproAcc = ApproximateAccuracy(args.logs_path)
    if args.dataset == 'CIFAR10':
        radii = np.arange(0,2.75,0.25)
    elif args.dataset == 'ImageNet':
        radii = np.arange(0,4.0,0.5)
    certified = ApproAcc.at_radii(radii)*100
    f = open(args.logs_path.replace("skip-%d"%args.skip, "ACR"), 'w')
    print('\n-------- Log-path: {}'.format(args.logs_path), file=f, flush=True)
    print('\n-------- ACR = %.3f '%ApproAcc.acr(), file=f, flush=True)
    for idx, radius in enumerate(radii):
        print('-------- At radius = %.2f achieving certified radius %.1f'%(radius, certified[idx]), file=f, flush=True)
    
    f.close()

    return

class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()

class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()

    def acr(self):
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return (df["correct"] * df["radius"]).mean()



# ======== startpoint
if __name__ == '__main__':
    main()