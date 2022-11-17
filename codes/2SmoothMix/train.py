from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import argparse
import numpy as np

from utils import setup_seed
from utils import get_datasets, get_model, get_model_mhead
from utils import Logger
from utils import AverageMeter, accuracy

from circular_teaching_smoothmix import ct_loss, log10_scheduler
from smoothmix import SmoothMix_PGD

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Training SPACTE')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/data/CIFAR10/',help='file path for data')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='folder to store logs')
parser.add_argument('--save_dir',type=str,default='./save/',help='folder to save model')
parser.add_argument('--runs_dir',type=str,default='./runs/',help='folder to save tensorboard')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    
parser.add_argument('--lr_init',type=float,default=0.1,help='learning rate (default: 0.1)')
parser.add_argument('--wd',type=float,default=1e-4,help='weight decay')
parser.add_argument('--epochs',type=int,default=150,help='number of epochs to train (default: 100)')
parser.add_argument('--save_freq',type=int,default=40,help='model save frequency (default: 20 epoch)')
parser.add_argument('--lr_step_size',type=int,default=50,help='How often to decrease learning by gamma.')
parser.add_argument('--gamma',type=float,default=0.1,help='LR is multiplied by gamma on schedule.')
parser.add_argument('--noise_sd',default=0.0,type=float,help="standard deviation of Gaussian noise for data augmentation")
# -------- multi-head param. --------
parser.add_argument('--num_heads',type=int,default=10,help='number of heads')
parser.add_argument('--alpha',type=float,default=1.0,help='coefficient of the cosine regularization term')
parser.add_argument('--eps',type=float,default=0.8,help='epsilon to control weight distribution')
parser.add_argument('--num_noise_vec',default=2,type=int,help="number of noise vectors. `m` in the paper.")
parser.add_argument('--lbdlast',type=float,default=0.5,help='the last value of lambda')
# -------- smoothmix --------
# Attack params
parser.add_argument('--attack_alpha',default=1.0,type=float,help="step-size for adversarial attacks.")
parser.add_argument('--num_steps',default=8,type=int,help="number of attack updates. `T` in the paper.")
parser.add_argument('--eta',default=1.0,type=float,help="hyperparameter to control the relative strength of the mixup loss.")
parser.add_argument('--mix_step',default=0,type=int, help="which sample to use for the clean side. `1` means to use of one-step adversary.")
parser.add_argument('--maxnorm_s',default=None,type=float)
parser.add_argument('--maxnorm',default=None,type=float)
parser.add_argument('--warmup',default=10,type=int)

args = parser.parse_args()

# ======== log writer init. ========
datanoise='noise-'+str(args.noise_sd)
hyperparam='h-'+str(args.num_heads)+'-eps-'+str(args.eps)+'-m-'+str(args.num_noise_vec)+'-lbdlast-'+str(args.lbdlast)
writer = SummaryWriter(os.path.join(args.runs_dir,args.dataset,args.arch,datanoise,hyperparam+'/'))
if not os.path.exists(os.path.join(args.save_dir,args.dataset,args.arch,datanoise,hyperparam)):
    os.makedirs(os.path.join(args.save_dir,args.dataset,args.arch,datanoise,hyperparam))
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,datanoise,'train')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,datanoise,'train'))
args.save_path = os.path.join(args.save_dir,args.dataset,args.arch,datanoise,hyperparam)
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,datanoise,'train',hyperparam+'-train.log')
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)



# -------- main function
def main():

    # ======== fix random seed ========
    setup_seed(666)
    
    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== initialize net
    net = get_model_mhead(args).cuda()
    print('-------- MODEL INFORMATION --------')
    print('---- arch.: '+args.arch)
    print('---- num_heads: '+str(args.num_heads))

    # ======== initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    # ======== initialize attacker
    if args.maxnorm_s is None:
        args.maxnorm_s = args.attack_alpha * args.mix_step
    attacker = SmoothMix_PGD(steps=args.num_steps, mix_step=args.mix_step, alpha=args.attack_alpha, maxnorm=args.maxnorm, maxnorm_s=args.maxnorm_s)
    
    print('-------- START TRAINING --------')
    for epoch in range(1, args.epochs+1):
        args.warmup_v = np.min([1.0, epoch / args.warmup])
        attacker.maxnorm_s = args.warmup_v * args.maxnorm_s

        # -------- train
        print('Training(%d/%d)...'%(epoch, args.epochs))
        train_epoch(net, trainloader, optimizer, epoch, attacker)
        scheduler.step()

        # -------- validation
        print('Validating...')
        valstats = {}
        acc_te = val(net, testloader)
        acc_te_str = ''
        for idx in range(args.num_heads):
            valstats['cleanacc-path-%d'%idx] = acc_te[idx].avg
            acc_te_str += '%.2f'%acc_te[idx].avg+'\t'
        writer.add_scalars('valacc', valstats, epoch)
        print('     Current test acc. of each head: \n'+acc_te_str)

        # -------- save model & print info
        if (epoch == 1 or epoch % args.save_freq == 0 or epoch == args.epochs):
            checkpoint = {'state_dict': net.state_dict()}
            args.model_path = 'epoch%d'%epoch+'.pth'
            torch.save(checkpoint, os.path.join(args.save_path,args.model_path))

        print('Current training %s of %d heads on data set %s.'%(args.arch, args.num_heads, args.dataset))
        print('===========================================')
    print('Finished training: ', args.save_path)

    
    return

def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)

def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]

def _mixup_data(x1, x2, y1, n_classes):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    device = x1.device

    _eye = torch.eye(n_classes, device=device)
    _unif = _eye.mean(0, keepdim=True)
    lam = torch.rand(x1.size(0), device=device) / 2

    mixed_x = (1 - lam).view(-1, 1, 1, 1) * x1 + lam.view(-1, 1, 1, 1) * x2
    mixed_y = (1 - lam).view(-1, 1) * y1 + lam.view(-1, 1) * _unif

    return mixed_x, mixed_y

# modified for mhead
def _avg_softmax(all_logits):
    num_heads = len(all_logits)
    m = len(all_logits[0])
    # obtain avg. softmax of each head
    avg_softmax = []
    for logits in all_logits:
        softmax = [F.softmax(logit, dim=1) for logit in logits]
        avg_softmax.append(sum(softmax) / m)
    return avg_softmax

def train_epoch(net, trainloader, optimizer, epoch, attacker):
    net.train()
    requires_grad_(net, True)

    batch_time = AverageMeter()
    losses, losses_ortho, losses_mix = AverageMeter(), AverageMeter(), AverageMeter()
    losses_ce = []
    for idx in range(args.num_heads):
        losses_ce.append(AverageMeter())
    losses_ce.append(AverageMeter())

    end = time.time()
    for batch_idx, batch in enumerate(trainloader):
        
        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for b_data, b_label in mini_batches:
            # -------- move to gpu
            b_data, b_label = b_data.cuda(), b_label.cuda()
            
            noises = [torch.randn_like(b_data).cuda() * args.noise_sd for _ in range(args.num_noise_vec)]

            # -------- generate adv. examples
            requires_grad_(net, False)
            net.eval()
            b_data, b_data_adv = attacker.attack(net, b_data, b_label, noises=noises)
            net.train()
            requires_grad_(net, True)

            # -------- get clean and compute loss
            in_clean_c = torch.cat([b_data + noise for noise in noises], dim=0)
            all_logits_c = net(in_clean_c)
            # b_label_c = b_label.repeat(args.num_noise_vec)

            # -------- compute the ce loss via circular-teaching
            all_logits_c_chunk = [torch.chunk(logits_c, args.num_noise_vec, dim=0) for logits_c in all_logits_c]

            threshold = log10_scheduler(current_epoch=epoch, total_epoch=args.epochs, num_classes=args.num_classes, lbd_last=args.lbdlast)
            loss_ce, all_losses, coeffs_head = ct_loss(all_logits_c_chunk, b_label, args.eps, threshold)
            for idx in range(args.num_heads):
                losses_ce[idx].update(all_losses[idx].float().item(), b_data.size(0))


            # -------- get clean avg. softmax for each head
            clean_avg_sm = _avg_softmax(all_logits_c_chunk)
            clean_avg_sm = sum([clean_avg_sm[idx]*coeffs_head[idx] for idx in range(args.num_heads)])

            in_mix, targets_mix = _mixup_data(b_data, b_data_adv, clean_avg_sm, args.num_classes)
            in_mix_c = torch.cat([in_mix + noise for noise in noises], dim=0)
            targets_mix_c = targets_mix.repeat(args.num_noise_vec, 1)
            all_logits_mix = [F.log_softmax(logit,dim=1) for logit in net(in_mix_c)]

            _, top1_idx = torch.topk(clean_avg_sm, 1)
            ind_correct = (top1_idx[:, 0] == b_label).float()
            ind_correct = ind_correct.repeat(args.num_noise_vec)

            # loss_mixup = F.kl_div(logits_mix_c, targets_mix_c, reduction='none').sum(1)
            loss_mixup = [(ind_correct * F.kl_div(logits_mix, targets_mix_c, reduction='none').sum(1)).mean() for logits_mix in all_logits_mix]
            # loss = loss_xent.mean() + args.eta * args.warmup_v * (ind_correct * loss_mixup).mean()
            loss_mixup = sum(loss_mixup[idx]*coeffs_head[idx] for idx in range(args.num_heads))











            #####################################################################################################
            # b_data_c = torch.cat([b_data + noise for noise in noises], dim=0)

            # all_logits = net(b_data_c)

            # # -------- compute the ce loss via circular-teaching
            # all_logits_chunk = [torch.chunk(logits, args.num_noise_vec, dim=0) for logits in all_logits]

            # threshold = log10_scheduler(current_epoch=epoch, total_epoch=args.epochs, num_classes=args.num_classes, lbd_last=args.lbdlast)
            # loss_ce, all_losses = ct_loss(all_logits_chunk, b_label, args.eps, threshold)
            # for idx in range(args.num_heads):
            #     losses_ce[idx].update(all_losses[idx].float().item(), b_data.size(0))
            
            # -------- compute the ORTHOGONALITY constraint
            loss_ortho = .0
            if args.num_heads > 1:
                loss_ortho = net[1].compute_cosin_loss()

            # -------- SUM the two losses
            total_loss = loss_ce + args.alpha * loss_ortho + args.eta * args.warmup_v * loss_mixup
        
            # -------- backprop. & update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # -------- record & print in termial
            losses.update(total_loss, b_data.size(0))
            losses_ce[args.num_heads].update(loss_ce.float().item(), b_data.size(0))
            losses_ortho.update(loss_ortho, b_data.size(0))
        # ----
        batch_time.update(time.time()-end)
        end = time.time()

    losses_ce_record = {}
    losses_ce_str = ''
    for idx in range(args.num_heads):
        losses_ce_record['head-%d'%idx] = losses_ce[idx].avg
        losses_ce_str += "%.4f"%losses_ce[idx].avg +'\t'
    losses_ce_record['avg.'] = losses_ce[args.num_heads].avg
    writer.add_scalars('loss-ce', losses_ce_record, epoch)
    writer.add_scalar('loss-ortho', losses_ortho.avg, epoch)
    print('     Epoch %d/%d costs %fs.'%(epoch, args.epochs, batch_time.sum))
    print('     CE      loss of each head: \n'+losses_ce_str)
    print('     Avg. CE loss = %f.'%losses_ce_record['avg.'])
    print('     ORTHO   loss = %f.'%losses_ortho.avg)

    return

def val(net, dataloader):
    
    net.eval()

    batch_time = AverageMeter()

    acc = []
    for idx in range(args.num_heads):
        measure = AverageMeter()
        acc.append(measure)
    
    end = time.time()
    with torch.no_grad():
        
        # -------- compute the accs.
        for test in dataloader:
            images, labels = test
            images, labels = images.cuda(), labels.cuda()
            images = images + torch.randn_like(images).cuda() * args.noise_sd

            # ------- forward 
            all_logits = net(images)
            for idx in range(args.num_heads):
                logits = all_logits[idx]
                logits = logits.detach().float()

                prec1 = accuracy(logits.data, labels)[0]
                acc[idx].update(prec1.item(), images.size(0))
            
            # ----
            batch_time.update(time.time()-end)
            end = time.time()

    print('     Validation costs %fs.'%(batch_time.sum))        
    return acc

# ======== startpoint
if __name__ == '__main__':
    main()