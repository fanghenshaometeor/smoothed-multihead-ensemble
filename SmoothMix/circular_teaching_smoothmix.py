"""
Created on Fri Sep 30 2022

@author: fanghenshao
"""

from multiprocessing.heap import reduce_arena
import torch
import torch.nn.functional as F 

import numpy as np

import math

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

def log10_scheduler(current_epoch, total_epoch, num_classes, lbd_last=0.5):
    
    init_thres = math.log(num_classes)
    last_thres = lbd_last

    # -log10(a*current_epoch+b)
    # current_epoch rangs from [1, total_epoch] <---- it's initial is 1, NOT 0.

    a = (math.pow(10,-init_thres) - math.pow(10,-last_thres)) / (1-total_epoch)
    b = math.pow(10,-init_thres) - a

    return - math.log(a * current_epoch + b, 10)


def ct_loss(net, b_data, b_data_adv, noises, label, num_classes, num_noise_vec, eps, threshold=0.5):
    m = num_noise_vec
    # -------- get clean and compute loss
    in_clean_c = torch.cat([b_data + noise for noise in noises], dim=0)
    all_logits_c = net(in_clean_c)
    all_logits_c_chunk = [torch.chunk(logits_c, num_noise_vec, dim=0) for logits_c in all_logits_c]

    num_heads = len(all_logits_c_chunk)

    ###########################################################################################################
    coeffs_spl = []
    with torch.no_grad():
        # -------- for each head, compute the avg. loss of m samplings
        avg_losses = []
        for head_idx, logits_chunk in enumerate(all_logits_c_chunk):
            avg_loss = .0
            for logits in logits_chunk:
                avg_loss += F.cross_entropy(logits, label, reduction='none')
            avg_loss = avg_loss / m
            avg_losses.append(avg_loss)
        # -------- circular-teaching among heads
        for head_idx, avg_loss in enumerate(avg_losses):
            coeff_spl = avg_loss.lt(threshold).float()  # <-- easy samples with coeffs. 1
            hard_idx = avg_loss.gt(threshold)           # <-- hard samples
            coeff_spl[hard_idx] = (1+math.exp(-threshold))/(1+torch.exp(avg_losses[head_idx-num_heads+1][hard_idx]-threshold)) # <-- key codes
            coeffs_spl.append(coeff_spl)
    
    # -------- circular-teaching among heads, weighted sample loss
    losses = []
    for head_idx, logits_chunk in enumerate(all_logits_c_chunk):
        loss = sum([F.cross_entropy(logits, label, reduction='none') for logits in logits_chunk]) / m
        wloss = coeffs_spl[head_idx-1].squeeze() * loss # <-- key codes
        wloss = wloss.mean()
        losses.append(wloss)

    ###########################################################################################################
    ###########################################################################################################

    # -------- weighted head loss
    if num_heads > 1:
        with torch.no_grad():
            _, best_head_idx = torch.stack(losses, dim=-1).min(-1)
            best_head_idx = int(best_head_idx.cpu().numpy().squeeze())

            # Occasionally randomly choose a head to avoid idle heads
            if np.random.binomial(1, 0.01):
                best_head_idx = np.random.choice(range(num_heads))
            
            coeffs_head  = [eps / (num_heads - 1) for _ in range(num_heads)]
            coeffs_head[best_head_idx] = 1 - eps
        loss_ce = sum([losses[idx]*coeffs_head[idx] for idx in range(num_heads)])
    else:
        assert False, "number of heads should be greater than 1."


    ###########################################################################################################
    ###########################################################################################################
    # -------- get clean avg. softmax for each head
    clean_avg_sm = _avg_softmax(all_logits_c_chunk)
    clean_avg_sm = sum([clean_avg_sm[idx]*coeffs_head[idx] for idx in range(num_heads)])

    # -------- create mixed data
    in_mix, targets_mix = _mixup_data(b_data, b_data_adv, clean_avg_sm, num_classes)
    in_mix_c = torch.cat([in_mix + noise for noise in noises], dim=0)
    # targets_mix_c = targets_mix.repeat(m, 1)
    all_logits_mix = [F.log_softmax(logit,dim=1) for logit in net(in_mix_c)]
    all_logits_mix_chunk = [torch.chunk(logits_mix, num_noise_vec, dim=0) for logits_mix in all_logits_mix]

    # -------- obtain the kl smoothmix loss
    losses_mix = []
    for head_idx, logits_mix_chunk in enumerate(all_logits_mix_chunk):
        loss_mix_one_head = [F.kl_div(logits_mix, targets_mix, reduction='none').sum(1) for logits_mix in logits_mix_chunk]
        loss_mix_one_head = sum(loss_mix_one_head)/m * coeffs_spl[head_idx-1].squeeze()
        losses_mix.append(loss_mix_one_head)
    loss_mix = sum([losses_mix[idx].mean()*coeffs_head[idx] for idx in range(num_heads)])

    return loss_ce, losses, loss_mix



    
    


