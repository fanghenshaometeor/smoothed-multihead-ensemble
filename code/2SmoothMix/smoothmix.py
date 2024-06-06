import torch
import torch.nn.functional as F

from typing import Optional

class SmoothMix_PGD(object):
    def __init__(self,
                 steps: int,
                 mix_step: int,
                 alpha: Optional[float] = None,
                 maxnorm_s: Optional[float] = None,
                 maxnorm: Optional[float] = None) -> None:
        super(SmoothMix_PGD, self).__init__()
        self.steps = steps
        self.mix_step = mix_step
        self.alpha = alpha
        self.maxnorm = maxnorm
        if maxnorm_s is None:
            self.maxnorm_s = alpha * mix_step
        else:
            self.maxnorm_s = maxnorm_s

    def attack(self, model, inputs, labels, noises=None):
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        def _batch_l2norm(x):
            x_flat = x.reshape(x.size(0), -1)
            return torch.norm(x_flat, dim=1)

        def _project(x, x0, maxnorm=None):
            if maxnorm is not None:
                eta = x - x0
                eta = eta.renorm(p=2, dim=0, maxnorm=maxnorm)
                x = x0 + eta
            x = torch.clamp(x, 0, 1)
            x = x.detach()
            return x

        adv = inputs.detach()
        init = inputs.detach()
        for i in range(self.steps):
            if i == self.mix_step:
                init = adv.detach()
            adv.requires_grad_()

            # softmax = [F.softmax(model(adv + noise), dim=1) for noise in noises]
            # avg_softmax = sum(softmax) / len(noises)
            # logsoftmax = torch.log(avg_softmax.clamp(min=1e-20))

            # loss = F.nll_loss(logsoftmax, labels, reduction='sum')

            ######## ##### ########
            ######## ##### ########
            ######## mhead ########
            all_logits = [model(adv+noise) for noise in noises]
            m = len(noises)
            num_heads = model[1].num_heads

            loss = .0
            for head_idx in range(num_heads):
                logits_i_head = [all_logits[m_idx][head_idx] for m_idx in range(m)]
                softmax_i_head = [F.softmax(logit) for logit in logits_i_head]
                avgsoftmax_i_head = sum(softmax_i_head) / m
                logsoftmax_i_head = torch.log(avgsoftmax_i_head.clamp(min=1e-20))
                loss = loss + F.nll_loss(logsoftmax_i_head, labels, reduction='sum')/num_heads

            ######## mhead ########
            ######## ##### ########
            ######## ##### ########

            grad = torch.autograd.grad(loss, [adv])[0]
            grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)
            adv = adv + self.alpha * grad

            adv = _project(adv, inputs, self.maxnorm)
        init = _project(init, inputs, self.maxnorm_s)

        return init, adv