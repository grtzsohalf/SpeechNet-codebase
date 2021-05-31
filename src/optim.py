import torch
import numpy as np
from functools import partial
from src.optimizer_utils import *

class Optimizer():
    def __init__(self, parameters, optimizer, lr, eps, lr_scheduler, lr_scheduler_option, tf_start=1, tf_end=1, tf_step=1, 
                 betas=[0.99, 0.98], d_model=512, warmup_step=25000, **kwargs):

        # Setup teacher forcing scheduler
        self.tf_type = tf_end != 1
        self.tf_rate = lambda step: max(
            tf_end, tf_start-(tf_start-tf_end)*step/tf_step)

        # Setup torch optimizer
        self.opt_type = optimizer
        self.init_lr = lr
        self.sch_type = lr_scheduler
        self.lr_scheduler_option = lr_scheduler_option
        opt = getattr(torch.optim, optimizer)
        if lr_scheduler == 'warmup':
            # init_lr = lr
            # self.lr_scheduler = lambda step: init_lr * (d_model ** (-0.5)) * \
            #     np.minimum((step+1)*warmup_step**(-1.5), (step+1)**(-0.5))
            if optimizer == 'Adam':
                self.opt = opt(parameters, lr=lr, eps=eps, betas=betas)
            else:
                self.opt = opt(parameters, lr=lr, eps=eps)
            self.lr_scheduler = eval(self.lr_scheduler_option)(self.opt,warmup_step,kwargs['max_step'])

        elif lr_scheduler == 'spec-aug-basic':
            # Scheduler from https://arxiv.org/pdf/1904.08779.pdf
            self.lr_scheduler = partial(speech_aug_scheduler, s_r=500,
                                        s_i=20000, s_f=80000, peak_lr=lr)
            self.opt = opt(parameters, lr=lr, eps=eps)
        elif lr_scheduler == 'spec-aug-double':
            # Scheduler from https://arxiv.org/pdf/1904.08779.pdf
            self.lr_scheduler = partial(speech_aug_scheduler, s_r=1000,
                                        s_i=40000, s_f=160000, peak_lr=lr)
            self.opt = opt(parameters, lr=lr, eps=eps)
        else:
            self.lr_scheduler = None
            self.opt = opt(parameters, lr=lr, eps=eps)  # ToDo: 1e-8 better?

    def get_param_groups(self):
        return self.opt.param_groups
    
    def get_opt_state_dict(self):
        return self.opt.state_dict()

    def load_opt_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

    def zero_grad(self, set_to_none=False):
        self.opt.zero_grad(set_to_none=set_to_none)

    def post_step(self, step, set_to_none=False):
        if self.lr_scheduler is not None:
            # cur_lr = self.lr_scheduler(step)
            self.lr_scheduler.step()
            cur_lr = self.lr_scheduler.get_last_lr()[0]
            # for param_group in self.opt.param_groups:
            #     param_group['lr'] = cur_lr
        else:
            cur_lr = None
        # self.opt.zero_grad(set_to_none=set_to_none)
        return self.tf_rate(step), cur_lr

    def step(self):
        self.opt.step()

    def create_msg(self):
        return ['Optim.spec.| Algo. = {}\t| Lr = {}\t (Scheduler = {})| Scheduled sampling = {}'
                .format(self.opt_type, self.init_lr, self.lr_scheduler_option, self.tf_type)]

def speech_aug_scheduler(step, s_r, s_i, s_f, peak_lr):
    # Starting from 0, ramp-up to set LR and  converge to 0.01*LR, w/ exp. decay
    final_lr_ratio = 0.01
    exp_decay_lambda = -np.log10(final_lr_ratio)/(s_f-s_i) # Approx. w/ 10-based
    cur_step = step+1

    if cur_step<s_r:
        # Ramp-up
        return peak_lr*float(cur_step)/s_r
    elif cur_step<s_i:
        # Hold
        return peak_lr
    elif cur_step<=s_f:
        # Decay
        return peak_lr*np.power(10,-exp_decay_lambda*(cur_step-s_i))
    else:
        # Converge
        return peak_lr*final_lr_ratio
