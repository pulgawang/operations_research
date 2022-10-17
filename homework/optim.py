# -*- coding: utf-8 -*-
"""
@Author  : Pulga Wang
@Contact : pulgawang@163.com
@Time    : 2022/10/17 16:48
@Description : 优化方法实现
"""
from typing import List, Optional

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
            lr = group["lr"]
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                alpha = -lr
                param.add_(d_p, alpha=alpha)

class Adam(Optimizer):
    pass

class NAdam(Optimizer):
    pass