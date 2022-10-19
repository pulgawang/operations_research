# -*- coding: utf-8 -*-
"""
@Author  : Pulga Wang
@Contact : pulgawang@163.com
@Time    : 2022/10/17 16:48
@Description : 优化方法实现
"""
import math

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    """重载Optimizer类"""
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = []
            d_p_list = []
            lr = group["lr"]

            # 计算梯度
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    d_p_list.append(p.grad)

            # 更新参数，减去lr*梯度
            for i, param in enumerate(params):
                d_p = d_p_list[i]
                alpha = -lr
                param.add_(d_p, alpha=alpha)


class Adam(Optimizer):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            lr = group["lr"]
            beta1, beta2 = group['betas']
            eps = group["eps"]

            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    state_steps.append(state['step'])

            for i, param in enumerate(params):

                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]
                # update step
                step_t += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                step = step_t.item()

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)


def get_optim_func(name="sgd"):
    if "sgd_torch" == name:
        return torch.optim.SGD
    elif "adam_torch" == name:
        return torch.optim.Adam
    elif "sgd" == name:
        return SGD
    elif "adam" == name:
        return Adam
    else:
        raise ValueError(f"Unsupported GD: {name}")