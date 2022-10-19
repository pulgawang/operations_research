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
            grads = []
            lr = group["lr"]

            # 计算梯度
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)

            # 更新参数，减去lr*梯度
            for i, param in enumerate(params):
                d_p = grads[i]
                alpha = -lr
                param.add_(d_p, alpha=alpha)


class Adam(Optimizer):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = []  # 参数
            grads = []  # 梯度
            vs = []  # v: 梯度的指数滑动平均数，梯度一阶矩表示
            ss = []  # s: 梯度平方的指数滑动平均数，梯度二阶矩表示
            steps = []
            lr = group["lr"]
            beta1, beta2 = group['betas']
            eps = group["eps"]

            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # 初始化
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        # v初始化:
                        state['vs'] = torch.zeros_like(p)
                        # s初始化
                        state['ss'] = torch.zeros_like(p)

                    vs.append(state['vs'])
                    ss.append(state['ss'])

                    steps.append(state['step'])

            for i, param in enumerate(params):

                grad = grads[i]
                v = vs[i]
                s = ss[i]
                step_t = steps[i]

                step_t += 1
                step = step_t.item()

                # 更新v和s
                v.mul_(beta1).add_(grad, alpha=1 - beta1)  # v
                s.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)  # s

                # 偏差修正，解决初始偏差
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                # 更新梯度
                param.addcdiv_(v, (s.sqrt() / math.sqrt(bias_correction2)).add_(eps), value=-step_size)


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