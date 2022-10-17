# -*- coding: utf-8 -*-
"""
@Author  : Pulga Wang
@Contact : pulgawang@163.com
@Time    : 2022/10/17 16:48
@Description : 优化方法实现
"""
import torch


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            grad_change = lr * param.grad
            param -= grad_change / batch_size
            param.grad.zero_()
