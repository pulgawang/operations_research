# -*- coding: utf-8 -*-
"""
@Author  : Pulga Wang
@Contact : pulgawang@163.com
@Time    : 2022/10/17 16:49
@Description : 主类
"""
from argparse import ArgumentParser

from optim import get_optim_func
from utils import (
    load_data_fashion_mnist,
    get_alexnet,
    get_cross_entropy_loss,
    try_gpu,
    train,
)


def main(optim_name):
    # 设定batch_size，读取数据
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=224)

    # 获取模型，并且初始化
    net = get_alexnet()

    # 训练
    lr = 0.005
    num_epochs = 20
    optimizer = get_optim_func(name=optim_name)(net.parameters(), lr=lr)
    loss = get_cross_entropy_loss()
    train(net, train_iter, test_iter, loss, num_epochs, optimizer, try_gpu(), filename=f"{optim_name}.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--optim", required=True, choices=["sgd", "adam", "sgd_torch", "adam_torch"])
    args = parser.parse_args()
    main(args.optim)
