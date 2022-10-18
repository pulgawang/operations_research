# -*- coding: utf-8 -*-
"""
@Author  : Pulga Wang
@Contact : pulgawang@163.com
@Time    : 2022/10/17 16:47
@Description : 作业相关工具方法
"""
import csv
import time

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=trans, download=True
    )
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
    train_iter.__setattr__("num_workers", 0)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)
    test_iter.__setattr__("num_workers", 0)
    return train_iter, test_iter


def init_weight(m):
    """初始化参数"""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.uniform_(m.weight, 0, 1)  # .xavier_uniform_(m.weight)


def get_alexnet():
    """AlexNet"""
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.LazyLinear(out_features=4096),
        nn.ReLU(),  # nn.Linear(256 * 5 * 5, 4096)
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10),
    )
    net.apply(init_weight)
    return net


def get_cross_entropy_loss():
    return nn.CrossEntropyLoss()


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def caculate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    correct_count = 0.0
    total_count = 0.0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            correct_count += accuracy(net(X), y)
            total_count += y.numel()
    return correct_count / total_count


def train_epoch(net, train_iter, loss, updater, device):
    if isinstance(net, torch.nn.Module):
        net.train()
    correct_count = 0.0
    total_count = 0.0
    total_loss = 0.0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            raise ValueError("Invalid updater")
        correct_count += accuracy(y_hat, y)
        total_count += y.numel()
        total_loss += l.sum()
    return total_loss / total_count, correct_count / total_count


def train(net, train_iter, test_iter, loss, num_epochs, updater, device, filename=None):
    net.to(device)
    results = [["epoch", "train loss", "train acc", "test acc", "escape time"]]
    for epoch in range(num_epochs):
        t1 = time.time()
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater, device)
        test_acc = caculate_accuracy_gpu(net, test_iter)
        t2 = time.time() - t1
        results.append([epoch, train_loss, train_acc, test_acc, t2])
        print(
            f"epoch: {epoch}, train loss: {train_loss:.5f}, train acc: {train_acc:.3f}, test acc: {test_acc:.3f}, escape time: {t2:.1f}s"
        )
    if filename:
        with open(filename, "w") as f:
            write = csv.writer(f)
            write.writerows(results)
