# 运筹学课堂作业——实现NADAM

在adam基础上加入Nesterov动量

> Dozat, Timothy. "Incorporating nesterov momentum into adam." (2016).

## 1. 问题描述

分别实现SGD、Adam和NAdam，并使用AlexNet在Fashion Minist分类问题上进行测试，考察不同的优化算法在非凸优化问题上的loss下降情况。

### 1.1. 实验环境
 
 - 数据集：Fashion Minist图像
 - 模型：AlexNet
 - Loss函数：Cross Entropy
 - 计算卡：NVIDIA 1080ti（cuda:11.7）（CPU亦可）
 - 语言：Python 3.8.13
 - 主要依赖库：Python torch 1.12.1+cu102（CPU为1.12.1）
 - 参考文献：
   - Dozat, Timothy. "Incorporating nesterov momentum into adam." (2016).
   - Zhang, Aston, et al. "Dive into deep learning." arXiv preprint arXiv:2106.11342 (2021).

### 1.2. 实验过程

```shell
$ cd operations_research
$ pip install -r requirements.txt
$ python homework/main.py --optim sgd
```

## 2. 实验过程
### 2.1. SGD实现
```python
# 更新参数，减去lr*梯度
# d_p_list为params的梯度，lr为learning_rate
for i, param in enumerate(params):
    d_p = d_p_list[i]
    alpha = -lr
    param.add_(d_p, alpha=alpha)
```
### 2.2. Adam实现

### 2.3. NAdam实现

## 3. 实验结果
