# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

#[一维自动微分]#
# create tensors
# 创建张量
x = Variable(torch.Tensor([1]), requires_grad = True) # x=1
w = Variable(torch.Tensor([2]), requires_grad = True) # w=2
b = Variable(torch.Tensor([3]), requires_grad = True) # b=3

# 创建一个函数
y = w * x + b  #y = 2x+3 或 y = w+3 或 y = 2+b

y.backward()

print(x.grad)    # y关于x的微分@y/@x|x=1 =  2
print(w.grad)    # y关于w的微分@y/@w|w=2 =  1
print(b.grad)    # y关于b的微分@y/@b|b=3 =  1

#[多维自动微分]#
x = Variable(torch.ones(3,3), requires_grad = True) # x是3*3值均为1的矩阵
y = x * x + 2
z = y.mean()	# 求平均数
z.backward()

print(x.grad)   # z关于x的微分@z/@x|x=3*3矩阵 = 值为2/9 = 0.2222的3*3矩阵 

