# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np

#[张量的使用]#
x = torch.rand(3, 2)
y = torch.rand(3, 2)
# x+y的方式
'''
#1. 
print(x+y) 
'''

'''
#2. 
x.add_(y)
print(x)
'''
#3.
result = torch.Tensor(3, 2)
torch.add(x, y, out=result)
print(result)

# 输出第二列（下标从0开始）
print(x) 
print(x[:, 1])

# tensor to numpy
b = x.numpy()
print("numpy:",b)

# numpy to tensor
a = np.ones((3,3))
b = torch.from_numpy(a)
print('numpy a:\n', a)
print('numpy to tensor b:\n', b)