# !/usr/bin/env python
# -*- coding:utf-8 -*-

# 使用autograd的自动差分来自动计算反向传播
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(torch.FloatTensor), requires_grad = True)
y = Variable(torch.randn(N, D_out).type(torch.FloatTensor), requires_grad = True)

w1 = Variable(torch.randn(D_in, H).type(torch.FloatTensor), requires_grad = True)
w2 = Variable(torch.randn(H, D_out).type(torch.FloatTensor),requires_grad = True)

learning_rate = 1e-6
iter = 500
for t in range(iter):
	y_pred = x.mm(w1).clamp(min=0).mm(w2)
	
	loss = (y_pred - y).pow(2).sum()
	print(t, loss)
	
	loss.backward()
	
	# update weights
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data
	
	# 清空本次w1，w2的梯度值
	w1.grad.data.zero_()
	w2.grad.data.zero_()
