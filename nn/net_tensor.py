# !/usr/bin/env python
# -*- coding: utf-8 -*-

# 用pytorch实现numpy_net.py相同功能
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in).type(torch.FloatTensor)
y = torch.randn(N, D_out).type(torch.FloatTensor)

w1 = torch.randn(D_in, H).type(torch.FloatTensor)
w2 = torch.randn(H, D_out).type(torch.FloatTensor)

learning_rate = 1e-6
iter = 500

for t in range(iter):
	# forward
	# torch.mm Performs a matrix multiplication of the matrices mat1 and mat2. 实现两个张量相乘
	h = x.mm(w1) # shape(N, H)
	# torch.clamp 和 np.maximum()作用相同
	h_relu = h.clamp(min=0) # shape(N, H)
	y_pred = h_relu.mm(w2)  # shape(N, D_out)
	
	# compute loss and print
	# torch.pow(n) 求n次方
	loss = (y_pred - y).pow(2).sum()
	print(t, loss)
	
	# backprop
	grad_y_pred = 2.0 * (y_pred - y)
	# torch.t() 求逆矩阵
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	# torch.clone() 科隆
	grad_h = grad_h_relu.clone()
	grad_h[h<0] = 0
	grad_w1 = x.t().mm(grad_h)

	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2

