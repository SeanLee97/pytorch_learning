# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

'''
@param N 批次大小 batch size
@param D_in 输入维数 input dimension
@param H 隐藏层维数 hidden dimension
@param D_out 输出层维数 output dimension
'''
N, D_in, H, D_out = 64, 1000, 100, 10

# create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# init weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
#print(learning_rate)
iter = 500
  
for t in range(iter):
	# forward pass: compute predicted y
	h = x.dot(w1)  # shape(N, H)
	# 关于np.maximum()的用法可以查看http://blog.csdn.net/lanchunhui/article/details/52700895
	# h的值每一个都和0比较，取较大者
	h_relu = np.maximum(h, 0)  # shape(N,H)
	y_pred = h_relu.dot(w2)    # shape(N, D_out)

	# compute and print loss
	# 平方
	loss = np.square(y_pred - y).sum()
	print(t, loss)

	# backprop to compute gradients of w1 and w2 with respect to loss
	# 根据loss反向传播计算w1, w2的梯度
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.T.dot(grad_y_pred) # h_relu.shape(H, N), shape(H,D_out)
	
	grad_h_relu = grad_y_pred.dot(w2.T) # shape(N, H)
	grad_h = grad_h_relu.copy() # shape(N,H)
	grad_h[h<0] = 0
	grad_w1 = x.T.dot(grad_h)
	
	# update weights
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2
	
		
