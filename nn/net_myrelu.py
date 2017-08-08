# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

# 自定义autograd函数来执行ReLU非线性，并用它来实现我们的两层网络
class MyReLU(torch.autograd.Function):
	# 前向传播
	def forward(self, input):
		"""
        In the forward pass we receive a Tensor containing the input and return a Tensor containing the output. You can cache arbitrary Tensors for use in the backward pass using the save_for_backward method.
        """
		self.save_for_backward(input)
		return input.clamp(min=0)
	
	def backward(self, grad_output):
		"""
        	In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output, and we need to compute the gradient of the loss  with respect to the input.
		"""	
		input, = self.saved_tensors  # 取出forward中save_for_backward
		grad_input = grad_output.clone()
		grad_input[input<0] = 0
		return grad_input

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(torch.FloatTensor), requires_grad = True)
y = Variable(torch.randn(N, D_out).type(torch.FloatTensor), requires_grad = True)

w1 = Variable(torch.randn(D_in, H).type(torch.FloatTensor), requires_grad = True)
w2 = Variable(torch.randn(H, D_out).type(torch.FloatTensor), requires_grad = True)

learning_rate = 1e-6
iter = 500

for t in range(iter):
	relu = MyReLU()
	y_pred = relu((x.mm(w1)).mm(w2))

	loss = (y_pred - y).pow(2).sum()
	print(t, loss.data[0])
	
	loss.backward()
	
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data
	
	w1.grad.data.zero_()
	w2.grad.data.zero_()

