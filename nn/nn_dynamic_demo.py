# !/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import torch
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(DynamicNet, self).__init__()
		self.input_linear = torch.nn.Linear(D_in, H)
		self.middle_linear = torch.nn.Linear(H, H)
		self.output_linear = torch.nn.Linear(H, D_out)
	def forward(self, x):
		h_relu = self.input_linear(x).clamp(min=0)
		# 使用许多隐藏层，重复使用相同权重多次计算最内层的隐藏层
		for x in range(random.randint(0, 3)):
			h_relu = self.middle_linear(h_relu).clamp(min=0)
		y_pred = self.output_linear(h_relu)
		return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))

model = DynamicNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

iter = 500

for t in range(iter):
	y_pred = model(x)
	loss = loss_fn(y_pred, y)
	print(t, loss.data[0])
	# 清空梯度
	optimizer.zero_grad()
	# 后向传播
	loss.backward()
	# 更新参数
	optimizer.step()

