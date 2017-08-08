# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.autograd import Variable

'''
有时您会想要指定比现有模块序列更复杂的模型; 对于这些情况，您可以通过子类化定义自己的模块，nn.Module并定义forward接收输入变量的值，并使用其他模块或变量上的其他自动格式化操作生成输出变量
'''

class TwoLayerNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(TwoLayerNet, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear2 = torch.nn.Linear(H, D_out)
	def forward(self, x):
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))

learning_rate = 1e-6
model = TwoLayerNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
