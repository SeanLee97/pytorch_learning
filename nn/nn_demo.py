# !/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in), requires_grad=True)
y = Variable(torch.randn(N, D_out), requires_grad=False)  # 注意MSELoss的target张量requires_grad=False
# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
# 模块将按照构造函数中传递的顺序添加到模块中
model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H), # Applies a linear transformation to the incoming data: y=Ax+b
	torch.nn.ReLU(),
	torch.nn.Linear(H, D_out)
)

# 损失函数用均方误差
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

iter = 500

for t in range(iter):
	
	y_pred = model(x)  # 再次相比之前求y_pred简化了，将所有操作都封装在model中依次执行
	loss = loss_fn(y_pred, y)
	print(t, loss.data[0])
	
	'''
	# 不使用优化器

	# 后向传播前清空梯度
	model.zero_grad()
	# 计算梯度损失
	loss.backward()
	# 更新weight
	for param in model.parameters():
		param.data -= learning_rate * param.grad.data
	'''
	# 使用优化器
	# 在后向传播之前先进行梯度清零
	optimizer.zero_grad()
	# 后向传播，计算梯度下降损失
	loss.backward()
	# 更新weight参数
	optimizer.step()
