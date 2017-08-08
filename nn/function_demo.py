# !/usr/bin/env python
# -*- coding: utf-8 -*-
import torch  
import torch.nn.functional as F  
from torch.autograd import Variable  
import matplotlib.pyplot as plt  
  
x = torch.linspace(-5, 5, 200)  
x_variable = Variable(x) #将x放入Variable  
x_np = x_variable.data.numpy()  
  
# 经过4种不同的激励函数得到的numpy形式的数据结果  
y_relu = F.relu(x_variable).data.numpy()  
y_sigmoid = F.sigmoid(x_variable).data.numpy()  
y_tanh = F.tanh(x_variable).data.numpy()  
y_softplus = F.softplus(x_variable).data.numpy()  
  
plt.figure(1, figsize=(8, 6))  
  
plt.subplot(221)  
plt.plot(x_np, y_relu, c='red', label='relu')  
plt.ylim((-1, 5))  
plt.legend(loc='best')  
  
plt.subplot(222)  
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')  
plt.ylim((-0.2, 1.2))  
plt.legend(loc='best')  
  
plt.subplot(223)  
plt.plot(x_np, y_tanh, c='red', label='tanh')  
plt.ylim((-1.2, 1.2))  
plt.legend(loc='best')  
  
plt.subplot(224)  
plt.plot(x_np, y_softplus, c='red', label='softplus')  
plt.ylim((-0.2, 6))  
plt.legend(loc='best')  
  
plt.show()  
