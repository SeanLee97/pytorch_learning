# !/usr/bin/env python
# -*- coding: utf-8 -*-
import torch  
from torch.autograd import Variable  
  
tensor = torch.FloatTensor([[1,2],[3,4]])  
variable = Variable(tensor, requires_grad=True)  
# 打印展示Variable类型  
print(tensor)  
print(variable)  

# 求平均数 torch.mean
t_out = torch.mean(tensor*tensor) # 每个元素的^ 2  
v_out = torch.mean(variable*variable)  
print(t_out)  
print(v_out)  
  
v_out.backward() # Variable的误差反向传递  
  
# 比较Variable的原型和grad属性、data属性及相应的numpy形式  
print('variable:\n', variable)  
# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤  
# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2  
print('variable.grad:\n', variable.grad) # Variable的梯度  
print('variable.data:\n', variable.data) # Variable的数据  
print(variable.data.numpy()) #Variable的数据的numpy形式
