# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# y=Ax+b
m = nn.Linear(2, 2)
x = torch.ones(2, 2)
print(x)
_input = autograd.Variable(x)
output = m(_input)
print(output)