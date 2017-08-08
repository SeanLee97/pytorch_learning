# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
基于CIFAR10数据集的图片分类器
'''
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import glob, os

'''
配置
'''
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
N_EPOCH = 100
IS_DOWNLOAD = True

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		# kernel
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		# 池化
		self.pool = nn.MaxPool2d(2, 2) 
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5) # reshape
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def show_img(img):
	img = img / 2 + 0.5  # 将图片模糊化
	np_img = img.numpy()
	plt.figure('CNN image classifier')
	plt.imshow(np.transpose(np_img, (1, 2, 0)))
	plt.show()

def save_model(model, epoch, max_keep=5):
	if not os.path.exists('./runtime'):
		os.makedirs('runtime')
	f_list = glob.glob(os.path.join('./runtime', 'model') + '-*.ckpt')
	if len(f_list) >= max_keep + 2:
		# this step using for delete the more than 5 and litter one
		epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
		to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
		for f in to_delete:
			os.remove(f)
	name = 'model-{}.ckpt'.format(epoch)
	file_path = os.path.join('./runtime', name)
	torch.save(model.state_dict(), file_path)

def load_previous_model(model):
	f_list = glob.glob(os.path.join('./runtime', 'model') + '-*.ckpt')
	start_epoch = 1
	if len(f_list) >= 1:
		epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
		last_checkpoint = f_list[np.argmax(epoch_list)]
		if os.path.exists(last_checkpoint):
			print('load from {}'.format(last_checkpoint))
			model.load_state_dict(torch.load(last_checkpoint))
			start_epoch = np.max(epoch_list)
	return model, start_epoch

def train(model, start_epoch=0):
	# torchvision数据集的输出范围是[0,1]的PILImage,在此归一化为[-1,1] 的张量
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=IS_DOWNLOAD, transform=transform)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	'''
	data_iter = iter(train_loader)
	imgs, labels = data_iter.next()
	#print(labels)
	print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
	show_img(torchvision.utils.make_grid(imgs))
	'''
	# start to train
	# 使用动量分类来计算loss
	learning_rate = 0.01
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	print('start to train...')
	for epoch in range(start_epoch, N_EPOCH+1):
		run_loss = 0.0
		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			inputs, labels = Variable(inputs), Variable(labels)
			# 清空梯度
			optimizer.zero_grad()
			outputs = model(inputs)
			# compute loss
			loss = loss_fn(outputs, labels)
			# 后向传播
			loss.backward()
			# 更新参数
			optimizer.step()

			run_loss += loss.data[0]
			if i % 1000 == 0:	# 每1000次输出一次
				print('[%d, %5d] loss: %.3f%%' %  (epoch, i+1, run_loss/1000))
				run_loss = 0.0
				save_model(model, epoch)
	print('train done!')

def evaluate(model):
	# torchvision数据集的输出范围是[0,1]的PILImage,在此归一化为[-1,1] 的张量
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=IS_DOWNLOAD, transform=transform)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
	data_iter = iter(test_loader)
	images, labels = data_iter.next()

	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

	outputs = model(Variable(images))
	_, predicted = torch.max(outputs.data, 1)  # 获取最大的值
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j][0]] for j in range(4)))
	show_img(torchvision.utils.make_grid(images))

	correct = 0
	total = 0
	print('start to evaluate...')
	for data in test_loader:
		images, labels = data
		outputs = model(Variable(images))
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted==labels).sum()
	print('Accuracy on the 10000 test images: %d %%' % (100.0 * correct/total))	

	# 获取各类别的准确率
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	for data in test_loader:
		images, labels = data
		outputs = model(Variable(images))
		_, predicted = torch.max(outputs.data, 1)
		# 压缩张量
		correct = (predicted==labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += correct[i]
			class_total[label] += 1
	for i in range(10):
		print('Accuracy of %5s is: %d %%' % (classes[i], 100.0 * class_correct[i]/class_total[i]))	

def main():
	model = CNN()
	model, start_epoch = load_previous_model(model)
	train(model, start_epoch)
	evaluate(model)

if __name__ == '__main__':
	main()
