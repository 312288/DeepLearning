#完整的模型训练过程
#图片大小不变padding为2
from torch import nn
import torchvision
from torch.nn import Sequential
from torch.utils.data import DataLoader
from model import *
import torch

#获取到CIFAR10数据
train_data=torchvision.datasets.CIFAR10(root="conv2d_data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="conv2d_data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
#将获取到的数据，通过dataloader加载
dataset_train=DataLoader(dataset=train_data,batch_size=64)
dataset_test=DataLoader(dataset=test_data,batch_size=64)

#创建网络模型
network=myNetwork()
#创建损失函数
loss_fn=nn.CrossEntropyLoss()
#创建优化器
lr=1e-2
optimizer=torch.optim.SGD(network.parameters(),lr=lr)

#训练参数
#总训练次数
total_train_step=0
#训练次数
epoch=10
for i in range(1,epoch+1):
    print("---------第 {} 轮训练开始-----------".format(i))
    for data in dataset_train:
        imgs,target=data
        output=network(imgs)
        loss=loss_fn(output,target)

        #优化器设置
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step+=1
        #loss.item()更加正规
        print("训练次数： {}， Loss= {} ".format(total_train_step,loss.item()))