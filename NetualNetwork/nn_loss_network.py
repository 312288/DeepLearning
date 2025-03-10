#在神经网络中损失函数的运用
import torch.nn as nn
import torch
from torch.nn import Conv2d,MaxPool2d,Sequential,Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset=torchvision.datasets.CIFAR10(root="conv2d_data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=1)
class Squrntial(nn.Module):
    def __init__(self):
        super(Squrntial, self).__init__()
        self.model1=Sequential(
            #所有的数据可以根据作者论文中的模型图片和公式，计算
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        x=self.model1(x)
        return x
network=Squrntial()
loss=nn.CrossEntropyLoss()
for data in dataloader:
    imgs,target=data
    output=network(imgs)
    # print(output)
    # print(target)
    #output就是一堆可以看做预测概率，target即取列表中哪一个数据的负数加上后面的交叉熵
    result=loss(output,target)
    #result即误差
    print(result)
    # 反向传播,才能计算各个节点的参数，然后计算梯度，降低loss
    # 梯度下降
    result.backward()
    print("ok")


