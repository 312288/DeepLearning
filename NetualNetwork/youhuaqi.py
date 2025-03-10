#优化器
#pytorch.optim
#利用.backward()，反向传播，grad每次清零，step到下一步
#优化器参数,lr =leaning rate
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
#lr一开始可以大点，后面可以小点
optim=torch.optim.SGD(network.parameters(),lr=0.01)
#多轮次训练，使得参数最好
for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs,target=data
        output=network(imgs)
        #output就是一堆可以看做预测概率，target即取列表中哪一个数据的负数加上后面的交叉熵
        result=loss(output,target)
        #result即误差
        print(result)
        # 梯度下降
        optim.zero_grad()#梯度归零，使得上一次的数据不影响下一次
        result.backward() # 反向传播,才能计算各个节点的参数，然后计算梯度，降低loss
        optim.step()#模型参数调优
        running_loss+=result
    print("runing loss is :",running_loss)

#常用套路
#grad清零
#backward
#step


