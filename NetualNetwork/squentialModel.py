#搭建网络模型
#squential，提供一个列表简化网络结构代码
import torch.nn as nn
import torch
from torch.nn import Conv2d,MaxPool2d,Sequential,Flatten,Linear
from torch.utils.tensorboard import SummaryWriter


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

data=torch.ones((64,3,32,32))
sq=Squrntial()
print(sq)
output=sq(data)
print(output.shape)

writer=SummaryWriter("sq_logs")
writer.add_graph(sq,data)
writer.close()