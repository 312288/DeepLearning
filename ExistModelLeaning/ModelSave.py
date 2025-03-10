#网络模型保存
import torchvision
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

vgg16=torchvision.models.vgg16(pretrained=False)
# #保存方式1 模型结构+模型参数
# torch.save(vgg16,"vgg16_method1.pth")

#保存方式2 模型参数（官方推荐，存储内存更小）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#方式1，陷阱
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

sq=Squrntial()

torch.save(sq,"sq_save.pth")

torch.save(sq.state_dict(),"sq_save2.pth")