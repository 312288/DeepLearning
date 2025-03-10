#正则化层，线性层，以及一些其他层
#关注pytorch.nn官网，中各类线性层
#属性bias 偏秩 即y=ax+b 线性网络层中的b
#pytorch 自己提供的一些网络模型 文字，图片的，视频的，都有很多，用于分类啊
#使得文件数据更加小
import torch
import torch.nn as nn
import torchvision.datasets
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("conv2d_data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64)
class Linears(nn.Module):
    def __init__(self):
        super(Linears, self).__init__()
        self.linear1=Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return  output
ln=Linears()
for data in dataloader:
     imgs,target=data
     print(imgs.shape)
     input=torch.reshape(imgs,(1,1,1,-1))
     print(input.shape)
     output=ln(input)

     print(output)

     #torch.flatten 使得数据变为一个数列
     print(torch.flatten(output))