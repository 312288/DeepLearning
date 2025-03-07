#https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ctrl+p 查看，或者pytorch官网查看过程
#2d矩阵卷积，因为一般对于图像来说是一个二维矩阵
#根据论文利用卷积公式推导参数
#卷积层入门 convolution
import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data=torchvision.datasets.CIFAR10(root="conv2d_data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataset=DataLoader(dataset=data,batch_size=64)


class Conv2D(nn.Module):
    def __init__(self):
        super(Conv2D, self).__init__()

        self.conv2d=Conv2d(in_channels=3,out_channels= 6,kernel_size=3, stride=1,padding=0)
    def forward(self,x):
        x=self.conv2d(x)
        return x

cv2=Conv2D()
step=1
writer=SummaryWriter("conv2d_logs")
for i in dataset:
    imgs,taget=i
    input=cv2(imgs)
    print(imgs.shape)
    print(input.shape)
    #因为add_images不接收6通道图片，所以要变换通道，因为未知所以-1根据图片格式自己推断
    output=torch.reshape(input,(-1,3,30,30))
    print(output.shape)
    writer.add_images("conv2d_output:",output,step)
    writer.add_images("origin_input:",imgs,step)
    step+=1
writer.close()