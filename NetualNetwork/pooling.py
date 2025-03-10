#池化层 入门 pooling
#cell_mode参数，即对结果向上cell取整 在取值时数量不足也取得当前的最大值，false与之相反，或者向下floor取整
#最大池化核，即取池化核矩阵圈住原矩阵中的最大值，最大池化层作用保留原始特征同时降低数据维度
#默认池化步长和kernel核矩阵的大小一致，从左到右，从上到下
import torch
import torch.nn as nn
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
#ctrl + 左键 查看CIFAR10数据，然后点击structure查看该类的源码，找到他的返回值，以此熟悉代码，编写接受函数
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("conv2d_data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64)

class maxPol(nn.Module):
    def __init__(self):
        super(maxPol, self).__init__()
        self.maxpol=MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output=self.maxpol(input)
        return output

#简单数据测试
#注意此处的tensor和Totesnsor，有所不同，在测试的时候要注意
#注意到此处只是一个数组，所以与图片的3通道是不同的，一般设置1
# mp=maxPol()
# testdata=torch.tensor([[1,2,3,4]
#                       ,[3,0,1,2]
#                       ,[1,0,0,3]
#                        ,[0,3,2,1]])
# #变换序列是元组不是列表
#reshape 妙用 -1即代表由系统自己计算
# input=torch.reshape(testdata,(-1,1,4,4))
# print(testdata)
# print(mp(input))

#图片测试
writer=SummaryWriter("MaxPool2d_logs")
mp=maxPol()
step=1
for data in dataloader:
    imgs,target=data
    writer.add_images("input_imgs:",imgs,step)
    writer.add_images("output_imgs:",mp(imgs),step)
    step+=1
writer.close()

