#非线性激活
#掌握常用激活函数sigmoid，RELU等，
#关注官网，获取激活函数数学公式
import torch
import torch.nn as nn
import torchvision
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root="conv2d_data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64)

class nonLinear(nn.Module):
     def __init__(self):
         super(nonLinear, self).__init__()
         #relu inplace =false,即去除掉为-1的数据,使其为0
         self.relu1=ReLU()
         self.sigmoid1=Sigmoid()

     def forward(self,input):
         # output=self.relu1(input)
         output = self.sigmoid1(input)
         return output

nl=nonLinear()
step=1
writer=SummaryWriter("nonlinear_logs")
for data in dataloader:
    imgs,target=data
    writer.add_images("imgs:",imgs,step)
    writer.add_images("nonlinear:",nl(imgs),step)
    step+=1
writer.close()





#test1
# test_data=torch.tensor([[1,-2,0],
#                         [-0.2,0.3,-3]])
#
# input=torch.reshape(test_data,(-1,1,2,3))
# print(input)
#
# nl=nonLinear()
# print(nl(input))
