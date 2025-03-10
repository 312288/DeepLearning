import torch
from torch import nn
import torchvision
from torch.nn import Sequential
from torch.utils.data import DataLoader

class myNetwork(nn.Module):
    def __init__(self):
        super(myNetwork, self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.model1(x)
        return x

if __name__=="__main__":
    network=myNetwork()
    input=torch.ones(64,3,32,32)
    output=network(input)
    #返回多少行数据，每一行10种类别的概率
    print(output.shape)