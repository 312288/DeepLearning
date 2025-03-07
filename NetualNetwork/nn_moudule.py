#torch是PyTorch的核心计算库，提供张量操作和神经网络基础模块。
#torchvision是扩展库，提供计算机视觉相关工具（如数据集加载、图像变换API），需依赖torch运行。
import torch
from torch import nn

class network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return output
nw=network()
x=torch.tensor(1.0)
output=nw(x)
print(output)