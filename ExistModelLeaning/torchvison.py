#pytorch官网与图片有关的模型
#一个简单的图片分类模型VGG13常用
#！！！！现有模型使用修改！！！
import torchvision.datasets
from torch.utils.data import DataLoader
from torch import nn
dataset=torchvision.datasets.CIFAR10(root="conv2d_data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=1)

#相当于一些默认参数
vgg16_false=torchvision.models.vgg16(pretrained=False)
#ture 即是训练好的参数
vgg16_True=torchvision.models.vgg16(pretrained=True)
# print("ok")


#利用现有网络Vgg16改动变成自己的网络结构
#因为cr10数据分为了10类，所以我们需要改动vgg16最后的线性输出为10
# vgg16_True.add_module("add_Linear",nn.Linear(1000,10))
#
# #classifierh中添加线性层
vgg16_True.classifier.add_module("classifier_add_module",nn.Linear(1000,10))

#直接修改原来的
# vgg16_True.classifier[6]=nn.Linear(1000,10)

print(vgg16_True)