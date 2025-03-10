import torch
#方式1，加载模型
import torchvision



#方式1引用导入时，把整个代码引入避免陷阱
from ModelSave import *
vgg161=torch.load("vgg16_method1.pth")
print(vgg161)


#方式2，加载模型，字典存储
# vgg162=torch.load("vgg16_method2.pth")

#还原模型加载方法
vgg16=torchvision.models.vgg16(pretrained=False)
#恢复网络模型结构方法
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)




#方式1存储陷阱
# sq1=torch.load("sq_save.pth")
# print(sq1)

#方式2必须要提供原始结构，因此方式1,2都需要导入原始网络结构
#.state_dict()和load_state_dict()对应
sq=Squrntial()
sq.load_state_dict(torch.load("sq_save2.pth"))
print(sq)



