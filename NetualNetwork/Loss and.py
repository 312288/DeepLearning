#损失函数和反向传播
#多种L1loss情况，反应预测与目标的差距
import torch
import torch.nn as nn
from torch.nn import L1Loss
inputs=torch.tensor([1,2,3],dtype=torch.float)
target=torch.tensor([1,2,5],dtype=torch.float)

inputs=torch.reshape(inputs,(1,1,1,3))
target=torch.reshape(target,(1,1,1,3))

loss=L1Loss(reduction='mean')
result=loss(inputs,target)
print(result)

#平方差
loss_mse=nn.MSELoss()
print(loss_mse(inputs,target))

#分类问题
#交叉熵损失
#根据对预测概率与实际的差距大小判断，越小越准确,即计算结果的具体图片的概率（x[target]）求反与总体的差
output=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(output,(1,3))
loss_cross=nn.CrossEntropyLoss()
print(loss_cross(x,y))
