import torch.nn.functional as F
import torch

input=([1,2,3,4],
       [0,2,1,3],
       [1,0,0,2],
       [1,0,2,0])
kernel=([1,0,2],
        [3,1,1],
        [2,3,1])
input=torch.tensor(input)
kernel=torch.tensor(kernel)

#要求输入是3维度或者是4维度的，所以需要改变状态
input=torch.reshape(input,[1,1,4,4])
kernel=torch.reshape(kernel,[1,1,3,3])
#strid 每次卷积移动的步进
#padding 相当于在卷积选取时，对输入矩阵进行了一个长宽扩充
#dialation 空洞卷积，即kernel对应时数据间空格大小
output=F.conv2d(input,kernel,stride=1,padding=1)
print(output)