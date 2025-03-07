import torchvision
#transform使用，改变图片数据格式
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)
# print(test_set[0])
# print(train_set.classes)
# img,target=test_set[0]
# print(test_set.classes[target])
# img.show()
writer=SummaryWriter("p10")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("数据集图片",img,i)
writer.close()
