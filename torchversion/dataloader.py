import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor())
#numworkers如果不等于0可能出现问题
#drop_last 当数量达不到batch_size时候，直接将最后的删除掉
#shuffle 打乱顺序，dataloader就是一个发牌器的感觉
test_dataloader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
writer=SummaryWriter("DataLoder")
for epoch in range(2):
    step=0
    for data in test_dataloader:
        imgs,target=data
        writer.add_images("epoch:{}".format(epoch),imgs,step)
        step+=1
writer.close()