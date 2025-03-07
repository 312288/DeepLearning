from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer=SummaryWriter("logs")
img=Image.open(".././pictures/hd1.jpg").convert("RGB")#rgba格式不满足3hw ,最多只接收3,4太多了所以需要截断
print(img)

#都图片都需要tensor格式数据
#tensor格式图片
trans_totensor=transforms.ToTensor()
imge_tensor=trans_totensor(img)
print(f"Tensor shape: {imge_tensor.shape}")
writer.add_image("ToTensor",imge_tensor)
print("img_tensor",imge_tensor[0][0][0])

#normalnise 归一化
trans_normalize=transforms.Normalize([0.2,0.1,0.3],[0.1,0.2,0.3])
imge_normalize=trans_normalize(imge_tensor)
print("img_normalize",imge_normalize[0][0][0])
writer.add_image("Normalize",imge_normalize,2)#3号参数是步次


#resize
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize=trans_totensor(img_resize)
writer.add_image("resize",img_resize,0)

#compose 组合管道，将多种操作合并，前面的输出作为后面的输出

trans_resize2=transforms.Resize(800)
trans_compose=transforms.Compose([trans_resize2,trans_totensor])
img_resize2=trans_compose(img)

writer.add_image("resize",img_resize2,2)

#random compose 随机裁剪
trans_randomResize=transforms.RandomCrop([100,150])
for i in range(10):
    trans_ransomcompose=transforms.Compose([trans_randomResize,trans_totensor])
    img_transrandomcompose=trans_ransomcompose(img)
    writer.add_image("随机hw",img_transrandomcompose,i)
writer.close()