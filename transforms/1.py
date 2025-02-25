from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer=SummaryWriter("logs")
img=Image.open(".././pictures/1.png").convert("RGB")#rgba格式不满足3hw ,最多只接收3,4太多了所以需要截断
print(img)


trans_totensor=transforms.ToTensor()
imge_tensor=trans_totensor(img)
print(f"Tensor shape: {imge_tensor.shape}")
writer.add_image("ToTensor",imge_tensor)
writer.close()
