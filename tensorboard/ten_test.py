from torch.utils.tensorboard import  SummaryWriter
from PIL import Image
import numpy as np
writer=SummaryWriter("logs")
imagepath="data/val/ants/94999827_36895faade.jpg"
image=Image.open(imagepath)

image_array=np.array(image)
print(image_array.shape)
writer.add_image("train",image_array,2,dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=3x",i,3*i)
writer.close()