from torch.utils.data import  Dataset
from PIL import Image
import os
class Mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.imageNames=os.listdir(self.path)
    def __getitem__(self, item):
        image_name=self.imageNames[item]
        image_path=os.path.join(self.root_dir,self.label_dir,image_name)
        image=Image.open(image_path)
        label=self.label_dir
        return  image,label

root_dir="data/val"
ants_label_dir="ants"
bees_label_dir="bees"
ants_dataSet=Mydata(root_dir,ants_label_dir)
image,label=ants_dataSet[10]
image.show()