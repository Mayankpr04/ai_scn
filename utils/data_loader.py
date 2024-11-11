# utils/data_loader.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self,lr_dir,hr_dir):
        self.lr_dir=lr_dir
        self.hr_dir=hr_dir
        if not os.path.exists(self.lr_dir):
            raise FileNotFoundError(f"Low-resolution directory '{self.lr_dir}' not found.")
        if not os.path.exists(self.hr_dir):
            raise FileNotFoundError(f"High-resolution directory '{self.hr_dir}' not found.")
        self.lr_images = sorted([
            f for f in os.listdir(self.lr_dir)
            if os.path.isfile(os.path.join(self.lr_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.hr_images = sorted([
            f for f in os.listdir(self.hr_dir)
            if os.path.isfile(os.path.join(self.hr_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        #Check for number images
        print("Number of LR images:",len(self.lr_images))
        print("Number of HR images:",len(self.hr_images))
        print("Sample LR images:",self.lr_images[:5])
        print("Sample HR images:",self.hr_images[:5])
        #while the images are of higher dimensions, scaled to be computationally less expensive - tradeoff with performance
        self.lr_transform=transforms.Compose([
            transforms.Resize((256,256)), 
            transforms.ToTensor(),
        ])
        self.hr_transform=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
        ])    
    def __len__(self):
        return min(len(self.lr_images), len(self.hr_images))
    def __getitem__(self, idx):
        #Loading is indexed based to prevent errors
        lr_image_path=os.path.join(self.lr_dir,self.lr_images[idx])
        hr_image_path=os.path.join(self.hr_dir,self.hr_images[idx])
        lr_image=Image.open(lr_image_path)
        hr_image=Image.open(hr_image_path)
        lr_image=self.lr_transform(lr_image)
        hr_image=self.hr_transform(hr_image)
        return lr_image, hr_image
