# utils/dataset_utils.py
from PIL import Image
import os

def create_lr_images(hr_dir,lr_dir,scale_factor): #generate LR images
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
    for img_name in os.listdir(hr_dir):
        hr_image_path=os.path.join(hr_dir,img_name)
        lr_image_path=os.path.join(lr_dir,img_name)
        with Image.open(hr_image_path) as hr_img:
            width, height=hr_img.size
            lr_img=hr_img.resize((width//scale_factor,height//scale_factor),Image.BICUBIC)
            lr_img.save(lr_image_path)
