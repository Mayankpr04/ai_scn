#evaluate.py - used to evaluate the model on any image dataset using both lr and hr images
#command structure : python3 evaluate.py --model_path --model_type --low_res_image_directory --high_res_image_directory --batch_size --out_dir
# run: 'python3 evaluate.py -h' for help on how to run

import argparse
import os
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
from models.scale_wise_model import ScaleWiseNetwork as OriginalScaleWiseNetwork
from models.sc_adaptive import ScaleWiseNetwork as EnhancedScaleWiseNetwork
from utils.data_loader import CustomDataset
from torch.utils.data import DataLoader
from utils.metrics import psnr, psnr_y  

#Arg Parser - Add any arg parsing commands here to maintain structure
parser = argparse.ArgumentParser(description="Evaluate Super-Resolution Model")
parser.add_argument("--model_path",type=str,required=True, help="Path to the trained model file")
parser.add_argument("--model_type",type=str,choices=["original", "enhanced"],required=True,help="Type of model architecture")
parser.add_argument("--test_lr_dir",type=str,required=True, help="Directory for low-resolution test images")
parser.add_argument("--test_hr_dir",type=str,required=True, help="Directory for high-resolution test images")
parser.add_argument("--batch_size",type=int,default=1, help="Batch size for testing")
parser.add_argument("--output_dir",type=str,default="evaluation_results",help="Directory to save output images")
parser.add_argument("--visualize",action="store_true",help="Save and display output images for visual comparison")
args = parser.parse_args()

#USE CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#There are two models: Original is simpler and trains faster.
if args.model_type == "original":
    model = OriginalScaleWiseNetwork().to(device)
    print("Using original model for evaluation.")
else:
    model = EnhancedScaleWiseNetwork().to(device)
    print("Using enhanced model for evaluation.")

#Loading the pre-trained model weights
checkpoint = torch.load(args.model_path,map_location=device)
model_state_dict = model.state_dict()
filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
model_state_dict.update(filtered_checkpoint)
model.load_state_dict(model_state_dict)
model.eval()

#load the data
test_dataset = CustomDataset(lr_dir=args.test_lr_dir,hr_dir=args.test_hr_dir)
test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)

#Automatically creates an output directory if one does not exist. 
if args.visualize and not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

#Main evaluation function - also displays images: comment out if not needed.
def evaluate(model,loader,device,output_dir="output_images",visualize=False):
    model.eval()
    avg_psnr=0.0
    avg_psnr_y=0.0
    count=0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with torch.no_grad():
        for idx, (lr_img,hr_img) in enumerate(loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr_img = model(lr_img) #output from model
            #metric used by author for evaluation of model
            avg_psnr+=psnr(sr_img,hr_img).item()
            avg_psnr_y+=psnr_y(sr_img,hr_img).item()
            count+=1
            #Convert Tensors to visualisable images
            sr_img_pil = ToPILImage()(sr_img.squeeze(0).cpu())
            hr_img_pil = ToPILImage()(hr_img.squeeze(0).cpu())
            lr_img_pil = ToPILImage()(lr_img.squeeze(0).cpu())
            if visualize:
                sr_img_pil.save(os.path.join(output_dir,f"sr_image_{idx}.png"))
            #Below section will display images
            if idx < 3:  # Displaying first 3 samples, change if necessary.
                fig, axes = plt.subplots(1,3,figsize=(15,5))
                axes[0].imshow(lr_img_pil)
                axes[0].set_title("Low-Resolution (LR)")
                axes[0].axis("off")
                axes[1].imshow(sr_img_pil)
                axes[1].set_title("Model Output")
                axes[1].axis("off")
                axes[2].imshow(hr_img_pil)
                axes[2].set_title("High-Resolution (HR)")
                axes[2].axis("off")
                plt.suptitle(f"Image {idx+1}")
                plt.show()
            print(f"Processed image {idx+1}/{len(loader)}- PSNR: {psnr(sr_img,hr_img).item():.2f} dB")
    #Overall avg psnr for the model on the dataset
    avg_psnr /= count
    avg_psnr_y /= count
    #print statements to print the results of metrics
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average PSNR_Y: {avg_psnr_y:.2f} dB")
    print("Evaluation completed.")
evaluate(model, test_loader, device, output_dir="output_images", visualize=True)

