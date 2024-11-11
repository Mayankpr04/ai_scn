# generate_lr_images.py
##########################################################
#How to run: python3 generate_lr_images.py --dataset_dir /path/to/dataset --scale_factor 4
##########################################################
import argparse
import os
from utils.dataset_utils import create_lr_images

# Set up argument parsing for dataset directory and scale factor
parser=argparse.ArgumentParser(description="Generate Low-Resolution Images")
parser.add_argument("--dataset_dir",type=str,required=True,help="Path to the dataset directory containing high-resolution images")
parser.add_argument("--scale_factor",type=int,default=2,help="Downscaling factor (e.g., 2, 3, 4)")
args=parser.parse_args()
# Define the high-resolution directory and create the low-resolution directory automatically
hr_dir=args.dataset_dir
lr_dir=f"{args.dataset_dir}_X{args.scale_factor}"
create_lr_images(hr_dir,lr_dir,args.scale_factor)
print(f"Low-resolution images saved to {lr_dir}")