################ How to run ######################
# python train.py \
#     --train_lr_dir "/path/to/train/LR" \
#     --train_hr_dir "/path/to/train/HR" \
#     --test_lr_dir "/path/to/test/LR" \
#     --test_hr_dir "/path/to/test/HR" \
#     --batch_size 2 \
#     --learning_rate 0.001 \
#     --num_epochs 20
# run: 'python3 train.py -h' for help in running the script
# The losses defined under get_loss_function are redundant. I modified the loss function
# due to average performance to be a combination of losses. Initial results in the report use the
# L1 loss function only and has been left untouched.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from models.scale_wise_model import ScaleWiseNetwork 
from models.sc_adaptive import ScaleWiseNetwork as AdaptiveNetwork
from utils.data_loader import CustomDataset 
from utils.metrics import psnr
from utils.metrics import psnr_y
from utils.meters import TimeMeter
from torchvision.transforms import ToPILImage
import argparse

torch.cuda.empty_cache() #to free up some gpu resources
#Arg Parser - Add any arg parsing commands here to maintain structure
parser = argparse.ArgumentParser(description="Super-resolution Training Script")
parser.add_argument("--train_lr_dir",type=str,required=True,help="Path to the training LR images directory")
parser.add_argument("--train_hr_dir",type=str,required=True,help="Path to the training HR images directory")
parser.add_argument("--test_lr_dir",type=str,required=True,help="Path to the test LR images directory")
parser.add_argument("--test_hr_dir",type=str,required=True,help="Path to the test HR images directory")
parser.add_argument("--batch_size",type=int,default=8,help="Batch size for training and evaluation")
parser.add_argument("--learning_rate",type=float, default=0.001,help="Learning rate for optimizer")
parser.add_argument("--num_epochs",type=int,default=20,help="Number of training epochs")
parser.add_argument("--model_type",type=str,choices=["original", "adaptive"],required=True,help="Type of model to use: 'original' or 'enhanced'")
args = parser.parse_args()

train_lr_dir=args.train_lr_dir
train_hr_dir=args.train_hr_dir
test_lr_dir=args.test_lr_dir
test_hr_dir=args.test_hr_dir
batch_size=args.batch_size
learning_rate=args.learning_rate
num_epochs=args.num_epochs
#USE CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Two models: Original trains faster and is simpler
if args.model_type == "original":
    model = ScaleWiseNetwork().to(device)
    print("Using original model for training.")
elif args.model_type == "adaptive":
    model = AdaptiveNetwork().to(device)
    print("Using enhanced model for training.")
#criterion = nn.L1Loss() 
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs) #use of 'dynamic' learning rate
# Data transformation and loading
transform = transforms.ToTensor()
train_dataset = CustomDataset(train_lr_dir, train_hr_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = CustomDataset(test_lr_dir, test_hr_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print("Starting training process...")

# Training function
def train(model, loader, criterion, optimizer, device, epoch, alpha=1, beta=0.5):
    model.train()
    epoch_loss = 0
    print(f"Epoch [{epoch}] - Training...")
    for batch_idx,(lr_imgs, hr_imgs) in enumerate(loader):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        l1_loss = nn.L1Loss()(sr_imgs,hr_imgs)
        smooth_l1_loss = nn.SmoothL1Loss()(sr_imgs,hr_imgs) #use of a combination of losses.
        loss = alpha*l1_loss + beta*smooth_l1_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if batch_idx % 10==0:
            print(f"Batch [{batch_idx + 1}/{len(loader)}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch [{epoch}] - Training completed with Avg Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, loader, criterion, device, epoch, save_images=False, output_dir="evaluation_images"):
    model.eval()
    total_loss=0
    total_psnr=0
    total_psnr_y=0
    to_pil = ToPILImage()  #to convert tensors to images
    time_meter = TimeMeter()
    print(f"Epoch [{epoch}] - Evaluating...")
    # Create output directory if it doesn't exist and if saving images
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for batch_idx,(lr_imgs,hr_imgs) in enumerate(loader):
            time_meter.update()
            lr_imgs,hr_imgs=lr_imgs.to(device),hr_imgs.to(device)
            outputs=model(lr_imgs)
            loss=criterion(outputs,hr_imgs)
            total_loss+=loss.item()
            # Compute PSNR metrics as used by Author
            total_psnr+=psnr(outputs,hr_imgs).item()
            total_psnr_y+=psnr_y(outputs,hr_imgs).item()
            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(loader)}],Validation Loss: {loss.item():.4f}")
            # Save the first few batches images as samples
            if save_images and batch_idx < 5:  #Change to save the entire dataset if needed, this saves only the first 5 batches.
                save_image_comparison(lr_imgs, outputs, hr_imgs, batch_idx, epoch, to_pil, output_dir)
    avg_loss=total_loss/len(loader)
    avg_psnr=total_psnr/len(loader)
    avg_psnr_y=total_psnr_y/len(loader)
    avg_speed=time_meter.avg
    print(f"Epoch [{epoch}] - Validation completed with Avg Loss: {avg_loss:.4f}, "
          f"PSNR: {avg_psnr:.2f} dB, PSNR_Y: {avg_psnr_y:.2f} dB,"
          f"Average speed: {avg_speed:.6f} seconds/sample")
    return avg_loss, avg_psnr,avg_psnr_y,avg_speed

def save_image_comparison(lr_img, sr_img, hr_img, batch_idx, epoch, to_pil, output_dir):
    lr_img=to_pil(lr_img.squeeze(0).cpu())  #convert tensors to PIL images
    sr_img=to_pil(sr_img.squeeze(0).cpu())
    hr_img=to_pil(hr_img.squeeze(0).cpu())
    lr_img.save(os.path.join(output_dir,f"epoch_{epoch}_batch_{batch_idx}_LR.png"))
    sr_img.save(os.path.join(output_dir,f"epoch_{epoch}_batch_{batch_idx}_SR.png"))
    hr_img.save(os.path.join(output_dir,f"epoch_{epoch}_batch_{batch_idx}_HR.png"))
    #To display some images
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Low-Resolution")
    plt.imshow(lr_img)
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.title("Model Output")
    plt.imshow(sr_img)
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.title("High-Resolution")
    plt.imshow(hr_img)
    plt.axis("off")
    plt.suptitle(f"Epoch {epoch} - Batch {batch_idx}")
    plt.show()
    comparison_image_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_comparison.png")
    plt.savefig(comparison_image_path)
    print(f"Comparison image saved to {comparison_image_path}")
    plt.close()

#below loss functions are for future tasks, denoisin and artificat removal are not tested yet. 
def get_loss_function(task):
    if task=='super_resolution':
        return nn.L1Loss()  #L1 loss is commonly used for resolution tasks
    elif task=='denoising':
        return nn.MSELoss()  #MSE is typically used for denoising
    elif task=='artifact_removal':
        return nn.SmoothL1Loss()  #Huber loss can work well for artifact removal
    else:
        raise ValueError("Unknown task type. Choose 'super_resolution', 'denoising', or 'artifact_removal'.")

checkpoint_dir = f"checkpoints/{args.model_type}"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Choose the task and corresponding loss function - future usage
task='super_resolution'  # Change to 'denoising' or 'artifact_removal' as needed, default is super_resolution
criterion=get_loss_function(task)
training_losses=[]
validation_losses=[]
psnr_values=[]
psnr_y_values=[]

for epoch in range(1, num_epochs + 1):
    print(f"Starting Epoch [{epoch}/{num_epochs}]...")
    #Training
    train_loss = train(model,train_loader,criterion,optimizer,device,epoch)
    #Validation
    val_loss, val_psnr, val_psnr_y, avg_speed=evaluate(model, test_loader, criterion, device, epoch)
    scheduler.step()
    training_losses.append(train_loss)
    validation_losses.append(val_loss)
    psnr_values.append(val_psnr)
    psnr_y_values.append(val_psnr_y)
    print(f"Learning Rate: {scheduler.get_last_lr()[0]}")
    # Logging
    print(f"Epoch [{epoch}/{num_epochs}] Summary - "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val PSNR: {val_psnr:.2f} dB, "
          f"Val PSNR_Y: {val_psnr_y:.2f} dB")    
    # Save model checkpoint every 5 epochs - .pth
    if epoch % 5==0:
        checkpoint_path=f"{checkpoint_dir}/scale_wise_model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
plt.plot(training_losses,label='Training Loss')
plt.plot(validation_losses,label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.show()
plt.plot(psnr_values,label='PSNR')
plt.plot(psnr_y_values,label='PSNR_Y')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.title('PSNR and PSNR_Y on Validation Set')
plt.legend()
plt.show()
