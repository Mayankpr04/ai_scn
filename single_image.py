import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from models.scale_wise_model import ScaleWiseNetwork as ScaleWiseNetworkOriginal
from models.sc_adaptive import ScaleWiseNetwork as ScaleWiseNetworkAdaptive

# Argument parser
parser = argparse.ArgumentParser(description="Run Super-Resolution Model on a Single Image")
parser.add_argument("--input_image", type=str, required=True, help="Path to the input LR image")
parser.add_argument("--output_image", type=str, default="output_sr.png", help="Path to save the output SR image")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
parser.add_argument("--model_type", type=str, required=True, choices=["original", "adaptive"], help="Type of model to use: 'original' or 'adaptive'")
parser.add_argument("--resize_input", action="store_true", help="Resize input to 256x256 if using the adaptive model")

args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select model based on the argument
if args.model_type == "original":
    model = ScaleWiseNetworkOriginal().to(device)
else:
    model = ScaleWiseNetworkAdaptive().to(device)

# Load the model checkpoint with filtering
checkpoint = torch.load(args.model_path, map_location=device)
model_state_dict = model.state_dict()
filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
model_state_dict.update(filtered_checkpoint)
model.load_state_dict(model_state_dict)
model.eval()  # Set to evaluation mode

# Load and preprocess the input image
input_image = Image.open(args.input_image).convert("RGB")  # Ensure it's in RGB mode
if args.model_type == "adaptive" and args.resize_input:
    input_image = input_image.resize((256, 256))  # Resize only if using adaptive model and resize flag is set
input_tensor = ToTensor()(input_image).unsqueeze(0).to(device)  # Add batch dimension

# Run the model on the input image
with torch.no_grad():
    output_tensor = model(input_tensor)

# Convert the output tensor to an image
output_image = ToPILImage()(output_tensor.squeeze(0).cpu())  # Remove batch dimension and convert to PIL

# Save the output image
output_image.save(args.output_image)
print(f"Model Output: {args.output_image}")

