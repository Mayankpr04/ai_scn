import torch
from torchinfo import summary  # or torchinfo

# Import your model
from models.sc_adaptive import ScaleWiseNetwork  # Replace with actual model path and class name

# Define the model
model = ScaleWiseNetwork()  # Initialize your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the appropriate device

# Count total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Optional: Use torchsummary to get a detailed summary (requires input size)
# Replace (3, 224, 224) with the input shape your model expects
#summary(model, input_size=(3, 256, 256))  # for torchsummary
# or
# from torchinfo import summary
summary(model, input_size=(1,3, 256, 256))  # for torchinfo
