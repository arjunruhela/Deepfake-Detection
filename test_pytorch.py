
import torch
import torch.nn as nn

# Select the available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Create a small neural network
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

model = model.to(device)

# Create sample input
sample_input = torch.randn(1, 10).to(device)

# Get the model's output
output = model(sample_input)

print("Model output:", output)
print("Output shape:", output.shape)
