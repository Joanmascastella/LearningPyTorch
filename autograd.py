import torch
import numpy as np

# Initialize tensors without checking for CUDA
x = torch.rand(3, requires_grad=True)
print(x)
y = x + 2
print(y)
z = y * y * 2
print(z)
# z = z.mean()
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)  # Calculating the backward can only be done with scalar values or in other words
                                                         # values that have calculations

z.backward(v)  # Calculate the gradient of z with respect to x
# print(x.grad)

# Remove the grad requirement
x.requires_grad_(False)
print(x)

# Another way is using detach that will create a new tensor from the same values of the tensor x but without the grad requirement
y = x.detach()
print(y)

# This will wrap it and again prevent the grad function
with torch.no_grad():
    y = x + 2
    print(y)

# Example of a training loop where grad is used 
# Here we create tensor of 4 which requires grad
weights = torch.ones(4, requires_grad=True)

# This is the training loop
# Value in range is the number of iterations I want
for epoch in range(5):
    # Simple computation to show the issues that can come when using grad
    model_output = (weights * 3).sum()
    # This applies the grad backward function which will add the values from each iteration on to itself
    model_output.backward()
    print(weights.grad)
    # To prevent that we call this below
    # This is usually done with PyTorch's inbuilt optimizers
    weights.grad.zero_()