import torch
import numpy as np

# Making an empty tensor
x = torch.empty(3, 3, 3)
print(x)

# Making a 2D tensor of random values
y = torch.rand(2,2)
print(y)

# Making a 2D tensor of zeros
y = torch.zeros(2,2)
print(y)

# Making a 2D tensor of ones
y = torch.ones(2,2)
print(y)

# Making a 2D tensor and determining the datatype
y = torch.ones(2,2)
print(y.dtype)

# Making a 2D tensor and making the datatype integer. This can be done with all datatypes such as double floats
y = torch.ones(2,2, dtype=torch.int)
print(y.dtype)

# Making a 2D tensor and determining the size
y = torch.ones(2,2, dtype=torch.int)
print(y.size())

# Making a tensor from data
y = torch.tensor([2.5, 0.1])
print(y)

# Mathematical operations with different tensors 
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x + y
z = torch.add(x, y)
z = torch.sub(x, y)
z = torch.mul(x, y)
z = torch.div(x, y)
print(z)

# Appending all elements of one tensor to another
x = torch.rand(2,2)
y = torch.rand(2,2)
z = y.add_(x)
print(z)

# Slicing tensors
x = torch.rand(5,2)
print(x)
# This will only print out the first column
print(x[:, 0])
# This will only print out the first row
print(x[1, :])
# This will only print out the first row and the item
print(x[1, 1])

# Reshaping tensor to display one dimension 
x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y)

# Converting from torch tensor to numpy
x = torch.ones(5)
print(x)
b = x.numpy()
print(b)

# Converting from numpy to tensor
a = np.ones(5) 
b = torch.from_numpy(a)
print(b)

# Check if CUDA is available and move tensors to the GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)
else:
    print("CUDA is not available")
