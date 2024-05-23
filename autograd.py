import torch 
import numpy as np 

#this will create a computatiopnal graph for gradient calculation
x = torch.rand(3, requires_grad=True)
print(x)
y=x+2
