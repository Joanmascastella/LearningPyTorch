import torch 
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is being used")


else:
    print("CUDA is not available")


