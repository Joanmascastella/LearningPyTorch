import torch 
import numpy as np

#backpropagation calculating loss 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is being used")
    x = torch.tensor(1.0)
    y = torch.tensor(2.0)

    w = torch.tensor(1.0, requires_grad=True)

    #forward pass to compute the loss
    y_hat = w*x
    loss = (y_hat - y)**2
    print(loss)

    #backward pass
    loss.backward()
    print(w.grad)


else:
    print("CUDA is not available")


