import torch 
import numpy as np 
if torch.cuda.is_available():
    device = torch.device("cuda")
#gradients track the computational history. this gives us the ability to map out the progression of our data 
#whenver we want to calculate gradients we need the grad requires to be true 
#we can then calculate the gradients by calling the backward function
#before we want to create the next iteration of our training loop we must call the zero function 
#this will create a computatiopnal graph for gradient calculation
    x = torch.rand(3, requires_grad=True)
    print(x)
    y=x+2
    print(y)
    z=y*y*2
    print(z)
    # z=z.mean()
    print(z)

    v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32) # claculating the backward can only be done with scalar values or in other words
                                                            # values that have calculations

    z.backward(v) # calculate the gradient of z to the gradient of x
    # print(x.grad)


    #removes the grad requirment
    x.requires_grad_(False)
    print(x)

    #another way is using detach that will create a new tensor from the same values of the tensor x but without the grad requirment
    y = x.detach()
    print(y)

    #this will wrap it and again prevent the grad function
    with torch.no_grad():
        y = x + 2
        print(y)



    #example of a training loop where grad is used 
    #here we create tensor of 4 which requires grad
    weights = torch.ones(4, requires_grad=True)

    #this is the training loop
    # value in range is the number of iterations I want
    for epoch in range(5):
        #simple computation to show the issues that can come when using grad
        model_output = (weights*3).sum()
        #this applies the grad backward function whick will add the values from each iteration on to itself
        model_output.backward()
        print(weights.grad)
        #to prevent that we call this below
        #this usually done with pytorchs inbuilt optimizers
        weights.grad.zero_()
else:
    print("CUDA is not available")


