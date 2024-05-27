import torch 
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is being used")
    # doing it manually with numpy
    X = np.array([1, 2, 3, 4], dtype=np.float32)
    Y= np.array([2, 4, 6, 8], dtype=np.float32)

    w = 0.0

    #calculate our model prediction 
    def forward(x):
        return w*x

    # calculate loss
    def loss(y, y_predicted):
        return ((y_predicted -y)**2).mean()

    # calculate gradient
    def gradient(x,y, y_predicted):
        return np.dot(2*x, y_predicted - y).mean()
    
    #print prediction before training 
    print(f'prediction before training: f(5) = {forward(5):.3f}')

    #training
    learning_rate = 0.01
    nr_iterations = 30

    for epoch in range(nr_iterations):
        #prediction = forward pass
        y_pred = forward(X)

        #loss
        l = loss(Y, y_pred)

        #gradients
        dw=gradient(X,Y,y_pred)

        #update our weights
        w -= learning_rate * dw

        #print
        if epoch % 2 == 0:
            print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        
        #print prediction after training 
        print(f'prediction after training: f(5) = {forward(5):.3f}')
else:
    print("CUDA is not available")


