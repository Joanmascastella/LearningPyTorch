import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is being used")
    
    X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    # Calculate our model prediction 
    def forward(x):
        return w * x

    # Calculate loss
    def loss(y, y_predicted):
        return ((y_predicted - y) ** 2).mean()
    
    # Print prediction before training 
    print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

    # Training
    learning_rate = 0.01
    nr_iterations = 100

    for epoch in range(nr_iterations):
        # Prediction = forward pass
        y_pred = forward(X)

        # Loss
        l = loss(Y, y_pred)

        # Gradients
        l.backward()

        # Update our weights
        with torch.no_grad():
            w -= learning_rate * w.grad  # Update using gradient, not loss

        # Zero gradients
        w.grad.zero_()    

        # Print
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')
    
    # Print prediction after training 
    print(f'Prediction after training: f(5) = {forward(5).item():.3f}')
else:
    print("CUDA is not available")
