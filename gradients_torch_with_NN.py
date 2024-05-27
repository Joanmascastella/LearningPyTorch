import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is being used")
    
    X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32).to(device)
    Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32).to(device)

    x_test = torch.tensor([5], dtype=torch.float32).to(device)
    n_samples, n_features = X.shape

    input_size = n_features
    output_size = n_features

    class LinearRegression(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegression, self).__init__()
            # define layers
            self.lin = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.lin(x)
        
    model = LinearRegression(input_size, output_size).to(device)

    # Print prediction before training 
    print(f'Prediction before training: f(5) = {model(x_test).item():.3f}')

    # Training
    learning_rate = 0.01
    nr_iterations = 100

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(nr_iterations):
        # Prediction = forward pass
        y_pred = model(X)

        # Loss
        l = loss(Y, y_pred)

        # Gradients
        l.backward()

        # Update our weights
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()    

        # Print
        if epoch % 10 == 0:
            [w, b] = model.parameters()
            print(f'Epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')
    
    # Print prediction after training 
    print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')
else:
    print("CUDA is not available")
