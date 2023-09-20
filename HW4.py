from typing import Tuple
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(1337)
torch.manual_seed(314)
random.seed(10)

def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False) #  1, # <- start with random weights (this will get adjusted as the model learns)
                                                #dtype=torch.float), # <- PyTorch loves float32 by default
                                   #requires_grad=True) # <- can we update this value with gradient descent?) (0, dtype=torch.float), requires_grad=False)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b) 
    

if __name__ == "__main__":
    # Get available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device = }")

    # Check PyTorch version
    print(f"Using {torch.__version__ = }")

    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)
    # Shuffle the data while keeping corresponding elements in the same order
    data = list(zip(x, y))  # Combine x and y into a list of tuples
    random.shuffle(data)  # Shuffle the list randomly
    x, y = zip(*data)  # Unzip the shuffled data back into separate x and y arrays


    x, y_true, y = torch.tensor(x), torch.tensor(y_true), torch.tensor(y)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")


    # Create train/test split
    # 80% of data used for training set, 20% for testing 
    train_split = int(0.8 * len(x)) 
    X_train, y_train = x[:train_split], y[:train_split]
    X_test, y_test = x[train_split:], y[train_split:]

    print(len(X_train), len(y_train), len(X_test), len(y_test))


    # Create an instance of the model (this is a subclass of 
    # nn.Module that contains nn.Parameter(s))
    model_0 = LinearRegressionModel()

    # Check the nn.Parameter(s) within the nn.Module 
    # subclass we created
    print(f"{list(model_0.parameters()) = }")
    # List named parameters 
    print(f"{model_0.state_dict() = }")


    # Create the loss function
    loss_fn = nn.L1Loss() # L1Loss loss is same as MAE

    # Create the optimizer
    # ``parameters`` of target model to optimize
    # ``learning rate`` (how much the optimizer should change parameters 
    # at each step, higher=more (less stable), lower=less (might take a long time))
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    # Lists to store training and test losses at each epoch
    train_losses = []
    test_losses = []

    # Set the number of epochs (how many times 
    # the model will pass over the training data)
    epochs = 100

    for epoch in range(epochs):
        ### Training

        # Put model in training mode (this is the default state of a model)
        model_0.train()

        # 1. Forward pass on train data using the forward() method inside 
        y_pred = model_0(X_train)

        # 2. Calculate the loss (how different are our models predictions 
        # to the ground truth)
        loss = loss_fn(y_pred, y_train)

        # 3. Zero grad of the optimizer
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Progress the optimizer
        optimizer.step()

        ### Testing

        # Put the model in evaluation mode
        model_0.eval()

        with torch.inference_mode():
            # 1. Forward pass on test data
            test_pred = model_0(X_test)

            # 2. Caculate loss on test data
            test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

            # Print out what's happening
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
            
            # Append the training and test losses to their respective lists
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
    # Print the model weights and compare with original ones
    print("Model Weights:")
    print(f"Weights: {model_0.weights.item()} | Bias: {model_0.bias.item()}")
    print("Original Weights:")
    print("Weights: 2.0 | Bias: 3.5")

    # Plot the loss curve and test loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.legend()
    plt.savefig("loss_curve.png")

    # Plot the results (predicted line along with dataset data)
    model_0.eval()
    with torch.no_grad():
        y_pred = model_0(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o', label='Data')
    plt.plot(x, y_pred, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig("results.png")
    plt.show()
