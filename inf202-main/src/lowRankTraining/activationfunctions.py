import torch
import torch.nn as nn


class CustomActivation(nn.Module):
    """
    CustomActivation class represents a module for applying custom activation functions.

    Parameters:
    - activation (str): The name of the activation function to be applied.

    Methods:
    - forward(X): Applies the specified activation function to the input tensor.

    Note: This class supports activation functions such as 'relu', 'tanh', 'linear', and 'sigmoid'.
    """
    def __init__(self, activation):
        super(CustomActivation, self).__init__()
        self.activation = activation

    def forward(self, X):
        if self.activation == 'relu':
            return torch.relu(X)
        elif self.activation == 'tanh':
            return torch.tanh(X)
        elif self.activation == 'linear':
            return X
        elif self.activation == 'sigmoid':
            return torch.sigmoid(X)
        # if you want to make a new activation function:
        # elif self.activation == "Activation name"
            # return "Put new activation method here!"
        
        # raise error if activation functions isn't called right
        else: 
            raise ValueError("Unknown activation function! Supported activations are relu, tanh and linear") 
