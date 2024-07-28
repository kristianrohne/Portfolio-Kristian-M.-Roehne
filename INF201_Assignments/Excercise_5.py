# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:53:39 2023

Kristian Mathias Røhne, mail: kristian.mathias.rohne@nmbu.no
Christian Aron Vea, mail: christian.aron.vea@nmbu.no"""

import numpy as np

# Define the dimensions for each layer in the network
n = [64, 128, 128, 128, 10] 

# Initialize the weight matrices (W) and bias vectors (b) for each layer
weights = [np.random.randn(n[i], n[i+1]) for i in range(len(n)-1)]
biases = [np.random.randn(n[i+1]) for i in range(len(n)-1)]

# Input data vector (x)
x = np.random.randn(n[0])  # Random input data for demonstration

# Define the ReLU activation function
def sigma(s):
    return np.maximum(0, s) #return the maximum of 0 or s, so if s<0 it returns 0

y = x.copy()

#making each layer 
for i in range(len(n)-1):
    y = sigma(np.dot(y, weights[i]) + biases[i])

# Print the final output (y) after passing through all layers
print("Final Output (y):", y)
