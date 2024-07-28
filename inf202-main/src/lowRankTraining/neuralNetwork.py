import torch
import torch.nn as nn
from src.lowRankTraining.layers import LowRankLayer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Class Inherits from nn.Module, which is a base class for all neural network modules in PyTorch.
class NeuralNetwork(nn.Module):
    """
    NeuralNetwork class.

    Methods:
    - forward(X): Performs forward pass through the neural network.
    - save_model(filepath): Saves the model's state and best accuracy to a specified file.
    - load_model(filepath): Loads the model's state and best accuracy from a specified file.
    - load_MNIST(batchSize): Loads and prepares the MNIST dataset for training and testing.
    - train(batchSize=64, num_epochs=10, lr=0.001, savedParmeter_pat="models\bestmodel.pth"):
      Trains the neural network on the MNIST dataset, with default setting. 
    """
    def __init__(self, layerList, logger=None):
        """
        Initializes the NeuralNetwork instance.

        Parameters:
        - layerList (list): List of PyTorch layers defining the neural network architecture.
        - logger (Logger, optional): Logger object for logging information. Default is None.
        - best_accuracy (float): Best accuracy achieved during training.
        """
        # Calls the constructor of the parent class
        super(NeuralNetwork, self).__init__()  
        self._flatten = nn.Flatten()  # Used later to flatten the input tensor.
        self._layers = torch.nn.Sequential(*layerList)
        self.best_accuracy = 0.0  # Initialize best_accuracy as an attribute
        self.logger = logger  # or self._configure_logger()

    def forward(self, X):
        """
        Performs forward pass through the neural network.

        Parameters:
        - X (torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Output of the neural network.
        """
        Z = self._flatten(X)  # Input stored in Z
        for layer in self._layers:  # Goes over every layer in layerlist
            Z = layer(Z)  # Applies current layer to Z
        return Z  # returns final output
    
    def save_model(self, filepath):
        """
        Saves the model's state and best accuracy to a specified file.

        Parameters:
        - filepath (str): Path to the file where the model should be saved.
        """
        state = {
            'model_state_dict': self.state_dict(),  # Stores state of network and best accuracy
            'best_accuracy': self.best_accuracy
        }                   
        torch.save(state, filepath) #saves state to filepath
        text= f'Model saved to {filepath}'
        print(text)
        if self.logger is not None:  # So you can use Network class without logger
            self.logger.info(text)  # logging
    
    def load_model(self, filepath):
        """
        Loads the model's state and best accuracy from a specified file.

        Parameters:
        - filepath (str): Path to the file from which the model should be loaded.
        """
        state = torch.load(filepath) #load state
        self.load_state_dict(state['model_state_dict']) #Updates the model's state
        self.best_accuracy = state['best_accuracy'] #Updates model's best accuracy
        text= f'Model loaded from {filepath}'
        print(text)
        if self.logger is not None:  # So you can use Network class without logger
            self.logger.info(text)  # logging

    def load_MNIST(batchSize):
        """
        Loads and prepares the MNIST dataset for training and testing.

        Parameters:
        - batchSize (int): Batch size for DataLoader.

        Returns:
        - tuple: Tuple containing criterion, trainloader, testloader, traindataset, and testdataset.
        """
        
        # Defines a sequence of image transformations using PyTorch's transforms.Compose. 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        # Create training dataset
        traindataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

        # Create testing dataset
        testdataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

        # Create DataLoader from training dataset
        trainloader = DataLoader(traindataset, batch_size=batchSize, shuffle=True)

        # Create DataLoader from testing dataset
        testloader = DataLoader(testdataset, batch_size=batchSize, shuffle=False)

        # Define the criterion for training (CrossEntropyLoss)
        criterion = nn.CrossEntropyLoss()

        return criterion, trainloader, testloader, traindataset, testdataset
   
    def train(self, batchSize=64, num_epochs=10, lr=0.001, savedParmeter_pat="models\bestmodel.pth"):
        """
        Trains the neural network on the MNIST dataset.

        Parameters:
        - batchSize (int, optional): Batch size for DataLoader. Default is 64.
        - num_epochs (int, optional): Number of training epochs. Default is 10.
        - lr (float, optional): Learning rate for training. Default is 0.001.
        - savedParmeter_pat (str, optional): Path to save the best model parameters. Default is "models\bestmodel.pth".
        """

        # gets criterion, trainloader and testloader from load_MNIST function from above
        criterion, trainloader, testloader, _, _ = NeuralNetwork.load_MNIST(batchSize) 
        
        # Iterates over the specified number of training epochs
        for epoch in range(num_epochs):
            # Iterates over batches of training data:
            for step, (images, labels) in enumerate(trainloader):
                self.zero_grad()
                out = self(images)      # the forward function is called, with images as parameter X
                loss = criterion(out, labels)
                loss.backward()     # sets backward here and not in each layer class

                # iterates over each layer in layer list
                for layer in self._layers:
                    layer.step(lr, s=False)     # Optimizing parameters of each layer with layer's step functions
                    if isinstance(layer, LowRankLayer):     # goes over step for LowRank again
                        layer.step(lr, s=True)      # s=True activates last bit of step funcion

                # Checks if the current step is a multiple of 100
                if (step + 1) % 100 == 0:  
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(trainloader)}], Loss: {loss.item():.4f}')

            # Evaluation phase
            correct = 0
            total = 0
            with torch.no_grad():
                # Iterates over batches of testing data
                for images, labels in testloader:
                    # Obtains model predictions for the testing data
                    outputs = self(images)
                    # Determines the predicted class labels based on the model's output
                    _, predicted = torch.max(outputs.data, 1)
                    # Accumulates the total number of examples in the testing dataset
                    total += labels.size(0)
                    # Accumulates the number of correctly predicted examples
                    correct += (predicted == labels).sum().item() 

            accuracy = correct / total  # Calculate accuracy
            text = f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {100 * accuracy:.2f}'
            print(text)

            # sets new best accuracy if accuracy > best_accuracy
            if accuracy > self.best_accuracy: 
                text = f'Previous best accuracy: {self.best_accuracy:.4f}, New best accuracy: {accuracy:.4f}'
                print(text)
                # So you can use Network class without logger:
                if self.logger is not None:
                    self.logger.info(text)
                    # sets new best accuracy if accuracy > best_accuracy
                self.best_accuracy = accuracy  
                # saves model with new best accuracy
                self.save_model(savedParmeter_pat) 

        text = f"Training finished for model! Best accuracy {self.best_accuracy} \n"
        print(text)
        # So you can use Network class withou logger.
        if self.logger is not None: 
            self.logger.info(text)

                

