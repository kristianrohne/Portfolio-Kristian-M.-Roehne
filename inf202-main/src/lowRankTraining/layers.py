import torch
import torch.nn as nn
from src.lowRankTraining.activationfunctions import CustomActivation


class VanillaLowRankLayer(nn.Module):
    """
    VanillaLowRankLayer class, represents a low-rank layer in a neural network.

    Parameters:
    - inputSize (int): The size of the input features.
    - outputSize (int): The size of the output features.
    - activation (str): The name of the activation function to be applied.
    - rank (int): The rank of the low-rank factorization.

    Attributes:
    - _U (nn.Parameter): Learnable parameter representing the input weight matrix.
    - _S (nn.Parameter): Learnable parameter representing the diagonal weight matrix.
    - _V (nn.Parameter): Learnable parameter representing the output weight matrix.
    - _b (nn.Parameter): Learnable parameter representing the bias.
    - _activation (str): Name of the activation function.

    Methods:
    - orthogonalize_weights(): Orthogonalizes the weight matrices _U and _V.
    - forward(X): Performs the forward pass through the layer.
    - step(lr, s): Updates the layer's parameters based on gradients and learning rate.

    Note: This class needs CustomActivation class for applying activation functions.
    """
    def __init__(self, inputSize, outputSize, activation, rank):
        """
        Initializes the VanillaLowRankLayer instance.

        Parameters:
        - inputSize (int): The size of the input features.
        - outputSize (int): The size of the output features.
        - activation (str): The name of the activation function to be applied.
        - rank (int): The rank of the low-rank factorization.
        """
        super(VanillaLowRankLayer, self).__init__()
        # defining parameters for vanilla low-rank, with chosen dimensions. 
        self._U = nn.Parameter(torch.randn(inputSize, rank), requires_grad=True)
        self._S = nn.Parameter(torch.randn(rank, rank), requires_grad=True)
        self._V = nn.Parameter(torch.randn(outputSize, rank), requires_grad=True)
        self._b = nn.Parameter(torch.randn(outputSize), requires_grad=True)
        self._activation = activation

        # For maintaining stability during training
        self.orthogonalize_weights()

    def orthogonalize_weights(self):
        """
        Orthogonalizes the weight matrices _U and _V using QR decomposition.
        """
        U_orth, _ = torch.linalg.qr(self._U, 'reduced')
        self._U.data = U_orth

        V_orth, _ = torch.linalg.qr(self._V, 'reduced')
        self._V.data = V_orth.T
    
    def forward(self, X):
        """
        Performs forward pass through the layer.

        Parameters:
        - X (torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Output of the layer after applying activation.
        """
        # matrix multiplication
        tmp1 = X @ self._U
        tmp2 = tmp1 @ self._S
        tmp3 = tmp2 @ self._V

        # returning the output
        return CustomActivation(self._activation)(tmp3 + self._b)   
    
    def step(self, lr, s):
        """
        Updates the layer's parameters based on gradients and learning rate.

        Parameters:
        - lr (float): Learning rate for parameter updates.
        - s (bool): Not in use here. Used for activating last bit of DynamiacalLowRank's step function.
        """

        # recipie for updating parameters in VanillaLowRankLayer
        dfU = self._U.grad
        self._U.data = self._U - lr*dfU  # Updates _U

        dfS = self._S.grad
        self._S.data = self._S - lr*dfS  # updates _S
        
        dfV = self._V.grad
        self._V.data = self._V - lr*dfV  # updates _V
       
        dfb = self._b.grad
        self._b.data = self._b - lr*dfb  # updates _b
        
        # sets gradients to zero
        self._U.grad.zero_()
        self._S.grad.zero_()
        self._V.grad.zero_()
        self._b.grad.zero_()


class DenseLayer(nn.Module):
    """
    DenseLayer class represents a dense (fully connected) layer in a neural network.

    Parameters:
    - inputSize (int): The size of the input features.
    - outputSize (int): The size of the output features.
    - activation (str): The name of the activation function to be applied.

    Attributes:
    - _W (nn.Parameter): Learnable parameter representing the weight matrix.
    - _b (nn.Parameter): Learnable parameter representing the bias.
    - _activation (str): Name of the activation function.

    Methods:
    - forward(X): Performs the forward pass through the layer.
    - step(lr, s): Updates the layer's parameters based on gradients and learning rate.

    Note: This class needs CustomActivation class for applying activation functions.
    """
    def __init__(self, inputSize, outputSize, activation):
        """
        Initializes the DenseLayer instance.

        Parameters:
        - inputSize (int): The size of the input features.
        - outputSize (int): The size of the output features.
        - activation (str): The name of the activation function to be applied.
        """
        super(DenseLayer, self).__init__()
        # defining weight and bias, with chosen dimensions
        self._W = nn.Parameter(torch.randn(inputSize, outputSize), requires_grad=True)
        self._b = nn.Parameter(torch.randn(outputSize), requires_grad=True)
        self._activation = activation

    def forward(self, X):
        """
        Performs forward pass through the layer.

        Parameters:
        - X (torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Output of the layer after applying activation.
        """
        # output
        return CustomActivation(self._activation)(X @ self._W + self._b)  
    
    def step(self, lr, s):
        """
        Updates the layer's parameters based on gradients and learning rate.

        Parameters:
        - lr (float): Learning rate for parameter updates.
        - s (bool): Not in use here. Used for activating last bit of DynamiacalLowRank's step function.
        """
        dfW = self._W.grad
        self._W.data = self._W - lr*dfW  # Updates _W

        dfb = self._b.grad
        self._b.data = self._b - lr*dfb  # Updates _b
        # sets gradients to zero
        self._W.grad.zero_()
        self._b.grad.zero_()   


class LowRankLayer(nn.Module):
    """
    LowRankLayer class represents a low-rank layer in a neural network.

    Parameters:
    - inputSize (int): The size of the input features.
    - outputSize (int): The size of the output features.
    - activation (str): The name of the activation function to be applied.
    - rank (int): The rank of the low-rank factorization.

    Attributes:
    - _U (nn.Parameter): Learnable parameter representing the input weight matrix.
    - _S (nn.Parameter): Learnable parameter representing the diagonal weight matrix.
    - _V (nn.Parameter): Learnable parameter representing the output weight matrix.
    - _b (nn.Parameter): Learnable parameter representing the bias.
    - _activation (str): Name of the activation function.

    Methods:
    - forward(X): Performs the forward pass through the layer.
    - step(lr, s): Updates the layer's parameters based on gradients and learning rate.

    Note: This class needs CustomActivation class for applying activation functions.
    """
    def __init__(self, inputSize, outputSize, activation, rank):
        """
        Initializes the LowRankLayer instance.

        Parameters:
        - inputSize (int): The size of the input features.
        - outputSize (int): The size of the output features.
        - activation (str): The name of the activation function to be applied.
        - rank (int): The rank of the low-rank factorization.
        """
        super(LowRankLayer, self).__init__()    
        # defining parameters for low-rank, with choosen dimensions. 
        self._U = nn.Parameter(torch.randn(inputSize, rank), requires_grad=True)
        self._S = nn.Parameter(torch.randn(rank, rank), requires_grad=True)
        self._V = nn.Parameter(torch.randn(outputSize, rank), requires_grad=True)
        self._b = nn.Parameter(torch.randn(outputSize), requires_grad=True)
        self._activation = activation

    def forward(self, X):
        """
        Performs forward pass through the layer.

        Parameters:
        - X (torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Output of the layer after applying activation.
        """
        # Matrix multiplication
        tmp1 = X @ self._U
        tmp2 = tmp1 @ self._S
        tmp3 = tmp2 @ self._V.T

        return CustomActivation(self._activation)(tmp3 + self._b)  # output
    
    def step(self, lr, s):
        """
        Updates the layer's parameters based on gradients and learning rate.

        Parameters:
        - lr (float): Learning rate for parameter updates.
        - s (bool): Activates last bit of step function if s == True
        """
        K_old = self._U @ self._S
        L_old = self._V @ self._S.T
        
        dfU = self._U.grad
        dfV = self._V.grad

        K = K_old - lr*(dfU @ self._S)
        U, _ = torch.linalg.qr(K, "reduced")
        M = U.T @ self._U

        L = L_old - lr*(dfV @ self._S.T)
        V, _ = torch. linalg.qr(L, "reduced")
        N = V.T @ self._V
    
        self._S.data = M @ self._S @ N.T
        # updates paramteres 
        self._b.data = self._b - lr*self._b.grad
        self._U.data = U
        self._V.data = V
        
        # sets gradients to zero 
        self._U.grad.zero_()
        self._V.grad.zero_()
        self._b.grad.zero_()
        
        if s:
            dfS = self._S.grad
            self._S.data = self._S - lr * dfS  # finally updates _S
            self._S.grad.zero_()
