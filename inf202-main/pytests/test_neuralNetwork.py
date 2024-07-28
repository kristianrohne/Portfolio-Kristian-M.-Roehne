import torch
from src.lowRankTraining.neuralNetwork import NeuralNetwork
from src.lowRankTraining.layers import DenseLayer


class TestNeuralNetwork:
    """
    Test class for the NeuralNetwork class.

    Test Methods:
    - test_neural_network_initialization: Tests if the NeuralNetwork class is initialized correctly.
    - test_neural_network_forward: Tests if the forward method is implemented correctly.
    """

    def make_layerlist(self):
        """
        Helper function to create a sample layer list.

        Returns:
        List: List of DenseLayer instances.
        """
        layer_list = [
            DenseLayer(10, 5, 'relu'),
            DenseLayer(5, 2, 'linear')
        ]
        return layer_list

    def test_neural_network_initialization(self):
        """
        Tests if the NeuralNetwork class is initialized correctly.
        """
        layer_list = self.make_layerlist()
        neural_network = NeuralNetwork(layer_list)
        assert isinstance(neural_network, NeuralNetwork)

    def test_neural_network_forward(self):
        """
        Tests if the forward method is implemented correctly.

        Asserts:
        - The output tensor shape is torch.Size([1, 2]).
        """
        layer_list = self.make_layerlist()
        neural_network = NeuralNetwork(layer_list)
        input_tensor = torch.randn(1, 10)
        output_tensor = neural_network(input_tensor)
        assert output_tensor.shape == torch.Size([1, 2])
