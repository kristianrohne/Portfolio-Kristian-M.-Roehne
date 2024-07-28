from src.lowRankTraining.activationfunctions import CustomActivation
import torch
import pytest


class TestCustomActivation:
    """
    Test class for CustomActivation class.

    Fixtures:
    - custom_activation_relu: Fixture for CustomActivation with 'relu' activation.
    - custom_activation_tanh: Fixture for CustomActivation with 'tanh' activation.
    - custom_activation_linear: Fixture for CustomActivation with 'linear' activation.
    - custom_activation_sigmoid: Fixture for CustomActivation with 'sigmoid' activation.

    Test Methods:
    - test_custom_activation_relu: Tests the 'relu' activation function of CustomActivation.
    - test_custom_activation_tanh: Tests the 'tanh' activation function of CustomActivation.
    - test_custom_activation_linear: Tests the 'linear' activation function of CustomActivation.
    - test_custom_activation_sigmoid: Tests the 'sigmoid' activation function of CustomActivation.
    - test_custom_activation_unknown: Tests the case of using an unknown activation function.
    """

    @pytest.fixture
    def custom_activation_relu(self):
        """Fixture for CustomActivation with 'relu' activation."""
        return CustomActivation(activation='relu')

    @pytest.fixture
    def custom_activation_tanh(self):
        """Fixture for CustomActivation with 'tanh' activation."""
        return CustomActivation(activation='tanh')

    @pytest.fixture
    def custom_activation_linear(self):
        """Fixture for CustomActivation with 'linear' activation."""
        return CustomActivation(activation='linear')

    @pytest.fixture
    def custom_activation_sigmoid(self):
        """Fixture for CustomActivation with 'sigmoid' activation."""
        return CustomActivation(activation='sigmoid')

    def test_custom_activation_relu(self, custom_activation_relu):
        """Tests the 'relu' activation function of CustomActivation."""
        input_tensor = torch.randn(5, 5)
        output = custom_activation_relu(input_tensor)
        assert torch.equal(output, torch.relu(input_tensor))

    def test_custom_activation_tanh(self, custom_activation_tanh):
        """Tests the 'tanh' activation function of CustomActivation."""
        input_tensor = torch.randn(5, 5)
        output = custom_activation_tanh(input_tensor)
        assert torch.equal(output, torch.tanh(input_tensor))

    def test_custom_activation_linear(self, custom_activation_linear):
        """Tests the 'linear' activation function of CustomActivation."""
        input_tensor = torch.randn(5, 5)
        output = custom_activation_linear(input_tensor)
        assert torch.equal(output, input_tensor)

    def test_custom_activation_sigmoid(self, custom_activation_sigmoid):
        """Tests the 'sigmoid' activation function of CustomActivation."""
        input_tensor = torch.randn(5, 5)
        output = custom_activation_sigmoid(input_tensor)
        assert torch.equal(output, torch.sigmoid(input_tensor))

    def test_custom_activation_unknown(self):
        """
        Tests the case of using an unknown activation function.

        Raises:
        - ValueError: If an unknown activation function is used.
        """
        with pytest.raises(ValueError, match="Unknown activation function"):
            custom_activation_unknown = CustomActivation(activation='unknown')
            input_tensor = torch.randn(5, 5)
            _ = custom_activation_unknown(input_tensor)

