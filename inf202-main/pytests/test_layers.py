import torch
import pytest
from src.lowRankTraining.layers import VanillaLowRankLayer, DenseLayer, LowRankLayer


class TestLayers:
    """
    Test class for layers in low-rank training.

    Fixtures:
    - settings_with_rank: Fixture providing settings with rank for testing.
    - settings_no_rank: Fixture providing settings without rank for testing.

    Test Methods:
    - test_vanilla_low_rank_layer: Tests the VanillaLowRankLayer class with provided settings.
    - test_dense_layer: Tests the DenseLayer class with provided settings.
    - test_low_rank_layer: Tests the LowRankLayer class with provided settings.
    """

    @pytest.fixture
    def settings_with_rank(self):
        """
        Fixture providing settings with rank for testing.

        Returns:
        Tuple: (input_size, output_size, rank, activation, batch_size)
        """
        input_size = 100
        output_size = 10000
        rank = 49
        activation = "relu"
        batch_size = 64
        return input_size, output_size, rank, activation, batch_size

    @pytest.fixture
    def settings_no_rank(self):
        """
        Fixture providing settings without rank for testing.

        Returns:
        Tuple: (input_size, output_size, activation, batch_size)
        """
        input_size = 64
        output_size = 10
        activation = "relu"
        batch_size = 64
        return input_size, output_size, activation, batch_size

    def test_vanilla_low_rank_layer(self, settings_with_rank):
        """
        Tests the VanillaLowRankLayer class with provided settings.

        Parameters:
        - settings_with_rank (Tuple): Tuple of settings with rank for testing.
        """
        input_size, output_size, rank, activation, batch_size = settings_with_rank
        layer = VanillaLowRankLayer(input_size, output_size, activation, rank)
        x = torch.randn((batch_size, input_size))
        output = layer(x)
        assert output.shape == (batch_size, output_size)

    def test_dense_layer(self, settings_no_rank):
        """
        Tests the DenseLayer class with provided settings.

        Parameters:
        - settings_no_rank (Tuple): Tuple of settings without rank for testing.
        """
        input_size, output_size, activation, batch_size = settings_no_rank
        layer = DenseLayer(input_size, output_size, activation)
        x = torch.randn((batch_size, input_size))
        output = layer(x)
        assert output.shape == (batch_size, output_size)

    def test_low_rank_layer(self, settings_with_rank):
        """
        Tests the LowRankLayer class with provided settings.

        Parameters:
        - settings_with_rank (Tuple): Tuple of settings with rank for testing.
        """
        input_size, output_size, rank, activation, batch_size = settings_with_rank
        layer = LowRankLayer(input_size, output_size, activation, rank)
        x = torch.randn((batch_size, input_size))
        output = layer(x)
        assert output.shape == (batch_size, output_size)

