import pytest
from src.lowRankTraining.configLayer import ConfigLayer


class TestConfigLayer:
    """
    Test class for ConfigLayer.

    Fixtures:
    - config_file_path: Fixture providing the path to a configuration file.

    Test Methods:
    - test_layerlist: Tests reading and extracting layer information from a configuration file.
    - test_batchSize: Tests reading and extracting batch size from a configuration file.
    - test_learningrate: Tests reading and extracting learning rate from a configuration file.
    - test_numEpochs: Tests reading and extracting the number of epochs from a configuration file.
    """

    @pytest.fixture
    def config_file_path(self):
        """Fixture providing the path to a configuration file."""
        return "testconfigfiles/config1_.toml"

    def test_layerlist(self, config_file_path):
        """
        Tests reading and extracting layer information from a configuration file.

        Parameters:
        - config_file_path (str): Path to the configuration file.

        Asserts:
        - The length of the extracted layer list is 3.
        """
        layerList, _, _, _ = ConfigLayer.read_configfile(config_file_path)
        assert len(layerList) == 3

    def test_batchSize(self, config_file_path):
        """
        Tests reading and extracting batch size from a configuration file.

        Parameters:
        - config_file_path (str): Path to the configuration file.

        Asserts:
        - The extracted batch size is 64.
        """
        _, batchSize, _, _ = ConfigLayer.read_configfile(config_file_path)
        assert batchSize == 64

    def test_learningrate(self, config_file_path):
        """
        Tests reading and extracting learning rate from a configuration file.

        Parameters:
        - config_file_path (str): Path to the configuration file.

        Asserts:
        - The extracted learning rate is 0.001.
        """
        _, _, learningrate, _ = ConfigLayer.read_configfile(config_file_path)
        assert learningrate == 0.001

    def test_numEpochs(self, config_file_path):
        """
        Tests reading and extracting the number of epochs from a configuration file.

        Parameters:
        - config_file_path (str): Path to the configuration file.

        Asserts:
        - The extracted number of epochs is 20.
        """
        _, _, _, numEpochs = ConfigLayer.read_configfile(config_file_path)
        assert numEpochs == 20
        