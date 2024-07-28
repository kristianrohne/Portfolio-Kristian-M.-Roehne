import toml
from src.lowRankTraining.layers import VanillaLowRankLayer, DenseLayer, LowRankLayer
import os
from src.lowRankTraining.neuralNetwork import NeuralNetwork
import logging


class ConfigLayer:

    """
    ConfigLayer class for handling configuration files and running neural network training.

    Attributes:
    - type (str): Type of the layer.
    - dim (tuple): Dimensions of the layer.
    - activation (str): Activation function of the layer.
    - rank (int): Rank of the layer (default is 0).

    Methods:
    - read_configfile(file_path): Reads the TOML file and extracts layer configurations.
    - run_folder(folderPath, model_folder="models"): Runs training for all configuration files in a folder.
    - run_single_file(filePath, model_folder="models", filePath2=None): Runs training for a single configuration file.
    """

    def __init__(self, type, dim, activation, rank=0):  # elements in a layer
        self.type = type
        self.dim = dim
        self.activation = activation
        self.rank = rank

    @staticmethod
    def read_configfile(file_path):
        """
        Reads the TOML file and extracts layer configurations.

        Parameters:
        - file_path (str): Path to the configuration file.

        Returns:
        - tuple: Tuple containing layer configurations, batch size, learning rate, and number of epochs.
        """

        # reading of the TOML file
        with open(file_path, "r") as configFile:
            userData = toml.load(configFile)

        # gets the values from config file with list comprehension,
        # only if the section is called [layer].
        # Puts this values into the class ConfigLayer
        layer_objects = [ConfigLayer(**layer_data) for layer_data in userData.get('layer', [])]

        layerList = []
        # creating list from config file, according to witch layer type is choosen
        for layer in layer_objects:
            if layer.type == 'vanillaLowRank':
                layerList.append(VanillaLowRankLayer(layer.dim[0], layer.dim[1], layer.activation, layer.rank))
            elif layer.type == 'dense':
                layerList.append(DenseLayer(layer.dim[0], layer.dim[1], layer.activation))
            elif layer.type == 'lowRank':
                layerList.append(LowRankLayer(layer.dim[0], layer.dim[1], layer.activation, layer.rank))

        # settings from config file
        settings = userData.get('settings', {})
        batchSize = settings.get('batchSize')
        learningrate = settings.get('learningRate')
        numEpochs = settings.get('numEpochs')

        # returning all values from config file in a readable way
        return layerList, batchSize, learningrate, numEpochs 
    
    @staticmethod
    def run_folder(folderPath, model_folder="models"):
        """
        Runs training for all configuration files in a folder.

        Parameters:
        - folderPath (str): Path to the folder containing configuration files.
        - model_folder (str): Path to the folder where models will be saved (default is "models").
        """

        # puts all files from the folder into a list
        configFilesList = os.listdir(folderPath)

        # prints the layers choosen by the user
        print(configFilesList)
        for fileName in configFilesList:  # runs trough all files
            # finding the file in the folder
            configfile = os.path.join(folderPath, fileName)
            # uses run_single_file method for each file
            ConfigLayer.run_single_file(filePath= fileName, model_folder= model_folder, filePath2=configfile)
    

    @staticmethod
    def run_single_file(filePath, model_folder="models", filePath2=None):
        """
        Runs training for a single configuration file.

        Parameters:
        - filePath (str): Path to the configuration file.
        - model_folder (str): Path to the folder where models will be saved (default is "models").
        - filePath2 (str, optional): Path to the configuration file (default is None, which uses the same as filePath).
        """

        # This code makes run_single_file work in the run_folder function.
        if filePath2 is None:
            filePath2 = filePath
        # runs the program on the inserted file
        layerList, batchSize, learningrate, numEpochs = ConfigLayer.read_configfile(filePath2)
       
        # Logger, Use the base filename without extension as the unique identifier
        log_file = f"{os.path.splitext(filePath)[0]}_log.log"  
        parameter_file = f"{os.path.splitext(filePath)[0]}_parameters.pth"
        savedParmeter_path = os.path.join(model_folder, parameter_file)
        # creates path to where you can save logfile
        savedLog_path = os.path.join(model_folder, log_file)

        # Makes model_folder if it doesnt exist
        if not os.path.exists(model_folder):
                os.makedirs(model_folder)

        # creates logger
        logger = ConfigLayer.create_logger(savedLog_path= savedLog_path)

        text = (
                f"Starting training on {filePath} \n"
                f"Layerlist: {layerList}, batchSize: {batchSize}, learningrate= {learningrate}, numEpochs: {numEpochs}"
                )
        print(text)
        logger.info(text)  # logging

        # calling the neural network with the layer and logger
        Model = NeuralNetwork(layerList, logger)

        try:    # loads earlier model, if existing, then starts training.
            if os.path.isfile(savedParmeter_path):
                Model.load_model(savedParmeter_path)
            Model.train(batchSize=batchSize, num_epochs=numEpochs, lr=learningrate, savedParmeter_pat=savedParmeter_path)
        except:
            Model.train(batchSize=batchSize, num_epochs=numEpochs, lr=learningrate, savedParmeter_pat=savedParmeter_path)

    @staticmethod
    def create_logger(savedLog_path):
        """
        Creates a logger and returns it.

        Parameters:
        - savedLog_path (str): Path to the log file.

        Returns:
        - logging.Logger: Logger instance.
        """
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(savedLog_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
