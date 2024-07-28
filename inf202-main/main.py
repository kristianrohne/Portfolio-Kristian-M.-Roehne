from src.lowRankTraining.configLayer import ConfigLayer
import argparse

if __name__ == "__main__":
    """
    Parses command-line arguments to determine whether to run a single configuration file or multiple files in a folder.

    Command-line arguments:
    - -f, --file: If specified, runs the program on the single config file provided as the filepath.
    - -fo, --folder: If specified, runs the program on all config files in the folder provided as the folderpath.
    """

    def parseInput():
        parser = argparse.ArgumentParser(description='Description of your script')
        parser.add_argument('-f', '--file', help='If you want to run only one config file, insert the filepath here')
        parser.add_argument('-fo', '--folder', help='If you want to run multiple files in a folder, insert the folderpath here')
        args = parser.parse_args()

        fileP = args.file
        folderP = args.folder

        try:        # runs the program on all files in inserted folder
            ConfigLayer.run_folder(folderPath=folderP)
        
        except:     # if a file is insertet trough the command line
            ConfigLayer.run_single_file(filePath=fileP)

    parseInput()