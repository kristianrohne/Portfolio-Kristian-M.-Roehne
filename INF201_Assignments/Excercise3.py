# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:32:09 2023

Excercise 3- Kristian Mathias Røhne and Christian Aron Vea
"""
#Task 1

def reading_list(filename):
    student_list= []
    with open (filename, "r", encoding= "utf-8") as infile: #reading file
        for line in infile:
            name,info= line.rstrip("\n").split(":") #splitting the columns
            age, number= info.split(",")
            student_list.append({"name": name, "age": age, "number": number}) #storing the info in a list, with seperate dictionaries
        print(student_list) #printing the list

reading_list("text_test.txt") #testing the function, put in a file.


#%% Task 2

import re
from pathlib import Path #importing the libraries needed for the task


def path_with_package(): 
    import_pattern=re.compile(r"import\s+(\w*)\b") #making regexes to find the imported packages
    from_pattern = re.compile(r"from\s+\w*\b\s+\'import'\s+(\w*)\b")
    
    current_directory = Path.cwd()
    py_files = current_directory.glob("*.py") #finding alle the python files in the working directory 
    
    for file_path in py_files: 
        with open(file_path, 'r', encoding='utf-8') as py_file: #Reading the python files
            for line in py_file: #going over the file line for line, trying to find the imported packages
                match_import = import_pattern.search(line)
                match_from = from_pattern.search(line)
                if match_import: print(f"{file_path}: [{match_import.group(1)}]") #printing the packages found and the filepath
                elif match_from: print(f"{file_path}: [{match_from.group(1)}]")


path_with_package() #testing the function
