# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:36:04 2022
__author__ = "Kristian Mathias Røhne"
__email__ = "kristian.mathias.rohne@nmbu.no"
"""

def extract_data(filename):
    with open(filename,"r") as infile:
        infile.readline()
        numberlist=[]
        for line in infile:
            words=line.split()
            numbers=(float(words[1]),float(words[2]))
            numberlist.append(numbers)
    
    return numberlist

data= extract_data("C:/Users/Kristian Røhne/OneDrive/Inf120/Inf120/filer/Oxygen.txt")

isototal=[]
for i in data:
    MtimesN=i[0]*i[1]
    isototal.append(MtimesN)
    
print(f"Oksygens morale masse er {sum(isototal):.4f} kg/mol.")
    



