# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:54:03 2022
__author__ = Kristian Mathias Røhne
__email__ = kristian.mathias.rohne@nmbu.no
"""
import random

class Katt:
    
    def __init__(self):
        self.dyre_slag= "katt"
        self.antall_bein= 4
        
    def __str__(self):
        return f"Dyret er en {self.dyre_slag} med {self.antall_bein} bein."
        

class Hund:
    
    def __init__(self):
        self.dyre_slag= "hund"
        self.antall_bein= 4
        
    def __str__(self):
        return f"Dyret er en {self.dyre_slag} med {self.antall_bein} bein."
 
    
class Undulat:
    
    def __init__(self):
        self.dyre_slag= "undulat"
        self.antall_bein= 2
        
    def __str__(self):
        return f"Dyret er en {self.dyre_slag} med {self.antall_bein} bein."
 
    
def lag_familiedyr(antall=2):
    katt= Katt()
    hund= Hund()
    undulat=Undulat()
    
    dyr= [katt, hund, undulat]
    random_liste=[]
    
    for r in range(1, antall+1):
        pets=random.choice(dyr)
        random_liste.append(pets)
    
    return random_liste

List=lag_familiedyr(5)


for i,dyr in enumerate(List):
    print(f"{i+1}: {dyr}")


    
    
