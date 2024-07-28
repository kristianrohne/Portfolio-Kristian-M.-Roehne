# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:29:19 2022
__author__ = Kristian Mathias Røhne
__email__ = kristian.mathias.rohne@nmbu.no
"""

nameDB= [
     ['Tore', 'Hansen'],
     ['Silje', 'Olavsen'],
     ['Aase', 'Lund'],
     ['Jens Petter', 'Oremo'],
     ['Tina', 'Kittelsen'],
     ['Dag', 'Paulsen'],
     ['Lena', 'Nilsen'],
     ['Karsten', 'Woll'],
     ['Ine', 'Ørstad'],
     ['Ravn', 'Havnås'],
     ['Jesper', 'Danberg']]



def name_check(first,family):
    if first[0]== "T" or len(family)>6 \
        or [first,family]==["Ravn","Havnås"]:
        return True,
    else:
        return False



for fullname in nameDB:
    if name_check(fullname[0],fullname[-1]):
        print((nameDB.index(fullname)+1),fullname[0],fullname[1])

          
    
        
   
    
            

    
    
     
 
        
    
    