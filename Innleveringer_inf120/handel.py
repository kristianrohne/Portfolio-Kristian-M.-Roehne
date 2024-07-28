# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:07:16 2022
__author__ = "Kristian Mathias Røhne"
__email__ = "kristian.mathias.rohne@nmbu.no"
"""
def innlesning():
    handleliste= []
    while True:
        beskrivelse= input("Vare beskrivelse (blank for å avslutte innlesning):")
        print()
        if beskrivelse != "":
            
            while True:
                try: 
                    antall = float(input("Antall:"))
                    print()
                    break
                except:
                    print("Det du skrev kan ikke konverteres til et tall, prøv igjen!")
                    continue
                
            while True:
                try:
                    pris = float(input("Pris:"))
                    print()
                    handleliste.append((beskrivelse, antall,antall*pris))
                    break
                except:
                    print("Det du skrev kan ikke konverteres til et tall, prøv igjen!")
                    continue
        else:
            break
        
    return handleliste

def utskrift():
    handleliste=innlesning()
    print('{:<10s}{:>30s}'.format("Beskrivelse", "Linjekost"))
    
    print("-"*43)
    
    for varer in handleliste:
        print("{:<10s}{:>30.2f}kr".format(varer[0], varer[2]))
         
    print("-"*43)
   
    totalt = sum(tup[2] for tup in handleliste)
    print("{:<10s}{:>30.2f}kr".format("Sum", totalt))