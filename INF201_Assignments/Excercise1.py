# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:57:23 2023

@author: Kristian Røhne
"""

""
#%% Task 1

formula= input("Please enter your name: ")

print(" ")
print(f"Welcome to INF201 {formula}!")
print(" ")


#%% Task 2

formula= input("Please enter your name: ")

text= f"* Welcome to INF201 {formula}! *"
print(" ")
print("*"*len(text))
print(text)
print("*"*len(text))


#%% Task 3
x= list(range(0,11)) #Making lists for the squares and third powers of the numbers from 0 to 10.
x_squares=[]
x_third=[]

for i in x: x_squares.append(i**2)    
for i in x: x_third.append(i**3)

header=f"{'x':>7}{'x^2':>7}{'x^3':>7}" #Making the header
print("-" *len(header))
print(f"{'x':>7}{'x^2':>7}{'x^3':>7}")
print("-" *len(header))

for i,j,k in zip(x,x_squares,x_third): #Printing out the numbers
    print(f"{i:>7}{j:>7}{k:>7}")   
print("-" *len(header))


#%% Task 4
district_population = {} #Making a dictionary where the key is going to be district, value is population
with open("Downloaded/norway_municipalities_2017.csv", 'r', encoding='utf-8') as infile:
    next(infile)
    for line in infile: 
        municipality, district, population_str = line.split(',') #Had to split by comma on my pc...
        if district in district_population: district_population[district] += int(population_str)
        else: district_population[district] = int(population_str) #Putting the data in the dictionary

def printing(dictionary): 
    for district, population in dictionary.items(): print(f"{district:22} {population}")
    print(" ")
    
sorted_alphabetic= (dict((sorted(district_population.items())))) #Sorting the dictionaries
sorted_population= dict(sorted(district_population.items(), key=lambda x: x[1], reverse=True))
printing(sorted_alphabetic)
printing(sorted_population)


#task 5

import matplotlib.pyplot as plt

plt.bar(sorted_population.keys(),sorted_population.values())

plt.xlabel('Districts')
plt.ylabel('Population')
plt.title('Population in each district')
plt.xticks(rotation=45)

plt.show()
