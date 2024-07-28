# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:16:55 2022
__author__ = Kristian Mathias Røhne
__email__ = kristian.mathias.rohne@nmbu.no
"""


with open("C:/Users/Kristian Røhne/OneDrive/Inf120/Inf120/filer/norec_corpus.txt", "r", encoding="utf-8") as infile:
    file= infile.readlines()


text = [line.upper() for line in file]


d={}           
for line in text:
    for w in line:
        if w.isalpha(): #can also use  "if w not in exclude_chars:"
          d[w] = d.get(w,0) +1
           
          # Could also use this for counting letters:
          # if w in d:
          #         d[w] += 1
          # else:
          #         d[w] = 1
                 
d["Counted letters"] = sum(d.values())


l = []
for key, value in d.items():
    tuples = (key,value/d["Counted letters"])
    l.append(tuples)   

del l[-1]

l.sort(key=lambda a: a[1], reverse=True)

print("Counted letters:", d["Counted letters"])
for letter, value in l:
    print(letter, f"{value*100:.2f}%")
    
# exclude_chars = [
#     ' ', '\n', ',', '.', '-', '–', '—', '*', '(', ')',
#     '«', '»', ':', ';', '’', '?', "'", '"', '/', '!', '…',
#     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  