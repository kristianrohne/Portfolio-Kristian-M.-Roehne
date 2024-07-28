# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:20:16 2022
__author__ = "Kristian Mathias Røhne"
__email__ = "kristian.mathias.rohne@nmbu.no"
"""
print("---------------------------")


list1 = [
    [2015, 86343, 123],
    [2016, 93512, 125],
    [2017, 83935, 119],
    [2018, 91274, 128],
    [2019, 88935, 127],
    [2020, 95182, 132],
    ]


list2=[]
for a in list1:
    g=a[1]/a[2]
    list2.append(g)

for c,b in zip(list1,list2):
   print(f" {c[0]}: {b:.2f} kr/bøssebærer, med til sammen {c[2]} bøssebærere.")
   
   
print("---------------------------")


max_value=max(list2)
max_index=list2.index(max_value)


print(f"I {list1[max_index][0]} ble det samlet inn mest \
penger pr. bøssebærer, som var {list2[max_index]:.2f}kr/bøssebærer.")

#I 2016 ble det samlet inn mest penger pr. bøssebærer, som var 748.10kr/bøssebærer

print("---------------------------")
    


