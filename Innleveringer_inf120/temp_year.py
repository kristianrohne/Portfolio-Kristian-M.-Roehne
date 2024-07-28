# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 23:02:57 2022
__author__ = Kristian Mathias Røhne
__email__ = kristian.mathias.rohne@nmbu.no
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("C:/Users/Kristian Røhne/OneDrive/Inf120/Inf120/filer/meteodata_aas_2012.csv", skiprows=1, header=0, sep=";")

T_avg = df.T_avg.to_numpy()

plt.plot(T_avg)
plt.legend(["Daily average measurment"])
plt.xlabel("day of year")
plt.ylabel("Temp [deg C]")
plt.title("Temperature readings for 2012")

def year_temp(day):
    return Tavg + A * np.sin(omega*(day + offset))


omega= (2*np.pi)/365
Tavg=5.5
A=11
offset=250

Days = np.linspace(1, 366,366)
temp_days = year_temp(Days)

plt.plot(temp_days)
plt.legend(["Daily average measurment", "TO + A * sin(omega*(day+offset))"])


"""
- Gjennomsnittstemperaturen i 2012 lå på omtrent 5.5 grader.

- Temperatursvingenen (A) var på omtrent 11. 

"""



