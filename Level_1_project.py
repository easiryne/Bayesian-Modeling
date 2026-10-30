
#Obj_1: Generating the csv file
import os
import pandas as pd
import numpy as np

s1_val = s1_val = list(range(-4, 5, 2))
s2_val = list(range(-10, 11, 1))
s1_var = 0.01
s2_var = s2_var = [0.25, 0.5, 1.0, 2.0, 4.0]
data={"Trial": [], "S1_val": [], "S1_var": [], "S2_val": [], "S2_var": []}
cpt_trial = 1

for s1 in s1_val:
    for s2 in s2_val:
        for var in s2_var:
            for i in range(20):
                trial = (s1, s2, var)
                data["Trial"].append(cpt_trial)
                data["S1_val"].append(s1)
                data["S1_var"].append(s1_var)
                data["S2_val"].append(s2)
                data["S2_var"].append(var)
                cpt_trial += 1

df = pd.DataFrame(data)
df.to_csv("Political_project.csv", index=False)

##########################################
#Obj_2: Analyzing the data by optimizing the parameters

