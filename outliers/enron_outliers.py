#!/usr/bin/python
#%%
import os
import pickle
import sys
import matplotlib.pyplot as plt

#%%
sys.path.append(os.getcwd())
sys.path.insert(0, "./tools/")
from tools.feature_format import featureFormat, targetFeatureSplit

#%%
### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("./final_project/final_project_dataset_unix.pkl", "rb") )
features = ["salary", "bonus"]

#%%
data_dict.pop('TOTAL')

#%%
data = featureFormat(data_dict, features)

#%%
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

