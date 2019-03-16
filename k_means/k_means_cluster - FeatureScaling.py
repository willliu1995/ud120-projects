#!/usr/bin/python

""" 
    Skeleton code for k-means clustering mini-project.
"""
#%%
import os
import sys
import pickle
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


sys.path.append(os.getcwd())
sys.path.insert(0, "./tools/")
from tools.feature_format import featureFormat, targetFeatureSplit

#%%
def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    # plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii]
                            [1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

#%%
# load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(
    open("./final_project/final_project_dataset_unix.pkl", "rb"))
# there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

#%%
# the input features we want to use
# can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
# feature_3 = "total_payments"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

#%%
scaled_feature = MinMaxScaler().fit_transform(finance_features)
print(scaled_feature)

#%%
# in the "clustering with 3 features" part of the mini-project,
# you'll want to change this line to
# for f1, f2, _ in finance_features:
# (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

#%%
def k_means_clustering(finance_features, num_clusters):
    # cluster here; create predictions of the cluster labels
    # for the data and store them to a list called pred
    clf = KMeans(n_clusters=int(num_clusters)).fit(finance_features)
    pred = clf.predict(finance_features)
    # rename the "name" parameter when you change the number of features
    # so that the figure gets saved to a different file
    try:
        Draw(pred, finance_features, poi, mark_poi=False,
            name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
    except NameError:
        print("no predictions object named pred found, no clusters to plot")

#%%
k_means_clustering(finance_features, 2)

#%%
# k_means_clustering(finance_features, 3)

#%%
df = pd.DataFrame.from_dict(data_dict, orient="index")
df.head()

#%%
df.info()

#%%
def find_enron(series):
    '''
    Find the max and min of a series in the enron dataset.
    Target should be numerical.
    '''
    series = str(series)
    df[series] = df[series].replace("NaN", numpy.nan)
    df[series] = df[series].astype("float")
    max_v = df[series].max()
    min_v = df[series].min()
    print(max_v, min_v)

#%%
# 行使的期权范围
find_enron("exercised_stock_options")

#%%
# 工资范围
find_enron("salary")