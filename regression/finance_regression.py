#!/usr/bin/python


"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


sys.path.append(os.getcwd())
sys.path.insert(0, "./tools/")

from tools.feature_format import featureFormat, targetFeatureSplit

def bonus_regression(features_list):
    dictionary = pickle.load(
        open("./final_project/final_project_dataset_modified_unix.pkl", "rb"))
    # list the features you want to look at--first item in the
    # list will be the "target" feature
    data = featureFormat(dictionary, features_list, remove_any_zeroes=True, sort_keys = "./tools/python2_lesson06_keys_unix.pkl")
    target, features = targetFeatureSplit(data)

    # training-testing split needed in regression, just like classification

    feature_train, feature_test, target_train, target_test = train_test_split(
        features, target, test_size=0.5, random_state=42)
    train_color = "b"
    test_color = "r"

    # Your regression goes here!
    # Please name it reg, so that the plotting code below picks it up and
    # plots it correctly. Don't forget to change the test_color above from "b" to
    # "r" to differentiate training points from test points.
    reg_salary = LinearRegression().fit(feature_train, target_train)

    # printing slop and intercept
    slope = reg_salary.coef_[0]
    intercept = reg_salary.intercept_
    score1 = reg_salary.score(feature_train, target_train)
    score2 = reg_salary.score(feature_test, target_test)
    print("slope: {}, \n intercept: {}, \n score_trainingdata: {}, \n score_testdata: {}".format(slope, intercept, score1, score2))

    # draw the scatterplot, with color-coded training and testing points
    for feature, target in zip(feature_test, target_test):
        plt.scatter(feature, target, color=test_color)
    for feature, target in zip(feature_train, target_train):
        plt.scatter(feature, target, color=train_color)
    # labels for the legend
    plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
    plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")
    # draw the regression line, once it's coded
    try:
        plt.plot(feature_test, reg_salary.predict(feature_test))
    except NameError:
        pass

    reg_outlier = LinearRegression().fit(feature_test, target_test)
    plt.plot(feature_train, reg_outlier.predict(feature_train), color="g")

    print(reg_outlier.coef_)

    plt.xlabel(features_list[1])
    plt.ylabel(features_list[0])
    plt.legend()
    plt.show()

# features_list = ["bonus", "salary"]
bonus_regression(["bonus", "salary"])

bonus_regression(["bonus", "long_term_incentive"])
