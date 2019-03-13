#!/usr/bin/python
import os
import sys
from time import time
from sklearn import tree

sys.path.append(os.getcwd())
sys.path.insert(0, "./tools/")
from tools.email_preprocess import preprocess
from tools import email_preprocess_dt

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
# your code goes here #
t0 = time()  # 初始化计时器
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
accu = clf.score(features_test, labels_test)
print(accu)

print(len(features_test[0]))
print("training time 1:", round(time()-t0, 3), "s")

#########################################################

t0 = time()
(features_train2, features_test2,
 labels_train2, labels_test2) = email_preprocess_dt.preprocess()

clf2 = tree.DecisionTreeClassifier(min_samples_split=40)
clf2.fit(features_train2, labels_train2)
accu2 = clf2.score(features_test2, labels_test2)
print(accu2)

print(len(features_test2[0]))
print("training time 2:", round(time()-t0, 3), "s")
