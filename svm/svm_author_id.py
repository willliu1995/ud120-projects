#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import os
import sys
from time import time
sys.path.append(os.getcwd())
sys.path.insert(0, "./tools/")
from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.svm import SVC

# training with full data
t0 = time()
clf = SVC(kernel='linear')
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s") #打印训练时间

t0 = time()
pred = clf.predict(features_test)
print("predicting time:", round(time()-t0, 3), "s")

accuracy = clf.score(features_test, labels_test)
print("accuracy: {}".format(accuracy))

##########################################################
# training with 1% of the data
features_train_1pct = features_train[:int(len(features_train)/100)]
labels_train_1pct = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf2 = SVC(kernel='linear')
clf2.fit(features_train_1pct, labels_train_1pct)
print("training time:", round(time()-t0, 3), "s") #打印训练时间

t0 = time()
pred2 = clf2.predict(features_test)
print("predicting time:", round(time()-t0, 3), "s")

accuracy2 = clf2.score(features_test, labels_test)
print("accuracy for training on 1pct data: {}".format(accuracy2))

#################################################
# training with rbf
t0 = time()
clf3 = SVC(kernel='rbf', gamma='auto')
clf3.fit(features_train_1pct, labels_train_1pct)
print("training time:", round(time()-t0, 3), "s") #打印训练时间

t0 = time()
pred3 = clf3.predict(features_test)
print("predicting time:", round(time()-t0, 3), "s")

accuracy3 = clf3.score(features_test, labels_test)
print("accuracy for training on 1pct data with rbf: {}".format(accuracy3))

#########################################################
# How c impacts accuracy, for c in 10, 100, 1000, 1000
for x in [10, 100, 1000, 10000]:
    clf_c = SVC(C=x, kernel='rbf', gamma='auto')
    clf_c.fit(features_train_1pct, labels_train_1pct)
    accuracy_c = clf_c.score(features_test, labels_test)
    print("accuracy for1pct data with rbfon c={}: {}".format(x, accuracy_c))

#########################################################
# retarining on complete datasets
clf_o_c = SVC(C=10000, kernel='rbf', gamma='auto')
clf_o_c.fit(features_train, labels_train)
accuracy_o_c = clf_o_c.score(features_test, labels_test)
print("accuracy for rbf on c=10000: {}".format(accuracy_o_c))

# finding the right answer for specific rows
clf_o_c.predict(features_test[[10]]) #零索引，因此要取哪个数就加哪一行
clf_o_c.predict(features_test[[26]])
clf_o_c.predict(features_test[[50]])

# 使用完整训练的模型进行预测，并计算每个人的邮件数量
pred_o_c = clf_o_c.predict(features_test)

import numpy as np
np.unique(pred_o_c, return_counts=True)