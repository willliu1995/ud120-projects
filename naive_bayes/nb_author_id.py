#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import os
import sys
from time import time
sys.path.append(os.getcwd()) #将整个工作区目录添加为包搜索范围
sys.path.insert(0, "./tools") # 将tool文件夹设为最高搜索优先级
# 上两行代码灵感来自： https://blog.csdn.net/dcrmg/article/details/79546962
from tools.email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
t0 = time() # 初始化计时器

def NBaccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    accuracy = clf.score(features_test, labels_test)
    print("accuracy: {}".format(accuracy))
    return accuracy

NBaccuracy(features_train, labels_train, features_test, labels_test)

print("training time:", round(time()-t0, 3), "s") #打印训练时间
#########################################################


