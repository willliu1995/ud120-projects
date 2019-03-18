#!/usr/bin/python
#%%
import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

#%%
np.random.seed(42) # 设定随机数种子
sys.path.append(os.getcwd()) #设置工作区

#%%
# The words (features) and authors (labels), already largely processed.
# These files should have been created from the previous (Lesson 10)
# mini-project.
words_file = "./text_learning/your_word_data.pkl"
authors_file = "./text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))

#%%
# test_size is the percentage of events assigned to the test set (the
# remainder go into training)
# feature matrices changed to dense representations for compatibility with
# classifier functions in versions 0.15.2 and earlier
features_train, features_test, labels_train, labels_test = train_test_split(
    word_data, authors, test_size=0.1, random_state=42)

#%%
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

#%%
# a classic way to overfit is to use a small number
# of data points and a large number of features;
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

#%%
# your code goes here
clf = DecisionTreeClassifier().fit(features_train, labels_train)
accu = clf.score(features_test, labels_test)
print("accuracy for the overfitted model: {}".format(accu))

#%%
# 特征重要性
importance_list = clf.feature_importances_
print(type(importance_list))
print(importance_list[importance_list > 0.2])

#%%
print(importance_list.max())
print(importance_list.argmax())

#%%



