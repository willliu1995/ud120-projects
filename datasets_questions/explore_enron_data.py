#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
#%%
import os
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append(os.getcwd)
sys.path.insert(0, "./final_project")

#%%
enron_data = pickle.load(
    open("./final_project/final_project_dataset_unix.pkl", "rb"))

enron_df = pd.DataFrame.from_dict(enron_data, orient='index')
enron_df = enron_df.replace("NaN", np.nan)
enron_df = enron_df.reset_index()

#%%
# 数据集长度
len(enron_data)

#%%
# 数据集特征
enron_df.info()

#%%
# 有多少嫌疑人
enron_df["poi"].value_counts()


#%%
# 将名字读取进数据集
poi = pd.read_csv("./final_project/poi_names.txt", sep=",")
# 将作为索引的姓氏和是否poi作为列
poi = poi.reset_index()
# 重命名列
poi.columns=["last_name+poi", "first_name"]
poi.head()

#%%
# 拆分姓氏和是否嫌疑人列，将是否poi转换成布尔值，分别储存
poi_temp = poi["last_name+poi"].str.split(expand=True)
poi['poi'] = poi_temp[0].str.contains("(y)")
poi['last_name'] = poi_temp[1]
# 去掉不必要的列
poi = poi.drop(columns=["last_name+poi"])
poi.head()

#%%
poi['poi'].value_counts()

#%%
poi.info()

#%%
enron_df.head()

#%%
# James Prentice的股票
enron_df[enron_df['index'].str.find("PRENTICE JAMES") == 0]['total_stock_value']

#%%
# Wesley Colwell发送给嫌疑人的邮件
enron_df[enron_df['index'].str.find("COLWELL WESLEY") == 0]['from_this_person_to_poi']

#%%
# Jeffrey Skilling的股票期权价值
enron_df[enron_df['index'].str.find("SKILLING JEFFREY") == 0]['exercised_stock_options']

#%%
# CEO, CFO和主席（创始人）分别拿了多少钱
enron_df[enron_df['index'].str.contains(r"SKILLING|FASTOW|LAY") == True][["index", "total_payments"]]

#%%
# POI的薪酬缺失
enron_df[enron_df['poi'] == True].info()