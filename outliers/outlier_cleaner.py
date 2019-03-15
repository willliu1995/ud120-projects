#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    import numpy as np
    import pandas as pd

    # 传入的numpy array是array of lists，直接使用报错
    # 使用hstack方法转换为普通array
    # https://stackoverflow.com/a/42499122
    predictions = np.hstack(predictions)
    ages = np.hstack(ages)
    net_worths = np.hstack(net_worths)
    residual = abs(np.subtract(predictions, net_worths))

    # 使用上述array构建主数据集df1
    df1 = pd.DataFrame({"0": ages, "1": net_worths, "2":residual})
    # 选取按照残差正序排列的头90%的行，行为整数
    df2 = df1.sort_values(by="2").iloc[:int(len(df1)*0.9)]

    # 将dataframe转换为要求的元组
    for x in df2[["0", "1", "2"]].values:
        cleaned_data.append(tuple(x))

    return cleaned_data