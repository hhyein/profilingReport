import pandas as pd
import numpy as np
from scipy import stats

def lower_upper(df, column):
    df = df.loc[:, [column]]
    df = df.dropna()

    q25 = np.quantile(df[column], 0.25)
    q75 = np.quantile(df[column], 0.75)

    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    return lower, upper

def quality_issue(df):
    missing = df.isnull().sum().tolist()
    extreme = []
    for column in df:
        lower, upper = lower_upper(df, column)
        data1 = df[df[column] > upper]
        data2 = df[df[column] < lower]

        extreme.append(data1.shape[0] + data2.shape[0])

    total = [x + y for x, y in zip(missing, extreme)]
    return missing, extreme, total

def quality_metric(df, column):
    df = df.iloc[:, column].dropna()

    kstest = round(stats.kstest(df, 'norm').pvalue, 5)
    skewness = round(stats.skew(df, nan_policy = 'omit'), 5)
    kurtosis = round(stats.kurtosis(df, nan_policy = 'omit'), 5)

    return kstest, skewness, kurtosis

def quality_metric_total(df):
    column_list = list(df)
    output = [[], [], [], []]

    for i in range(0, len(column_list)):
        kstest, skewness, kurtosis = quality_metric(df, i)

        output[0].append(kstest)
        output[1].append(skewness)
        output[2].append(kurtosis)

    return output