import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing as sklearn_preprocessing
from sklearn.decomposition import PCA
import joblib

import generate_features

def stock_preprocessing():
    filename = "GOOGL.csv"
    df = yahoo_finance_data_to_df(filename)
    # first_30_rows = df.head(30)
    # print(first_30_rows)
    df = generate_features.return_features(df)
    df = generate_features.target_value(df)
    df = clean_df(df)


def yahoo_finance_data_to_df(filename):
    """
    Params:
        filename: string
    Returns:
        pandas.Dataframe
    """

    df = pd.read_csv(filename, header = 0)
    df.drop(labels = "Close", axis = 1, inplace = True)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    return df

def clean_df(df):
    """
    Params:
        df: pandas.Dataframe
    Returns:
        pandas.DataFrame
    """
    df = missing_values(df)
    df = outliers(df)
    return df

def missing_values(df):
    """
    Replace missing values with latest available
    Params:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    missing_values_count = df.isnull().sum()
    print ("Original data #missing values:\n{}".format(missing_values_count))
    if sum(missing_values_count) == 0:
        return df
    else:
        print("Fill of missing values necessary")
        df = df.fillna(method = "ffill", axis = 0).fillna("0")
        missing_values_count = fd.isnull().sum()
        assert sum(missing_values_count) == 0
        return df

def outliers(df):
    """
    Analyze outliers of dataset
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    df_outliers = df.loc[:, ["date", "return", "close_to_open", "close_to_high", "close_to_low"]]
    column_to_analysts = "return"
    df_smallest = df_outliers.sort_values(by=column_to_analysts, ascending=True)
    df_largest = df_outliers.sort_values(by=column_to_analysts, ascending=False)
    print(df_smallest.iloc[:5])
    print(df_largest.iloc[:5])
    return df

stock_preprocessing()