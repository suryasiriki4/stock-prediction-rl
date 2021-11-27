import pandas as pd
import numpy as np

produce_charts = False
if produce_charts:
    import matplotlib.pyplot as plt

def return_features(df):
    """
    Params:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    df["return"] = df["close"] / df["close"].shift(1)
    df["close_to_open"] = df["close"] / df["open"]
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]
    df = df.iloc[1:] # first row doesn't have the return value
    return df

def target_value(df):
    """
    Params:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    df["y"] = df["return"].shift(-1)
    df = df.iloc[:len(df)-1]
    return df

