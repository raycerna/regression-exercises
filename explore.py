 import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans



def plot_variable_pairs(df):
    """Takes a DataFrame and all of the pairwise relationships along with the regression line for each pair"""
    sns.pairplot(df, kind="reg",plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.9}})

def plot_categorical_and_continous_vars(categorical_var, continuous_var, df):
    """outputs 3 different plots for plotting a categorical variable with a continuous variable"""
    #f, axes = plt.subplots(1, sharey=True, figsize=(6, 4))
    plt.figure()
    plt.figure(figsize=(12,6))
    sns.boxplot(x= categorical_var, y= continuous_var, data=df)
    plt.figure()
    plt.figure(figsize=(12,6))
    sns.swarmplot(x= categorical_var, y= continuous_var, data=df)
    plt.figure()
    plt.figure(figsize=(12,6))
    sns.barplot(x= categorical_var, y= continuous_var, data=df)