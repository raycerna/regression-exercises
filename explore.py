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

def plot_scatter(x,y):
    f, (ax1) = plt.subplots(1, 1)

    ax1.scatter(x, y)
    ax1.plot([0, 6000000], [0, 6000000], '--k')
    ax1.set_ylabel('Target predicted')
    ax1.set_xlabel('True Target')
    ax1.set_title('Target vs Predicted')
    ax1.text(1, 4000000, r'$R^2$=%.2f, MAE=%.2f' % (r2_score(x, y), median_absolute_error(x, y)))
    ax1.set_xlim([0, 6000000])
    ax1.set_ylim([0, 6000000])
    plt.show()

def plot_scatter_log(x,y):
    f, (ax1) = plt.subplots(1, 1)

    ax1.scatter(x, y)
    ax1.plot([0, 40], [0, 40], '--k')
    ax1.set_ylabel('log2 (Target predicted)')
    ax1.set_xlabel('log2 (True Target)')
    ax1.set_title('Target vs Predicted')
    ax1.text(1, 30, r'$R^2$=%.2f, MAE=%.2f' % (r2_score(x, y), median_absolute_error(x, y)))
    ax1.set_xlim([0, 40])
    ax1.set_ylim([0, 40])
    plt.show()