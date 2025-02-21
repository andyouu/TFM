import pandas as pd
import numpy as np
from typing import Tuple
from datetime import timedelta
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
from scipy.special import erf
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import statsmodels.formula.api as smf
import os
import matplotlib.patches as mpatches


def probit(x, beta, alpha):
        """
        Return probit function with parameters alpha and beta.

        Parameters
        ----------
        x : float
            independent variable.
        beta : float
            sensitivity term. Sensitivity term corresponds to the slope of the psychometric curve.
        alpha : TYPE
            bias term. Bias term corresponds to the shift of the psychometric curve along the x-axis.

        Returns
        -------
        probit : float
            probit value for the given x, beta and alpha.

        """
        probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
        return probit

def psychometric_fit(ax,df_glm_mice):
    #plt.hist(df_glm_mice, bins=30, edgecolor='black')  # 30 bins for the histogram
    #plt.title('Histogram of the df_glm_mice in 100 bins')
    #plt.xlabel('Trial bins')
    #plt.ylabel('log(Odds)')
    #plt.show()
    #print(df_glm_mice)
    n_bins = 5000
    bins = np.linspace(df_glm_mice['evidence'].min(), df_glm_mice['evidence'].max(), n_bins)
    df_glm_mice['binned_ev'] = pd.cut(df_glm_mice['evidence'], bins=bins)
    grouped = df_glm_mice.groupby('binned_ev').agg(
    ev_mean=('evidence', 'mean'),
    p_right_mean=('probability_r', 'mean')
    ).dropna() 
    ev_means = grouped['ev_mean'].values
    p_right_mean = grouped['p_right_mean'].values
    print(ev_means)
    print(p_right_mean)
    [beta, alpha],_ = curve_fit(probit, ev_means, p_right_mean, p0=[0, 1])
    print(beta)
    print(alpha)
    ax.plot(ev_means, probit(ev_means, beta,alpha), marker='o', color='grey')
    #plt.show()

def psychometric(x):
    y = np.exp(-x)
    return 1/(1 + y)

def psychometric_plot(ax,df_glm_mice, data_label):
    n_bins = 10
    #equiespaced bins
    bins = np.linspace(df_glm_mice['evidence'].min(), df_glm_mice['evidence'].max(), n_bins)
    df_glm_mice['binned_ev'] = pd.cut(df_glm_mice['evidence'], bins=bins)
    #equipopulated bins
    #df_glm_mice['binned_ev'] = pd.qcut(df_glm_mice['evidence'], n_bins,duplicates='drop')
    #bin_counts = df_glm_mice['binned_ev'].value_counts().sort_index()
    #print histograms
    histogram = 0
    if histogram:
        bin_counts = df_glm_mice['binned_ev'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        bin_counts.plot(kind='bar', width=0.8, color='skyblue', edgecolor='black')
        plt.title('Histogram of Elements in Each Bin', fontsize=16)
        plt.xlabel('Bin Interval', fontsize=14)
        plt.ylabel('Number of Elements', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    grouped = df_glm_mice.groupby('binned_ev').agg(
    ev_mean=('evidence', 'mean'),
    p_right_mean=(data_label, 'mean')
    ).dropna()
    ev_means = grouped['ev_mean'].values
    p_right_mean = grouped['p_right_mean'].values
    #print(ev_means)
    #print(p_right_mean)
    ax.plot(ev_means,psychometric(ev_means), label = 'GLM Model', color = 'grey')
    ax.plot(ev_means, p_right_mean, marker = 'o', label = 'Data', color = 'black')



def psychometric_data(ax,df_glm_mice, GLM_df,regressors_string,data_label):
    #we will first compute the evidence:
    regressors_vect = regressors_string.split(' + ')
    coefficients = GLM_df['coefficient']
    df_glm_mice['evidence'] = 0
    for j in range(len(regressors_vect)):
        df_glm_mice['evidence']+= coefficients[regressors_vect[j]]*df_glm_mice[regressors_vect[j]]
    #psychometric_fit(ax,df_glm_mice)
    psychometric_plot(ax,df_glm_mice,data_label)
    

