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
        y = np.exp(-beta * x + alpha)
        return 1/(1 + y)


def psychometric(x):
    y = np.exp(-x)
    return 1/(1 + y)

def psychometric_fit(ax,data_vec):
    n_bins = 10
    phi= 1
    for df_glm_mice in data_vec:
        df_80, df_20 = df_glm_mice
        bins = np.linspace(df_80['V_t'].min(), df_80['V_t'].max(), n_bins)
        df_80['binned_ev'] = pd.cut(df_80['V_t'], bins=bins)
        histogram = 1
        if histogram:
            bin_counts = df_80['binned_ev'].value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            bin_counts.plot(kind='bar', width=0.8, color='skyblue', edgecolor='black')
            plt.title('Histogram of Elements in Each Bin', fontsize=16)
            plt.xlabel('Bin Interval', fontsize=14)
            plt.ylabel('Number of Elements', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        #plt.show()
        grouped = df_80.groupby('binned_ev').agg(
        ev_mean=('V_t', 'mean'),
        p_right_mean=('choice_num', 'mean')
        ).dropna() 
        ev_means = grouped['ev_mean'].values
        p_right_mean = grouped['p_right_mean'].values
        [beta, alpha],_ = curve_fit(probit, ev_means, p_right_mean, p0=[0, 1])
        df_20['binned_ev_20'] = pd.qcut(df_20['V_t'], n_bins,duplicates='drop')
        grouped_20 = df_20.groupby('binned_ev_20').agg(
        ev_mean_20 =('V_t', 'mean'),
        p_right_mean_20=('choice_num', 'mean')
        ).dropna() 
        ev_means_20 = grouped_20['ev_mean_20'].values
        p_right_mean_20 = grouped_20['p_right_mean_20'].values
        bin_sizes = df_20['binned_ev_20'].value_counts(sort=False)
        #print(ev_means)
        #print(p_right_mean)
        [beta, alpha],_ = curve_fit(probit, ev_means, p_right_mean, p0=[0, 1])
        print(beta)
        print(alpha)
        ax.plot(ev_means_20, probit(ev_means_20, beta,alpha), color='green', label = 'Model', alpha = phi)
        #ax.plot(ev_means, psychometric(ev_means), color='grey', alpha = 0.5)
        ax.plot(ev_means_20, p_right_mean_20, marker = 'o', color = 'black',label = 'Data', alpha = phi)
        phi -= 0.5
        print(40*'__')



def psychometric_plot(ax,df_glm_mice):
    n_bins = 20
    bins = np.linspace(df_glm_mice['V_t'].min(), df_glm_mice['V_t'].max(), n_bins)
    df_glm_mice['binned_ev'] = pd.cut(df_glm_mice['V_t'], bins=bins)

    # Print the bin intervals themselves
    #for bin_interval in pd.cut(df_glm_mice['V_t'], bins=bins).cat.categories:
    #    print(bin_interval)
    #    filtered_df = df_glm_mice[df_glm_mice['binned_ev'] == bin_interval]
    #
        # Print choice_num and V_t for the current bin
    #    print("choice_num:")
    #    print(filtered_df['choice_num'])
    #    print("V_t:")
    #    print(filtered_df['V_t'])
    #    print(40*'=')

    #print(40*'_')    

    grouped = df_glm_mice.groupby('binned_ev').agg(
    ev_mean=('V_t', 'mean'),
    p_right_mean=('choice_num', 'mean')
    ).dropna() 
    ev_means = grouped['ev_mean'].values
    p_right_mean = grouped['p_right_mean'].values
    #print(ev_means)
    print(p_right_mean)
    ax.plot(ev_means,psychometric(ev_means), color = 'grey')
    ax.plot(ev_means, p_right_mean, marker = 'o', color = 'black')



def psychometric_data(ax,df_glm_mice, GLM_df,regressors_string):
    #we will first compute the evidence:
    regressors_vect = regressors_string.split(' + ')
    coefficients = GLM_df['coefficient']
    n = len(df_glm_mice['r_plus_1'])
    df_glm_mice['evidence'] = 0
    for j in range(len(regressors_vect)):
        df_glm_mice['V_t']+= coefficients[regressors_vect[j]]*df_glm_mice[regressors_vect[j]]
    #psychometric_fit(ax,df_glm_mice)
    psychometric_plot(ax,df_glm_mice)
    

