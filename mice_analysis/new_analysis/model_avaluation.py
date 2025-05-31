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
from extra_plotting import *


def select_train_sessions(df):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get unique sessions and shuffle them
    unique_sessions = df['session'].unique()
    np.random.shuffle(unique_sessions)
    
    # Calculate fold sizes
    n_sessions = len(unique_sessions)
    fold_size = n_sessions // 5
    remainder = n_sessions % 5
    
    # Initialize all split columns
    for i in range(5):
        df[f'split_{i}'] = 'train'  # Default all to train
    
    # Create 5 folds
    start = 0
    for fold in range(5):
        # Calculate test session indices for this fold
        end = start + fold_size
        if fold < remainder:  # Distribute remainder sessions across first folds
            end += 1
        
        # Get test sessions for this fold
        test_sessions = unique_sessions[start:end]
        
        # Mark these sessions as test in the current split
        df.loc[df['session'].isin(test_sessions), f'split_{fold}'] = 'test'
        
        # Update start for next fold
        start = end
    
    return df

def avaluation(df_20,df_80):
    n_bins = 20

    # Process df_80
    bins = np.linspace(df_80['V_t'].min(), df_80['V_t'].max(), n_bins)
    df_80['binned_ev'] = pd.cut(df_80['V_t'], bins=bins)
    print(df_80['binned_ev'])
    grouped = df_80.groupby('binned_ev').agg(
        ev_mean=('V_t', 'mean'),
        p_right_mean=('choice_num', 'mean'),
        side = ('side_num','mean')
    ).dropna()
    ev_means = grouped['ev_mean'].values
    p_right_mean = grouped['p_right_mean'].values
    side = grouped['side'].values
    [beta, alpha], _ = curve_fit(probit, [ev_means,side], p_right_mean, p0=[0, 1])

    # Process df_20
    bins = np.linspace(df_20['V_t'].min(), df_20['V_t'].max(), n_bins)
    df_20['binned_ev_20'] = pd.cut(df_20['V_t'], bins=bins)
    print(df_20['binned_ev_20'])

    grouped_20 = df_20.groupby('binned_ev_20').agg(
        ev_mean_20=('V_t', 'mean'),
        p_right_mean_20=('choice_num', 'mean'),
        side_20 = ('side_num','mean')
    ).dropna()
    ev_means_20 = grouped_20['ev_mean_20'].values
    p_right_mean_20 = grouped_20['p_right_mean_20'].values
    side_mean_20 = grouped_20['side_20'].values
    bin_sizes = df_20['binned_ev_20'].value_counts(sort=False)
    bin_sizes = bin_sizes.reindex(grouped_20.index)
    weights = bin_sizes / np.sum(bin_sizes)  # Normalize weights to sum to 1
    predicted_p_right = probit([ev_means_20,side_mean_20], beta,alpha)
    print(ev_means_20)
    print(p_right_mean_20)
    print(40*'--')
    print(len(bin_sizes))
    print(len(weights))
    print(len(p_right_mean))
    print(len(predicted_p_right))
    wmse = np.sum(weights * (p_right_mean_20 - predicted_p_right) ** 2)    
    print(wmse)
    return wmse
  