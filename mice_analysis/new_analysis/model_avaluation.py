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
    # Step 1: Get unique sessions
    unique_sessions = df['session'].unique()

    # Step 2: Shuffle the sessions
    np.random.shuffle(unique_sessions)

    # Step 3: Split the sessions into 80% and 20%
    split_index = int(len(unique_sessions) * 0.8)
    sessions_80 = unique_sessions[:split_index]
    sessions_20 = unique_sessions[split_index:]

    # Step 4: Partition the DataFrame
    df_80 = df[df['session'].isin(sessions_80)]
    df_20 = df[df['session'].isin(sessions_20)]

    # Print results
    #print("DataFrame with 0.8 of sessions:")
    #print(df_80)
    #print("\nDataFrame with 0.2 of sessions:")
    #print(df_20)
    return df_80,df_20

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
  