import numpy as np
import pandas as pd
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


def parsing(df,trained,opto_yes):
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    #df = df[df['task'] != 'S4']
    if opto_yes:
        new_df = df[df['date'] > "2024-08-30"]
        
    else:
        if trained: 
            new_df = df[df['task'] == 'S4_5']
            new_df = new_df[new_df['date'] > '2024/05/31']
        else:
            new_df = df[df['task'] == 'S4_5']
            new_df = new_df[new_df['date'] < '2024/05/31']
    new_df = new_df[~new_df['subject'].isin(['A10', 'R1', 'R2'])]
    session_counts = new_df['session'].value_counts()
    mask = new_df['session'].isin(session_counts[session_counts > 50].index)
    new_df['sign_session'] = 0
    new_df.loc[mask, 'sign_session'] = 1
    new_df = new_df[new_df['sign_session'] == 1]
    return new_df


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

def performance(df):
    print(df)
    df.loc[(df['probability_r'] > 0.5) &(df['side'] == 'right'), 'performance'] = 1
    df.loc[(df['probability_r'] > 0.5) &(df['side'] == 'left'), 'performance'] = 0
    df.loc[(df['probability_r'] <  0.5) &(df['side'] == 'left'), 'performance'] = 1
    df.loc[(df['probability_r'] < 0.5) &(df['side'] == 'right'), 'performance'] = 0
    df['performance'] = pd.to_numeric(df['performance'], errors='coerce')
    avge_performance= df.groupby('subject')['performance'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(avge_performance['subject'], avge_performance['performance'], marker='o', linestyle='-', color='b')
    plt.title('Average Performance by Subject', fontsize=16)
    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Average Performance', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

def expo_fit(x,tau):
    return np.exp(-x/tau)

def exp_regressors(GLM_df):
    # Extract coefficients for r_plus and r_minus
    r_plus = GLM_df.loc[GLM_df.index.str.startswith('r_plus_'), "coefficient"].values
    r_minus = GLM_df.loc[GLM_df.index.str.startswith('r_minus_'), "coefficient"].values
    
    # Create orders array (indices for fitting)
    orders_plus = np.arange(len(r_plus))  # Orders for r_plus
    orders_minus = np.arange(len(r_minus))  # Orders for r_minus
    
    # Fit exponential function for r_plus
    tau_plus, _ = curve_fit(expo_fit, orders_plus, r_plus, p0=[1])  # Initial guess for tau
    print('tau +:', tau_plus[0])
    
    # Fit exponential function for r_minus
    tau_minus, _ = curve_fit(expo_fit, orders_minus, r_minus, p0=[1])  # Initial guess for tau
    print('tau -:', tau_minus[0])
    return tau_plus[0],tau_minus[0]

