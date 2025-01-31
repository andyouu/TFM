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
from inference_based import *


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
    print("DataFrame with 0.8 of sessions:")
    print(df_80)
    print("\nDataFrame with 0.2 of sessions:")
    print(df_20)
    return df_80,df_20

def avaluation(df_20,df_80):
    n_bins = 20
    df_80['binned_ev'] = pd.qcut(df_80['V_t'], n_bins,duplicates='drop')
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
    print(bin_sizes)
    weights = bin_sizes / np.sum(bin_sizes)  # Normalize weights to sum to 1
    predicted_p_right = probit(ev_means_20, beta,alpha)
    wmse = np.sum(weights * (p_right_mean_20 - predicted_p_right) ** 2)    
    print(wmse)
    return wmse

def inference_plot(prob_switch,prob_rwd,df):
    mice_counter = 0
    errors = np.zeros(len(df['subject'].unique()))
    for mice in df['subject'].unique():
        if mice != 'A10':
            df_mice = df.loc[df['subject'] == mice]
            session_counts = df_mice['session'].value_counts()
            mask = df_mice['session'].isin(session_counts[session_counts > 50].index)
            df_mice['sign_session'] = 0
            df_mice.loc[mask, 'sign_session'] = 1
            new_df_mice = df_mice[df_mice['sign_session'] == 1]
            df_values = compute_values(new_df_mice,  prob_switch, prob_rwd)
            df_80, df_20 = select_train_sessions(df_values)
            df_values = compute_values(df_80,  prob_switch, prob_rwd)
            mice_counter += 1
            errors[mice_counter]  = avaluation(df_20,df_80)
    plt.plot(range(0, len(df['subject'].unique())), errors, marker='o')
    plt.xticks(range(0, len(df['subject'].unique())), df['subject'].unique())
    plt.xlabel('Mice')
    plt.ylabel('Error')
    plt.title('Weighed error of the inference-based model')
    plt.grid(True)
    plt.show()   
    
    

if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials_maybe_updated.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    #df = df[df['task'] != 'S4']
    df = df[df['subject'] != 'manual']
    prob_rwd = 0.99
    prob_switch = 0.5
    new_df = df[['subject','session', 'outcome', 'side', 'iti_duration','probability_r']]
    new_df = new_df.copy()
    new_df['outcome_bool'] = np.where(new_df['outcome'] == "correct", 1, 0)

    inference_plot(prob_switch,prob_rwd,new_df)