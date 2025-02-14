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



def number_fails_plot(df, n):
    num_rewrd = 0
    num_fail = 0
    count = np.zeros((n,2))
    i = 0
    print(len(df))
    while i < len(df)-2:
        while(i < len(df)-2)&(df.loc[i, 'choice'] == df.loc[i+1,'choice'])&(df.loc[i+1,'outcome'] == 'correct'):
            if(num_rewrd < n): num_rewrd += 1 
            if(df.loc[i+1,'session'] != df.loc[i,'session']): 
                num_rewrd = 0
                num_fail = 0
            i += 1
        
        while(i < len(df)-2)&(df.loc[i, 'choice'] == df.loc[i+1,'choice'])&(df.loc[i+1,'outcome'] == 'incorrect'):
            num_fail += 1
            if(df.loc[i+1,'session'] != df.loc[i,'session']): 
                num_rewrd = 0
                num_fail = 0
            i += 1
        if num_rewrd > 0 and num_rewrd <= n:
            count[num_rewrd - 1][0] = (count[num_rewrd - 1][0] * count[num_rewrd - 1][1] + num_fail) / (count[num_rewrd - 1][1] + 1)
            count[num_rewrd - 1][1] += 1
        num_rewrd = 0
        num_fail = 0
        i += 1 
    plt.plot(range(1, n + 1), count[:, 0], marker='o')
    plt.xlabel('Number of Consecutive Rewards')
    plt.ylabel('Average Number of Failures')
    plt.title('Average Failures After Consecutive Rewards')
    plt.grid(True)
    plt.show()           



def prob_switch(df, n):
    count = np.zeros((n, 3))  # Initialize counters

    i = 0
    while i < len(df) - 1:
        j = 0
        # Count consecutive incorrect outcomes on the same side
        correct_case = 1
        while (i + j < len(df)) and (df.loc[i + j, 'outcome'] == 'incorrect') and correct_case:
            if j > 0 and df.loc[i + j, 'choice'] != df.loc[i + j - 1, 'choice']:
                correct_case = 0 # Exit if choices are not on the same side
                break  
            j += 1

        if j > 0 and correct_case:
            if j < n:
                count[j][0] += 1  # Total cases with j consecutive incorrect choices on the same side
                if (i + j < len(df)) and (df.loc[i + j, 'choice'] != df.loc[i + j - 1, 'choice']):
                    count[j][1] += 1  # Count switches after the sequence

        i += max(j, 1)

    # Calculate probabilities and standard errors
    print(count)
    probabilities = []
    std_errors = []
    for k in range(n):
        if count[k][0] > 0:
            p = count[k][1] / count[k][0]
            probabilities.append(p)
            std_errors.append(np.sqrt(p * (1 - p) / count[k][0]))
        else:
            probabilities.append(np.nan)
            std_errors.append(np.nan)

    # Plot 1: Probability of switching with error bars
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.errorbar(range(0, n), probabilities, yerr=std_errors, fmt='o-', capsize=5)
    plt.xlabel('Number of Consecutive Failures on the Same Side')
    plt.ylabel('Probability of Switching')
    plt.title('Probability of Switching Given Number of Errors on the Same Side')
    plt.grid(True)

    # Plot 2: Histogram of cases for each number of errors
    plt.subplot(1, 2, 2)
    plt.bar(range(0, n), count[:, 0], color='skyblue')
    plt.xlabel('Number of Consecutive Failures on the Same Side')
    plt.ylabel('Number of Cases')
    plt.title('Histogram of Cases for Each Number of Errors on the Same Side')
    plt.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials_maybe_updated.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    #df = df[df['task'] != 'S4']
    df = df[df['subject'] != 'manual']
    new_df = df[['subject','session', 'outcome', 'side']]
    new_df = new_df.copy()
    new_df['outcome_bool'] = np.where(new_df['outcome'] == "correct", 1, 0)
        #A column indicating significant columns will be constructed
    session_counts = new_df['session'].value_counts()
    mask = new_df['session'].isin(session_counts[session_counts > 50].index)
    new_df['sign_session'] = 0
    new_df.loc[mask, 'sign_session'] = 1
    new_df = new_df[new_df['sign_session'] == 1]
    #A column with the choice of the mice will now be constructed
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'right'), 'choice'] = 'left'
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'left'), 'choice'] = 'left'
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'left'), 'choice'] = 'right'
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'right'), 'choice'] = 'right'
    new_df['choice'].fillna('other', inplace=True)
    n_max_rewards = 10
    #number_fails_plot(new_df,n_max_rewards)
    prob_switch(new_df.dropna(subset=['outcome']).reset_index(drop=True),n_max_rewards)