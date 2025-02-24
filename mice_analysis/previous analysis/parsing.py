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


def parsing(df,trained):
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    #df = df[df['task'] != 'S4']
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



