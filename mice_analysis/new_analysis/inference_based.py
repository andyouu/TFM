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
#from extra_plotting import *


def compute_values(df) -> Tuple[pd.DataFrame, str]:
    """
    Summary:
    This function computes the rate R_t at each step and the value V_t given a side a nd 

    Args:
        df ([Dataframe]): [dataframe with experimental data]
        n ([int]): [number of trials back considered]

    Returns:
        new_df([Dataframe]): [dataframe with processed data restrcted to the regression]
        regressors_string([string]) :  [regressioon formula]
    """
    column_counts = df['session'].value_counts()
    print(column_counts)


def inference_plot(prob_switch,prob_rwd,df):
    mice_counter = 0
    f, axes = plt.subplots(1, len(df['subject'].unique()), figsize=(15, 5), sharey=True)
    # iterate over mice
    for mice in df['subject'].unique():
        if mice != 'A10':
            df_mice = df.loc[df['subject'] == mice]
            session_counts = df_mice['session'].value_counts()
            mask = df_mice['session'].isin(session_counts[session_counts > 50].index)
            df_mice['sign_session'] = 0
            df_mice.loc[mask, 'sign_session'] = 1
            new_df_mice = df_mice[df_mice['sign_session'] == 1]
            #print(new_df_mice)
            print(new_df_mice['subject'])
            print(mice)
            compute_values(new_df_mice)
            mice_counter += 1
    plt.show

if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials_maybe_updated.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    #df = df[df['task'] != 'S4']
    df = df[df['subject'] != 'manual']
    prob_rwd = 0.9
    prob_switch = 0.3
    new_df = df[['subject','session', 'outcome', 'side', 'iti_duration']]
    new_df = new_df.copy()
    new_df['outcome_bool'] = np.where(new_df['outcome'] == "correct", 1, 0)
    #A column indicating significant columns will be constructed
    inference_plot(prob_switch,prob_rwd,new_df)