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


def compute_values(df,prob_switch,prob_rwd) -> Tuple[pd.DataFrame, str]:
    """
    Summary:
    This function computes the rate R_t at each step and the value V_t given a side a nd 

    Args:
        df ([Dataframe]): [dataframe with experimental data]
        n ([int]): [number of trials back considered]

    Returns:
        value V_t from the Vertechi paper
    """
    
    df= df.reset_index(drop=True)
    df['V_t'] = np.nan
    df['R_t'] = np.nan

    df.loc[0, 'V_t'] = prob_rwd * (0.7 - 0.3)

    df.loc[0, 'R_t'] = 0

    for i in range(len(df) - 1):
        # Comment the following two lines if we are considering a constant probability of switching
        #if (df.loc[i+1,'side'] == 'right'): prob_switch = df.loc[i+1,'probability_r']
        #if (df.loc[i+1,'side'] == 'left'): prob_switch = 1 - df.loc[i+1,'probability_r']
        if df.loc[i+1, 'outcome_bool'] == 1:
            df.loc[i + 1, 'R_t'] = 0
        if df.loc[i+1, 'outcome_bool'] == 0:
            if(df.loc[i+1, 'side'] == df.loc[i, 'side']):
                df.loc[i + 1, 'R_t'] = (df.loc[i, 'R_t']+prob_switch)/(1-prob_switch)/(1-prob_rwd)
            if(df.loc[i+1, 'side'] != df.loc[i, 'side']):
                df.loc[i + 1, 'R_t'] = (1-prob_switch)/(df.loc[i, 'R_t']+prob_switch)/(1-prob_rwd)

        if df.loc[i + 1, 'side'] == 'right':
            df.loc[i + 1, 'V_t'] = prob_rwd * (1 - 2 / ((1 +1 / df.loc[i + 1, 'R_t']) ))
            
        if df.loc[i + 1, 'side'] == 'left':
            df.loc[i + 1, 'V_t'] = prob_rwd * (1 - 2 / (df.loc[i + 1, 'R_t'] + 1))
    #print(df['V_t'])
    #print(df['R_t'])
    #A column with the choice of the mice will now be constructed
    df.loc[(df['outcome_bool'] == 0) & (df['side'] == 'right'), 'choice'] = 'left'
    df.loc[(df['outcome_bool'] == 1) & (df['side'] == 'left'), 'choice'] = 'left'
    df.loc[(df['outcome_bool'] == 0) & (df['side'] == 'left'), 'choice'] = 'right'
    df.loc[(df['outcome_bool'] == 1) & (df['side'] == 'right'), 'choice'] = 'right'
    df['choice'].fillna('other', inplace=True)

    #create a column where the side matches the regression notaion:
    df.loc[df['choice'] == 'right', 'choice_num'] = 1
    df.loc[df['choice'] == 'left', 'choice_num'] = 0
    df['choice'] = pd.to_numeric(df['choice'].fillna('other'), errors='coerce')
    return df


def inference_plot(prob_switch,prob_rwd,df):
    mice_counter = 0
    f, axes = plt.subplots(1, len(df['subject'].unique()), figsize=(15, 5), sharey=True)
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
            df_values = compute_values(new_df_mice,  prob_switch, prob_rwd,)
            axes[mice_counter].set_title(mice)
            psychometric_fit(axes[mice_counter],df_values)
            mice_counter += 1
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