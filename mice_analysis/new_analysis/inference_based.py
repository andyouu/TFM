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
import itertools
from model_avaluation import *


def compute_values(df,prob_switch,prob_rwd) -> pd.DataFrame:
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
    df.loc[(df['outcome_bool'] == 0) & (df['side'] == 'right'), 'choice'] = 'left'
    df.loc[(df['outcome_bool'] == 1) & (df['side'] == 'left'), 'choice'] = 'left'
    df.loc[(df['outcome_bool'] == 0) & (df['side'] == 'left'), 'choice'] = 'right'
    df.loc[(df['outcome_bool'] == 1) & (df['side'] == 'right'), 'choice'] = 'right'
    df['choice'].fillna('other', inplace=True)
    for i in range(len(df) - 1):
        # Comment the following two lines if we are considering a constant probability of switching
        #if (df.loc[i+1,'side'] == 'right'): prob_switch = df.loc[i+1,'probability_r']
        #if (df.loc[i+1,'side'] == 'left'): prob_switch = 1 - df.loc[i+1,'probability_r']
        #A column with the choice of the mice will now be constructed
        if df.loc[i+1, 'outcome_bool'] == 1:
            df.loc[i + 1, 'R_t'] = 0
        if df.loc[i+1, 'outcome_bool'] == 0:
            if(df.loc[i+1, 'choice'] == df.loc[i, 'choice']):
                df.loc[i + 1, 'R_t'] = (df.loc[i, 'R_t']+prob_switch)/(1-prob_switch)/(1-prob_rwd)
            if(df.loc[i+1, 'choice'] != df.loc[i, 'choice']):
                df.loc[i + 1, 'R_t'] = (1-prob_switch)/(df.loc[i, 'R_t']+prob_switch)/(1-prob_rwd)

        if df.loc[i + 1, 'choice'] == 'right':
            df.loc[i + 1, 'V_t'] = prob_rwd * (1 - 2 *(df.loc[i + 1, 'R_t']) / (1 + df.loc[i + 1, 'R_t'])) 
            
        if df.loc[i + 1, 'choice'] == 'left':
            df.loc[i + 1, 'V_t'] = prob_rwd * (1 - 2 / (df.loc[i + 1, 'R_t'] + 1))

    print(df['V_t'])
    #print(df['R_t'])

    #create a column where the side matches the regression notaion:
    df.loc[df['choice'] == 'right', 'choice_num'] = 1
    df.loc[df['choice'] == 'left', 'choice_num'] = 0
    df['choice'] = pd.to_numeric(df['choice'].fillna('other'), errors='coerce')
    return df

def compute_values_manually(df,prob_switch,prob_rwd) -> pd.DataFrame:
    df= df.reset_index(drop=True)
    def_new = df.copy()
    def_new['V_t'] = np.nan
    def_new['R_t'] = np.nan
    def_new = def_new.dropna(subset=['probability_r']).reset_index(drop=True)
    def_new.loc[0, 'V_t'] = prob_rwd * (0.7 - 0.3)
    def_new.loc[0, 'R_t'] = 0
    print(def_new['probability_r'])
    for i in range(len(def_new) - 1):
        # Comment the following two lines if we are considering a constant probability of switching
        if (def_new.loc[i+1,'side'] == 'right'): prob_quocient = (1 - def_new.loc[i+1,'probability_r'])/def_new.loc[i+1,'probability_r']
        if (def_new.loc[i+1,'side'] == 'left'): prob_quocient = def_new.loc[i+1,'probability_r']/(1 - def_new.loc[i+1,'probability_r'])
        if(def_new.loc[i+1, 'side'] == def_new.loc[i, 'side']):
            def_new.loc[i + 1, 'R_t'] = (def_new.loc[i, 'R_t']+prob_switch)/(1-prob_switch)*prob_quocient
        if(def_new.loc[i+1, 'side'] != def_new.loc[i, 'side']):
            def_new.loc[i + 1, 'R_t'] = (1-prob_switch)/(def_new.loc[i, 'R_t']+prob_switch)*prob_quocient
        if def_new.loc[i + 1, 'side'] == 'right':
            def_new.loc[i + 1, 'V_t'] = prob_rwd * (1 - 2 / ((1 + 1 / def_new.loc[i + 1, 'R_t']) ))
        if def_new.loc[i + 1, 'side'] == 'left':
            def_new.loc[i + 1, 'V_t'] = prob_rwd * (1 - 2 / (def_new.loc[i + 1, 'R_t'] + 1))
    #A column with the choice of the mice will now be constructed
    def_new.loc[(def_new['outcome_bool'] == 0) & (def_new['side'] == 'right'), 'choice'] = 'left'
    def_new.loc[(def_new['outcome_bool'] == 1) & (def_new['side'] == 'left'), 'choice'] = 'left'
    def_new.loc[(def_new['outcome_bool'] == 0) & (def_new['side'] == 'left'), 'choice'] = 'right'
    def_new.loc[(def_new['outcome_bool'] == 1) & (def_new['side'] == 'right'), 'choice'] = 'right'
    def_new['choice'].fillna('other', inplace=True)

    #create a column where the side matches the regression notaion:
    def_new.loc[def_new['choice'] == 'right', 'choice_num'] = 1
    def_new.loc[def_new['choice'] == 'left', 'choice_num'] = 0
    def_new['choice'] = pd.to_numeric(def_new['choice'].fillna('other'), errors='coerce')
    print(def_new['V_t'])
    return def_new

def manual_computation(df: pd.DataFrame, prob_switch: float, prob_rwd: float, n_back: int) -> pd.DataFrame:
    #A column with the choice of the mice will now be constructed
    new_df = df.copy()
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'right'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'left'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'left'), 'choice'] = 1
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'right'), 'choice'] = 1
    new_df['choice'] = pd.to_numeric(new_df['choice'].fillna('other'), errors='coerce')
    new_df['choice_num'] = new_df['choice']
    #Choide-reward will be created
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 0), 'choice_rwd'] = '00'    
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 1), 'choice_rwd'] = '01'  
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 0), 'choice_rwd'] = '10'  
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 1), 'choice_rwd'] = '11'  
    new_df['choice_rwd'] = new_df['choice_rwd'].fillna(' ')
    new_df = new_df.dropna(subset=['probability_r']).reset_index(drop=True)
    new_df['sequence'] = ''
    for i in range(n_back):
        new_df[f'choice_rwd{i+1}'] = new_df.groupby('session')['choice_rwd'].shift(i+1)
        new_df['sequence'] = new_df['sequence'] + new_df[f'choice_rwd{i+1}']
    new_df = new_df.dropna(subset=['sequence']).reset_index(drop=True)
    new_df['right_active'] = 0
    new_df['left_active'] = 0
    new_df.loc[(new_df['probability_r'] > 0.5), 'right_active'] = 1
    new_df.loc[(new_df['probability_r'] < 0.5), 'left_active'] = 1

    new_df['right_outcome'] = new_df.groupby('session')['right_active'].shift(-1)
    new_df['left_outcome'] = new_df.groupby('session')['left_active'].shift(-1)
    new_df['prob_right'] = new_df.groupby('sequence')['right_outcome'].transform('mean')
    new_df['prob_left'] = new_df.groupby('sequence')['left_outcome'].transform('mean')
    #print(new_df[new_df['prob_left'] < 0.5])
    #les probabilitats no surten complementàries
    new_df['V_t'] = prob_rwd*(new_df['prob_right']- new_df['prob_left'])
    #print(new_df['V_t'])
    return new_df




def inference_plot(prob_switch,prob_rwd,df):
    mice_counter = 0
    n_subjects = len(df['subject'].unique())
    avaluate= 1
    if not avaluate:
        n_cols = int(np.ceil(n_subjects / 2))
        f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
        for mice in df['subject'].unique():
            if mice != 'A10':
                df_mice = df.loc[df['subject'] == mice]
                session_counts = df_mice['session'].value_counts()
                mask = df_mice['session'].isin(session_counts[session_counts > 50].index)
                df_mice['sign_session'] = 0
                df_mice.loc[mask, 'sign_session'] = 1
                new_df_mice = df_mice[df_mice['sign_session'] == 1]
                #print(new_df_mice)
                #print(new_df_mice['subject'])
                print(mice)
                #df_values_new = compute_values(new_df_mice,  prob_switch, prob_rwd)
                #df_values_new = compute_values_manually(new_df_mice,  prob_switch, prob_rwd)
                df_values_new = manual_computation(new_df_mice,  prob_switch, prob_rwd,n_back=5)
                df_80, df_20 = select_train_sessions(df_values_new)
                ax = axes[mice_counter//n_cols, mice_counter%n_cols]
                psychometric_fit(ax,[[df_80,df_20]])
                ax.set_title(f'Psychometric Function: {mice}')
                ax.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.set_xlabel('Evidence')
                ax.set_ylabel('Prob of going right')
                ax.legend(loc='upper left')
                mice_counter += 1
        plt.show()
    else:
        n_back_vect = np.array([1,3,5,7])
        unique_subjects = df['subject'][df['subject'] != 'A10'].unique()
        errors = np.zeros((len(unique_subjects),len(n_back_vect)))
        #vector wit the trials back we are considering (the memory of the mice)
        phi = 1
        for i in range(len(n_back_vect)):
            mice_counter = 0
            for mice in unique_subjects:
                df_mice = df.loc[df['subject'] == mice]
                session_counts = df_mice['session'].value_counts()
                mask = df_mice['session'].isin(session_counts[session_counts > 50].index)
                df_mice['sign_session'] = 0
                df_mice.loc[mask, 'sign_session'] = 1
                new_df_mice = df_mice[df_mice['sign_session'] == 1]
                df_values = manual_computation(new_df_mice,  prob_switch, prob_rwd,n_back_vect[i])
                df_80, df_20 = select_train_sessions(df_values)
                errors[mice_counter][i]  = avaluation(df_20,df_80)
                mice_counter += 1
            print(errors)
            print('phi=', phi)
            print(errors[:,i])
            plt.plot(range(0, len(unique_subjects)), errors[:,i], color='green',marker='o',label = f'n = {n_back_vect[i]}', alpha = phi)
            plt.xticks(range(0, len(unique_subjects)), unique_subjects)
            phi = phi - 1/(len(n_back_vect))
        plt.xlabel('Mice')
        plt.ylabel('Error')
        plt.title('Weighed error of the inference-based model')
        plt.legend(loc='upper right')
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