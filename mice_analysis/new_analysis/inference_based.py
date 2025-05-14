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
from parsing import parsing


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

    return def_new

def manual_computation(df: pd.DataFrame, prob_switch: float, prob_rwd: float, n_back: int) -> pd.DataFrame:
    #A column with the choice of the mice will now be constructed
    df= df.reset_index(drop=True)
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

    new_df['side_num'] = np.nan
    new_df.loc[new_df['side'] == 'right', 'side_num'] = 1
    #Vertechi I believe states that s(left) = -1 but the fit looks to be atrocious for that value
    new_df.loc[new_df['side'] == 'left', 'side_num'] = 0
    return new_df

def plot_all_mice_correct_inf_combined(df, figsize=(46.8, 33.1)):
    """
    Plot ALL mice's correct inference weights in a SINGLE figure with:
    - Custom colors for coefficient types
    - Mice differentiated by alpha levels
    - A0 poster sizing
    """
    # Set your custom colors
    beta_color = '#d62728'  # Red for beta (V_t)
    side_color = '#1f77b4'  # Blue for side bias
    
    # Set global styling for poster
    plt.rcParams.update({
        'axes.titlesize': 50,
        'axes.labelsize': 50,
        'xtick.labelsize': 35,
        'ytick.labelsize': 35,
        'legend.fontsize': 25,
        'lines.linewidth': 4,
        'lines.markersize': 15
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get list of mice (excluding A10)
    mice_list = [m for m in df['subject'].unique() if m != 'A10']
    n_mice = len(mice_list)
    
    # Create alpha levels for mice (from light to dark)
    alphas = np.linspace(0.3, 1, n_mice)
    
    # Plot each mouse's coefficients
    for i, mice in enumerate(mice_list):
        df_mice = df.loc[df['subject'] == mice]
        
        # Filter sessions with sufficient trials
        session_counts = df_mice['session'].value_counts()
        mask = df_mice['session'].isin(session_counts[session_counts > 50].index)
        df_mice['sign_session'] = 0
        df_mice.loc[mask, 'sign_session'] = 1
        new_df_mice = df_mice[df_mice['sign_session'] == 1]
        
        # Compute values and fit model
        df_values_new = manual_computation(new_df_mice, prob_switch, prob_rwd, n_back=3)
        df_80, _ = select_train_sessions(df_values_new)
        
        try:
            mM_logit = smf.logit(formula='choice ~ V_t + side_num', data=df_80).fit()
        except Exception as e:
            print(f"Model fitting failed for {mice}: {str(e)}")
            continue  # Skip this mouse if model fails

        # Create coefficients DataFrame
        GLM_df = pd.DataFrame({
            'coefficient': mM_logit.params,
            'std_err': mM_logit.bse,
            'z_value': mM_logit.tvalues,
            'p_value': mM_logit.pvalues,
            'conf_Interval_Low': mM_logit.conf_int()[0],
            'conf_Interval_High': mM_logit.conf_int()[1],
            'regressor': mM_logit.params.index
        })

        # Get coefficients
        beta = GLM_df.loc[GLM_df['regressor'].str.contains('V_t'), 'coefficient'].values[0]
        side = GLM_df.loc[GLM_df['regressor'].str.contains('side_num'), 'coefficient'].values[0]
        
        # Plot with your colors and mouse-specific alpha
        ax.bar(i-0.2, beta, width=0.4, color=beta_color, alpha=alphas[i], label=f'{mice} β' if i == 0 else "")
        ax.bar(i+0.2, side, width=0.4, color=side_color, alpha=alphas[i], label=f'{mice} side' if i == 0 else "")
    
    # Add reference line and styling
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_title('Combined Correct Inference Weights', pad=20)
    ax.set_ylabel('Coefficient Value', labelpad=20)
    ax.set_xticks(np.arange(n_mice))
    ax.set_xticklabels(mice_list, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.3)
    
    # Create simplified legends
    # Legend 1: Coefficient types with your colors
    coeff_handles = [
        mpatches.Patch(color=beta_color, label='β (Value)'),
        mpatches.Patch(color=side_color, label='Side Bias')
    ]
    legend1 = ax.legend(handles=coeff_handles, title='Coefficient Types',
                       loc='upper left', bbox_to_anchor=(1.01, 1))
    
    # Legend 2: Mice with alpha gradient
    mice_handles = []
    for i, mice in enumerate(mice_list):
        mice_handles.append(mpatches.Patch(color='gray', alpha=alphas[i], label=mice))
    
    ax.legend(handles=mice_handles, title='Mice (by alpha)',
             loc='lower left', bbox_to_anchor=(1.01, 0))
    
    # Add the first legend back
    ax.add_artist(legend1)
    
    # Adjust layout
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(right=0.75, bottom=0.2)  # Make space for legends and x-labels
    plt.show()


def inference_plot(prob_switch,prob_rwd,df):
    mice_counter = 0
    n_subjects = len(df['subject'].unique())
    avaluate= 0
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
        plot_all_mice_correct_inf_combined(df, figsize=(46.8, 33.1))
    else:
        #this have been chosen to ensure enough bins result from the processing of the data
        n_back_vect = np.array([4,5,7,8])
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
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials1.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    #df = df[df['task'] != 'S4']
    df = df[df['subject'] != 'manual']
    prob_rwd = 0.99
    prob_switch = 0.5
    new_df = df[['subject','session', 'outcome', 'side', 'iti_duration','probability_r','task','date']]
    new_df = new_df.copy()
    new_df['outcome_bool'] = np.where(new_df['outcome'] == "correct", 1, 0)
    trained = 1
    new_df = parsing(new_df, trained)
    inference_plot(prob_switch,prob_rwd,new_df)
