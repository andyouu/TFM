
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
from model_avaluation import *


def obt_regressors(df,n) -> Tuple[pd.DataFrame, str]:
    """
    Summary:
    This function processes the data needed to obtain the regressors and derives the 
    formula for the glm

    Args:
        df ([Dataframe]): [dataframe with experimental data]
        n ([int]): [number of trials back considered]

    Returns:
        new_df([Dataframe]): [dataframe with processed data restrcted to the regression]
        regressors_string([string]) :  [regressioon formula]
    """
    # Select the columns needed for the regressors
    new_df = df[['session', 'outcome', 'side', 'iti_duration','probability_r']]
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
    
    # prepare the data for the correct_choice regresor L_+  
    new_df.loc[new_df['outcome_bool'] == 0, 'r_plus']  = 0
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 'left'), 'r_plus'] = -1
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 'right'), 'r_plus'] = 1
    new_df['r_plus'] = pd.to_numeric(new_df['r_plus'].fillna('other'), errors='coerce')
    
    # prepare the data for the wrong_choice regressor L- 
    new_df.loc[new_df['outcome_bool'] == 1, 'r_minus']  = 0
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 'left'), 'r_minus'] = -1
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 'right'), 'r_minus'] = 1
    new_df['r_minus'] = pd.to_numeric(new_df['r_minus'].fillna('other'), errors='coerce')

    #prepare the switch regressor
    new_df['choice_1'] = new_df.groupby('session')['choice'].shift(1)
    new_df.loc[(new_df['choice'] == new_df['choice_1']), 'switch_num'] = 0
    new_df.loc[(new_df['choice'] != new_df['choice_1']), 'switch_num'] = 1

    #
    new_df['outcome_bool_1'] = new_df.groupby('session')['outcome_bool'].shift(1)

    # build the regressors for previous trials
    regr_plus = ''
    regr_minus = ''
    for i in range(2, n + 1):
        new_df[f'choice_{i}'] = new_df.groupby('session')['choice'].shift(i)
        
        #prepare the data for the error_switch regressor rss_-
        new_df.loc[(new_df[f'choice_{i}'] == new_df['choice_1']) & (new_df['outcome_bool_1'] == 0), f'rss_minus{i}'] = 1
        new_df.loc[(new_df[f'choice_{i}'] == new_df['choice_1']) & (new_df['outcome_bool_1'] == 1), f'rss_minus{i}'] = 0
        new_df.loc[new_df[f'choice_{i}'] != new_df['choice_1'], f'rss_minus{i}'] = 0
        new_df[f'rss_minus{i}'] = pd.to_numeric(new_df[f'rss_minus{i}'].fillna('other'), errors='coerce')

        #prepare the data for the error_switch regressor rss_-
        new_df.loc[(new_df[f'choice_{i}'] == new_df['choice_1']) & (new_df['outcome_bool_1'] == 1), f'rss_plus{i}'] = 1
        new_df.loc[(new_df[f'choice_{i}'] == new_df['choice_1']) & (new_df['outcome_bool_1'] == 0), f'rss_plus{i}'] = 0
        new_df.loc[new_df[f'choice_{i}'] != new_df['choice_1'], f'rss_plus{i}'] = 0
        new_df[f'rss_plus{i}'] = pd.to_numeric(new_df[f'rss_plus{i}'].fillna('other'), errors='coerce')
        regr_plus += f'rss_plus{i} + '
        regr_minus += f'rss_minus{i} + '
        print(new_df[f'rss_plus{i}'])
    regressors_string = regr_plus + regr_minus[:-3]

    return new_df, regressors_string

def plot_GLM(ax, GLM_df, alpha=1):
    """
    Summary: In this function all the plotting of the glms is performed

    Args:
        ax ([type]): [description]
        GLM_df ([type]): [description]
        alpha (int, optional): [description]. Defaults to 1.
    """
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    r_plus = GLM_df.loc[GLM_df.index.str.contains('rss_plus'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.contains('rss_minus'), "coefficient"]
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(r_plus)], r_plus, marker='o', color='indianred', alpha=alpha)
    ax.plot(orders[:len(r_minus)], r_minus, marker='o', color='teal', alpha=alpha)

    # Create custom legend handles with labels and corresponding colors
    legend_handles = [
        mpatches.Patch(color='indianred', label='r+'),
        mpatches.Patch(color='teal', label='r-')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles)
    # ax.axhline(y=intercept, label='Intercept', color='black')
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_ylabel('GLM weight')
    ax.set_xlabel('Previous trials')

def glm(df):
    mice_counter = 0
    n_subjects = len(df['subject'].unique())
    avaluate = 0
    if not avaluate:
        n_cols = int(np.ceil(n_subjects / 2))
        f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
        f1, axes1 = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
        # iterate over mice
        for mice in df['subject'].unique():
            if mice != 'A10':
                print(mice)
                df_mice = df.loc[df['subject'] == mice]
                # fit glm ignoring iti values
                #df_mice['iti_bins'] = pd.cut(df_mice['iti_duration'], iti_bins)
                #print(df_mice['subject'])
                df_glm_mice, regressors_string = obt_regressors(df=df_mice,n = 10)
                df_80, df_20 = select_train_sessions(df_glm_mice)
                mM_logit = smf.logit(formula='switch_num ~ ' + regressors_string, data=df_80).fit()
                GLM_df = pd.DataFrame({
                    'coefficient': mM_logit.params,
                    'std_err': mM_logit.bse,
                    'z_value': mM_logit.tvalues,
                    'p_value': mM_logit.pvalues,
                    'conf_Interval_Low': mM_logit.conf_int()[0],
                    'conf_Interval_High': mM_logit.conf_int()[1]
                })
                # subplot title with name of mouse
                ax = axes[mice_counter//n_cols, mice_counter%n_cols]
                ax1 = axes1[mice_counter//n_cols, mice_counter%n_cols]

                ax.set_title(f'GLM weights: {mice}')
                ax1.set_title(f'Psychometric Function: {mice}')
                plot_GLM(ax, GLM_df)
                #data_label can be either 'choice_num' or 'probability_r'
                psychometric_data(ax1, df_20, GLM_df, regressors_string,'switch_num')
                ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.set_xlabel('Evidence')
                ax1.set_ylabel('Prob of switching')
                ax1.legend(loc='upper left')
                mice_counter += 1
        plt.tight_layout()
        plt.show()
    else:
        unique_subjects = df['subject'][df['subject'] != 'A10'].unique()
        n_back_vect = [2,3,5,7]
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
                df_glm_mice, regressors_string = obt_regressors(df=new_df_mice,n = n_back_vect[i])
                df_80, df_20 = select_train_sessions(df_glm_mice)
                mM_logit = smf.logit(formula='switch_num ~ ' + regressors_string, data=df_80).fit()
                GLM_df = pd.DataFrame({
                    'coefficient': mM_logit.params,
                    'std_err': mM_logit.bse,
                    'z_value': mM_logit.tvalues,
                    'p_value': mM_logit.pvalues,
                    'conf_Interval_Low': mM_logit.conf_int()[0],
                    'conf_Interval_High': mM_logit.conf_int()[1]
                })
                errors[mice_counter][i]  = avaluation(GLM_df,df_80,regressors_string,'switch_num')
                mice_counter += 1
            print(errors)
            print('phi=', phi)
            print(errors[:,i])
            plt.plot(range(0, len(unique_subjects)), errors[:,i], color='blue',marker='o',label = f'n = {n_back_vect[i]}', alpha = phi)
            plt.xticks(range(0, len(unique_subjects)), unique_subjects)
            phi = phi - 1/(len(n_back_vect))
        plt.xlabel('Mice')
        plt.ylabel('Error')
        plt.title('Weighed error of the switch logistic model')
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
    glm(df)