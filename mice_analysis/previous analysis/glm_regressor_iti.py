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
from model_avaluation import *


def obt_regressors(df,n,iti_bins) -> Tuple[pd.DataFrame, str]:
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
    new_df = df[['session', 'outcome', 'side', 'iti_duration']]
    new_df = new_df.copy()
    new_df['outcome_bool'] = np.where(new_df['outcome'] == "correct", 1, 0)

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
    
    # prepare the data for the small_iti regressor  
    for i in range(len(iti_bins)-1):
        new_df.loc[(new_df['iti_duration'] > iti_bins[i]) & (new_df['iti_duration'] < iti_bins[i+1]), f'iti_bin_{i}'] = 1
    for i in range(len(iti_bins)-1):
        new_df[f'iti_bin_{i}'] = new_df.get(f'iti_bin_{i}', 0).fillna(0)
    #create a column where the side matches the regression notaion:
    new_df.loc[new_df['choice'] == 'right', 'choice_num'] = 1
    new_df.loc[new_df['choice'] == 'left', 'choice_num'] = 0
    new_df['choice'] = pd.to_numeric(new_df['choice'].fillna('other'), errors='coerce')


    #create the new regressors (product of iti_bins and r_+,r_-)
    for i in range(1, n + 1):
        new_df[f'r_plus__{i}'] = new_df.groupby('session')['r_plus'].shift(i)
        new_df[f'r_minus__{i}'] = new_df.groupby('session')['r_minus'].shift(i)
        for j in range(len(iti_bins)-1):
            new_df[f'r_plus_iti{i}{j}'] = new_df[f'r_plus__{i}']*new_df[f'iti_bin_{j}']
            new_df[f'r_minus_iti{i}{j}'] = new_df[f'r_minus__{i}']*new_df[f'iti_bin_{j}']

    for j in range(len(iti_bins)-1):        
        for i in range(1, n + 1):
            new_df[f'r_plus_iti{i}{j}'] = new_df.get(f'r_plus_iti{i}{j}', 0).fillna(0)
            new_df[f'r_minus_iti{i}{j}'] = new_df.get(f'r_minus_iti{i}{j}', 0).fillna(0)
    # build the regressors for previous trials
    regr_plus = ''
    regr_minus = ''
    regressors_string = ''
    for i in range(1, n + 1):
        regr_plus += f'r_plus__{i} + '
        regr_minus += f'r_minus__{i} + '
    for j in range(len(iti_bins)-1): regressors_string += f'iti_bin_{j} + '
    for j in range(len(iti_bins)-1):        
        for i in range(1, n + 1):
            regr_plus += f'r_plus_iti{i}{j} + '
            regr_minus += f'r_minus_iti{i}{j} + '
    regressors_string += regr_plus + regr_minus[:-3]

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
    r_plus = GLM_df.loc[GLM_df.index.str.contains('r_plus__'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.contains('r_minus__'), "coefficient"]
    plus_iti_bin_0 = GLM_df.loc[GLM_df.index.str.contains('r_plus_iti[1-5]0'), "coefficient"]
    plus_iti_bin_1 = GLM_df.loc[GLM_df.index.str.contains('r_plus_iti[1-5]1'), "coefficient"]
    plus_iti_bin_2 = GLM_df.loc[GLM_df.index.str.contains('r_plus_iti[1-5]2'), "coefficient"]
    plus_iti_bin_3 = GLM_df.loc[GLM_df.index.str.contains('r_plus_iti[1-5]3'), "coefficient"]
    minus_iti_bin_0 = GLM_df.loc[GLM_df.index.str.contains('r_minus_iti[1-5]0'), "coefficient"]
    minus_iti_bin_1 = GLM_df.loc[GLM_df.index.str.contains('r_minus_iti[1-5]1'), "coefficient"]
    minus_iti_bin_2 = GLM_df.loc[GLM_df.index.str.contains('r_minus_iti[1-5]2'), "coefficient"]
    minus_iti_bin_3 = GLM_df.loc[GLM_df.index.str.contains('r_minus_iti[1-5]3'), "coefficient"]

    print(40*'_')
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(r_plus)], r_plus, marker='o', color='indianred', alpha=alpha)
    ax.plot(orders[:len(r_minus)], r_minus, marker='o', color='teal', alpha=alpha)
    ax.plot(orders[:len(plus_iti_bin_0)], plus_iti_bin_0, marker='o', color='indianred', alpha=alpha-0.5)
    ax.plot(orders[:len(plus_iti_bin_1)], plus_iti_bin_0, marker='o', color='indianred', alpha=alpha-0.6)
    ax.plot(orders[:len(plus_iti_bin_2)], plus_iti_bin_1, marker='o', color='indianred', alpha=alpha-0.7)
    ax.plot(orders[:len(plus_iti_bin_3)], plus_iti_bin_3, marker='o', color='indianred', alpha=alpha-0.8)
    ax.plot(orders[:len(minus_iti_bin_0)], minus_iti_bin_0, marker='o', color='teal', alpha=alpha-0.5)
    ax.plot(orders[:len(minus_iti_bin_1)], minus_iti_bin_0, marker='o', color='teal', alpha=alpha-0.6)
    ax.plot(orders[:len(minus_iti_bin_2)], minus_iti_bin_1, marker='o', color='teal', alpha=alpha-0.7)
    ax.plot(orders[:len(minus_iti_bin_3)], minus_iti_bin_3, marker='o', color='teal', alpha=alpha-0.8)

    legend_handles = [
        mpatches.Patch(color='indianred', label=r'$r_+$'),
        mpatches.Patch(color='teal', label= r'$r_-$')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles)
    # ax.axhline(y=intercept, label='Intercept', color='black')
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_ylabel('GLM weight')
    ax.set_xlabel('Previous trials')


def glm(df,iti_bins):
    mice_counter = 0
    n_subjects = len(df['subject'].unique())
    n_cols = int(np.ceil(n_subjects / 2))
    f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)    
    # iterate over mice
    for mice in df['subject'].unique():
        if mice != 'A10':
            df_mice = df.loc[df['subject'] == mice]
            # fit glm ignoring iti values
            df_glm_mice, regressors_string = obt_regressors(df=df_mice,n=5,iti_bins=iti_bins)
            #print(regressors_string)
            #df_80, df_20 = select_train_sessions(df_glm_mice)
            mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_glm_mice).fit()
            GLM_df = pd.DataFrame({
                'coefficient': mM_logit.params,
                'std_err': mM_logit.bse,
                'z_value': mM_logit.tvalues,
                'p_value': mM_logit.pvalues,
                'conf_Interval_Low': mM_logit.conf_int()[0],
                'conf_Interval_High': mM_logit.conf_int()[1]
            })
            print(GLM_df['coefficient'])
            # subplot title with name of mouse
            ax = axes[mice_counter//n_cols, mice_counter%n_cols]
            ax.set_title(f'GLM weights: {mice}')
            plot_GLM(ax, GLM_df)
            mice_counter += 1
    plt.show()


if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials_maybe_updated.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    df = df[df['task'] != 'S4']
    df = df[df['subject'] != 'manual']
    #select the iti bins
    glm(df,[2,4,6,12,20])