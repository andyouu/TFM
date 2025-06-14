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
    new_df = df[['session', 'outcome', 'side', 'iti_duration']]
    new_df = new_df.copy()
    new_df['outcome_bool'] = np.where(new_df['outcome'] == "correct", 1, 0)

    #A column with the choice of the mice will now be constructed
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'right'), 'choice'] = 'left'
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'left'), 'choice'] = 'left'
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'left'), 'choice'] = 'right'
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'right'), 'choice'] = 'right'
    new_df['choice'].fillna('other', inplace=True)
    new_df.loc[new_df['choice'] == 'right', 'choice_num'] = 1
    new_df.loc[new_df['choice'] == 'left', 'choice_num'] = -1
    new_df['choice_num'] = pd.to_numeric(new_df['choice_num'].fillna('other'), errors='coerce')
    new_df['choice_num_+1'] = new_df.groupby('session')['choice'].shift(1)
    new_df['choice_num_+1'] = pd.to_numeric(new_df['choice_num_+1'].fillna('other'), errors='coerce')

    print(new_df['choice_num'] )
    print(40*'__')
    new_df['choice_num_+1'] 

    # prepare the data for the switch regresor switch s_k
    new_df.loc[(new_df['choice']  != new_df['choice_num_+1']), 'switch'] = 1
    new_df.loc[(new_df['choice']  == new_df['choice_num_+1']), 'switch'] = 0
    new_df['switch'] = pd.to_numeric(new_df['switch'].fillna('other'), errors='coerce')
    
    # prepare the data for the choice coditioned on outocome regressor r- 
    new_df.loc[new_df['outcome_bool'] == 1, 'r_minus']  = 0
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'right'), 'r_minus'] = 1
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'left'), 'r_minus'] = -1
    new_df['r_minus'] = pd.to_numeric(new_df['r_minus'].fillna('other'), errors='coerce')

    #create a column where the side matches the regression notaion:
    
    
    # build the regressors for previous trials
    regr_plus = ''
    regr_minus = ''
    for i in range(1, n + 1):
        new_df[f'r_plus_{i}'] = new_df.groupby('session')['r_plus'].shift(i)
        new_df[f'r_minus_{i}'] = new_df.groupby('session')['r_minus'].shift(i)
        regr_plus += f'r_plus_{i} + '
        regr_minus += f'r_minus_{i} + '
    regressors_string = regr_plus + regr_minus[:-3]

    return new_df, regressors_string

def plot_GLM(ax, GLM_df, alpha=1):
    """Summary: This function performs the glm plots

    Args:
        ax ([type]): [description]
        GLM_df ([type]): [description]
        alpha (int, optional): [description]. Defaults to 1.
    """ 
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    r_plus = GLM_df.loc[GLM_df.index.str.contains('r_plus'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.contains('r_minus'), "coefficient"]
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

def glm(df,n_bins_iti,iti_bins):
    """
    Defines the glm formula and fits the model with de data available

    Args:
        n_bins_iti (Int): number of bins
        iti_bins (Array): intervals (bins), of iti values
    """
    mice_counter = 0
    f, axes = plt.subplots(1, len(df['subject'].unique()), figsize=(15, 5), sharey=True)
    # iterate over mice
    for mice in df['subject'].unique():
        df_mice = df.loc[df['subject'] == mice]
        df_mice=df_mice.copy()
        # fit glm ignoring iti values
        df_mice['iti_bins'] = pd.cut(df_mice['iti_duration'], iti_bins)
        df_glm_mice, regressors_string = obt_regressors(df=df_mice,n=10)
        # get 3 equipopulated bins of iti values
        for iti_index in range(n_bins_iti-1):
            iti = [iti_bins[iti_index], iti_bins[iti_index + 1]]
            # get all trials within the iti bin
            df_glm_mice_iti = df_glm_mice[df_glm_mice['iti_duration'].between(iti[0], iti[1])]
            mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_glm_mice_iti).fit()
            GLM_df = pd.DataFrame({
                'coefficient': mM_logit.params,
                'std_err': mM_logit.bse,
                'z_value': mM_logit.tvalues,
                'p_value': mM_logit.pvalues,
                'conf_Interval_Low': mM_logit.conf_int()[0],
                'conf_Interval_High': mM_logit.conf_int()[1]
            })
            #print(GLM_df['coefficient']['r_plus_1'])
            # alpha = 1 if iti_index == 0 else subtract 0.3 for each iti_index
            alpha = 1 - 0.3 * iti_index
            # subplot title with name of mouse
            axes[mice_counter].set_title(mice)
            plot_GLM(axes[mice_counter], GLM_df, alpha=alpha)
        #psychometric_data(df_glm_mice,GLM_df,regressors_string)
        mice_counter += 1
    plt.show()



if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    df = df[df['task'] != 'S4']
    df = df[df['subject'] != 'manual']
    #select the iti bins
    iti_bins = [0, 2, 6, 12, 20]
    n_iti_bins = len(iti_bins)
    glm(df,n_iti_bins,iti_bins)