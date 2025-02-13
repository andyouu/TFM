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

    #create a column where the side matches the regression notaion:
    new_df.loc[new_df['choice'] == 'right', 'choice_num'] = 1
    new_df.loc[new_df['choice'] == 'left', 'choice_num'] = 0
    new_df['choice'] = pd.to_numeric(new_df['choice'].fillna('other'), errors='coerce')

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
    #print("DataFrame with 0.8 of sessions:")
    #print(df_80)
    #print("\nDataFrame with 0.2 of sessions:")
    #print(df_20)
    return df_80,df_20

def avaluation(GLM_df,df_20,regressors_string,label_data):
    regressors_vect = regressors_string.split(' + ')
    coefficients = GLM_df['coefficient']
    df_20['evidence'] = 0
    for j in range(len(regressors_vect)):
        df_20['evidence']+= coefficients[regressors_vect[j]]*df_20[regressors_vect[j]]
    n_bins = 20
    df_20['binned_ev'] = pd.qcut(df_20['evidence'], n_bins,duplicates='drop')
    grouped = df_20.groupby('binned_ev').agg(
    ev_mean=('evidence', 'mean'),
    p_right_mean=(label_data, 'mean')
    ).dropna() 
    ev_means = grouped['ev_mean'].values
    p_right_mean = grouped['p_right_mean'].values
    bin_sizes = df_20['binned_ev'].value_counts(sort=False)
    #print(bin_sizes)
    weights = bin_sizes / np.sum(bin_sizes)  # Normalize weights to sum to 1
    predicted_p_right = psychometric(ev_means)
    wmse = np.sum(weights * (p_right_mean - predicted_p_right) ** 2)    
    print(wmse)
    return(wmse)


def glm(df):
    mice_counter = 0
    errors = np.zeros(len(df['subject'].unique()))
    for mice in df['subject'].unique():
        if mice != 'A10':
            print(mice)
            df_mice = df.loc[df['subject'] == mice]
            # fit glm ignoring iti values
            #df_mice['iti_bins'] = pd.cut(df_mice['iti_duration'], iti_bins)
            print(df_mice['subject'])
            df_glm_mice, regressors_string = obt_regressors(df=df_mice,n=10)
            df_80, df_20 = select_train_sessions(df_glm_mice)
            mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_80).fit()
            GLM_df = pd.DataFrame({
                'coefficient': mM_logit.params,
                'std_err': mM_logit.bse,
                'z_value': mM_logit.tvalues,
                'p_value': mM_logit.pvalues,
                'conf_Interval_Low': mM_logit.conf_int()[0],
                'conf_Interval_High': mM_logit.conf_int()[1]
            })
            mice_counter += 1
            errors[mice_counter]=avaluation(GLM_df,df_20,regressors_string)
            #psychometric_data(axes1[mice_counter],df_80,GLM_df,regressors_string)
    print(errors)
    plt.plot(range(0, len(df['subject'].unique())), errors, marker='o')
    plt.xticks(range(0, len(df['subject'].unique())), df['subject'].unique())
    plt.xlabel('Mice')
    plt.ylabel('Error')
    plt.title('Weighed error of the probability right GLM model')
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