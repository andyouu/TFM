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
from parsing import parsing



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
    new_df.loc[new_df['probability_r'] > 0.7, 'prob_high'] = 1
    new_df.loc[new_df['probability_r'] < 0.8, 'prob_high'] = 0
    new_df.loc[new_df['probability_r'] > 0.7, 'prob_low'] = 0
    new_df.loc[new_df['probability_r'] < 0.8, 'prob_low'] = 1
    
    #build the block_probability regressor
    new_df['prob_block'] = np.nan
    new_df.loc[new_df['probability_r'] > 0.5, 'prob_block'] = new_df['probability_r']
    new_df.loc[new_df['probability_r'] < 0.5, 'prob_block'] = new_df['probability_r'] - 1 # we want the probabilities to be symmetric, if we considered 
                                                                                          # p in both sides we would not give the same weight to both options

    for i in range(1, n+1):
        new_df[f'r_plus_{i}'] = new_df.groupby('session')['r_plus'].shift(i)
        new_df[f'r_minus_{i}'] = new_df.groupby('session')['r_minus'].shift(i)
        new_df[f'prb_r_r_plus{i}'] = new_df[f'r_plus_{i}']*new_df['prob_block']
        new_df[f'prb_r_r_minus{i}'] = new_df[f'r_minus_{i}']*new_df['prob_block']

    regr_plus = ''
    regr_minus = ''
    for i in range(1, n + 1):
        #regr_plus += f'r_plus_{i} + ' f'prb_r_r_plus{i} + '
        #regr_minus += f'r_minus_{i} + ' f'prb_r_r_minus{i} + '
        #try method
        regr_plus += f'r_plus_{i} + prob_block * 'f'r_plus_{i} + '
        regr_minus += f'r_minus_{i} + prob_block * 'f'r_minus_{i} + '
    
    #add the for the block probability regressror
    regressors_string = 'prob_block + ' + regr_plus + regr_minus[:-3]


    #exponential regressor
    expo_yes = 1
    if expo_yes:
        new_df = new_df.reset_index(drop=True)
        tau_plus = 1.75
        tau_minus = 0.24
        new_df['exponent_plus'] = np.nan
        new_df['exponent_minus'] = np.nan
        new_df.loc[0,'exponent_plus'] = 0
        new_df.loc[0,'exponent_minus'] = 0
        for i in range(len(new_df)-1):
            if (new_df.loc[i+1, 'session'] == new_df.loc[i,'session']):
                new_df.loc[i+1, 'exponent_plus'] = np.exp(-1/tau_plus) * (new_df.loc[i,'r_plus'] + new_df.loc[i,'exponent_plus'])
                new_df.loc[i+1, 'exponent_minus'] = np.exp(-1/tau_minus) * ( new_df.loc[i,'r_minus'] + new_df.loc[i,'exponent_minus'])
            else:
                new_df.loc[i+1, 'exponent_plus'] = np.exp(-1/tau_plus) * new_df.loc[i,'r_plus']
                new_df.loc[i+1, 'exponent_minus'] = np.exp(-1/tau_minus) * new_df.loc[i,'r_minus']
        regressors_string = 'prob_block + ' + 'exponent_plus + ' + 'exponent_minus + ' + 'prob_block * exponent_plus + ' + 'prob_block * exponent_minus'
            
        
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
    r_plus = GLM_df.loc[GLM_df.index.str.startswith('r_plus_'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.startswith('r_minus_'), "coefficient"]
    pr_r_plus = GLM_df.loc[GLM_df.index.str.startswith('prob_block:r_p'), "coefficient"]
    pr_r_minus = GLM_df.loc[GLM_df.index.str.startswith('prob_block:r_m'), "coefficient"]


    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(r_plus)], r_plus, marker='o', color='indianred', alpha=alpha)
    ax.plot(orders[:len(r_minus)], r_minus, marker='o', color='teal', alpha=alpha)
    ax.plot(orders[:len(pr_r_plus)], pr_r_plus, marker='o', color='indianred', alpha=alpha-0.5)
    ax.plot(orders[:len(pr_r_minus)], pr_r_minus, marker='o', color='teal', alpha=alpha-0.5)



    # Create custom legend handles with labels and corresponding colors
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

def glm(df):
    mice_counter = 0
    n_subjects = len(df['subject'].unique())
    avaluate = 0
    if not avaluate:
        exponentiate = 1
        if not exponentiate:
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
                    df_glm_mice, regressors_string = obt_regressors(df=df_mice,n = 5)
                    df_80, df_20 = select_train_sessions(df_glm_mice)
                    print(regressors_string)
                    mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_80).fit()
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
                    ax1 = axes1[mice_counter//n_cols, mice_counter%n_cols]

                    ax.set_title(f'GLM weights: {mice}')
                    ax1.set_title(f'Psychometric Function: {mice}')
                    plot_GLM(ax, GLM_df)
                    #data_label can be either 'choice_num' or 'probability_r'
                    #psychometric_data(ax1, df_20, GLM_df, regressors_string,'choice_num')
                    ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax1.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax1.set_xlabel('Evidence')
                    ax1.set_ylabel('Prob of going right')
                    ax1.legend(loc='upper left')
                    mice_counter += 1
            plt.tight_layout()
            plt.show()
        else:
            all_mice_coefficients = pd.DataFrame()

            # Loop through each mouse
            for mice in df['subject'].unique():
                if mice != 'A10':  # Exclude 'A10'
                    print(f"Processing mouse: {mice}")
                    
                    # Filter data for the current mouse
                    df_mice = df.loc[df['subject'] == mice]
                    
                    # Obtain regressors and split data into training and testing sets
                    df_glm_mice, regressors_string = obt_regressors(df=df_mice, n=5)
                    df_80, df_20 = select_train_sessions(df_glm_mice)
                    
                    # Fit the logistic regression model
                    mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_80).fit()
                    
                    # Extract coefficients and other statistics
                    GLM_df = pd.DataFrame({
                        'coefficient': mM_logit.params,
                        'std_err': mM_logit.bse,
                        'z_value': mM_logit.tvalues,
                        'p_value': mM_logit.pvalues,
                        'conf_Interval_Low': mM_logit.conf_int()[0],
                        'conf_Interval_High': mM_logit.conf_int()[1]
                    })
                    
                    # Add a column for the mouse ID
                    GLM_df['mouse'] = mice
                    
                    # Append the results to the all_mice_coefficients DataFrame
                    all_mice_coefficients = pd.concat([all_mice_coefficients, GLM_df], axis=0)

            # Reset index for the combined DataFrame
            all_mice_coefficients = all_mice_coefficients.reset_index().rename(columns={'index': 'regressor'})
            # Plot box plot with individual data points
            plt.figure(figsize=(12, 6))

            # Create the box plot
            sns.boxplot(x='regressor', y='coefficient', data=all_mice_coefficients, color='lightblue', width=0.6)

            # Overlay the individual data points
            sns.stripplot(x='regressor', y='coefficient', data=all_mice_coefficients, color='black', alpha=0.6, jitter=True)

            # Add title and labels
            plt.title('Distribution of Logistic Regression Coefficients Across Mice')
            plt.xlabel('Regressors')
            plt.ylabel('Coefficient Value')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
   
    else:
        unique_subjects = df['subject'][df['subject'] != 'A10'].unique()
        n_back_vect = [1,3,5,7]
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
                mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_80).fit()
                GLM_df = pd.DataFrame({
                    'coefficient': mM_logit.params,
                    'std_err': mM_logit.bse,
                    'z_value': mM_logit.tvalues,
                    'p_value': mM_logit.pvalues,
                    'conf_Interval_Low': mM_logit.conf_int()[0],
                    'conf_Interval_High': mM_logit.conf_int()[1]
                })
                errors[mice_counter][i]  = avaluation(GLM_df,df_80,regressors_string,'choice_num')
                mice_counter += 1
            print(errors)
            print('phi=', phi)
            print(errors[:,i])
            plt.plot(range(0, len(unique_subjects)), errors[:,i], color='blue',marker='o',label = f'n = {n_back_vect[i]}', alpha = phi)
            plt.xticks(range(0, len(unique_subjects)), unique_subjects)
            phi = phi - 1/(len(n_back_vect))
        plt.xlabel('Mice')
        plt.ylabel('Error')
        plt.title('Weighed error of the prob right logistic model')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials1.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # 1 for analisis of trained mice, 0 for untrained
    print(df['task'].unique())
    trained = 1
    new_df = parsing(df,trained)
    glm(new_df)