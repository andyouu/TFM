import pandas as pd
import numpy as np
from typing import Tuple
from datetime import timedelta
from collections import defaultdict
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
import matplotlib.dates as mdates
import statsmodels.formula.api as smf
import os
import matplotlib.patches as mpatches
import scipy
from matplotlib import rcParams
from extra_plotting import *
from model_avaluation import *
from parsing import *


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
    for i in range(1, n):
        new_df[f'r_plus_{i}'] = new_df.groupby('session')['r_plus'].shift(i)
        new_df[f'r_minus_{i}'] = new_df.groupby('session')['r_minus'].shift(i)
        regr_plus += f'r_plus_{i} + '
        regr_minus += f'r_minus_{i} + '
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
    r_plus = GLM_df.loc[GLM_df.index.str.contains('r_plus'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.contains('r_minus'), "coefficient"]
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(r_plus)], r_plus, marker='o', color='indianred', alpha=alpha)
    ax.plot(orders[:len(r_minus)], r_minus, marker='o', color='teal', alpha=alpha)

    # Create custom legend handles with labels and corresponding colors
    legend_handles = [
        mpatches.Patch(color='indianred', label=r'$\beta_{t}^+$'),
        mpatches.Patch(color='teal', label= r'$\beta_{t}^-$')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles, location = 'best', bbox_to_anchor=(0.8, 0.5),)
    # ax.axhline(y=intercept, label='Intercept', color='black')
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_ylabel('GLM weight')
    ax.set_xlabel('Previous trials')

def plot_all_mice_glm_combined(df, n_back, figsize=(46.8, 33.1)):
    """
    Plot ALL mice's GLM weights with significance testing using Fisher's method,
    including the intercept's combined p-value.
    """
    # Set colors and styling
    plus_color = '#d62728'  # Red for r_plus
    minus_color = '#1f77b4'  # Blue for r_minus
    intercept_color = 'green'  # Color for intercept
    
    plt.rcParams.update({
        'axes.titlesize': 50,
        'axes.labelsize': 50,
        'xtick.labelsize': 35,
        'ytick.labelsize': 35,
        'legend.fontsize': 25,
        'lines.linewidth': 4,
        'lines.markersize': 15
    })

    fig, ax = plt.subplots(figsize=figsize)
    mice_list = np.unique(df['subject'])
    mice_list = mice_list[mice_list!= 'A10']
    alphas = np.linspace(0.3, 1, len(mice_list))
    
    # Initialize storage for p-values (one list per lag + intercept)
    plus_pvalues_per_lag = defaultdict(list)
    minus_pvalues_per_lag = defaultdict(list)
    intercept_pvalues = []  # Store intercept p-values
    GLM_data = []
    # Collect p-values from each mouse
    all_metrics = []
    for mice in mice_list:
        df_mice = df.loc[df['subject'] == mice]
        df_glm_mice, regressors_string = obt_regressors(df=df_mice, n=n_back)
        #implement 5-fold cross-validation
        df_cv = select_train_sessions(df_glm_mice)
        cv_plus_pvalues_per_lag = defaultdict(list)
        cv_minus_pvalues_per_lag = defaultdict(list)
        cv_intercept_pvalues = []  # Store intercept p-values

        for i in range(5):
            df_80 = df_cv[df_cv[f'split_{i}'] == f'train']
            df_test = df_cv[df_cv[f'split_{i}'] == f'test']
            mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_80).fit()
            GLM_df = pd.DataFrame({
                'coefficient': mM_logit.params,
                'p_value': mM_logit.pvalues
            })
            GLM_df['subject'] = mice
            GLM_df['split'] = i
            df_reset = GLM_df.reset_index()
            df_reset = df_reset.rename(columns={'index': 'regressor'})
            GLM_data.append(df_reset)
            df_test['pred_prob'] = mM_logit.predict(df_test)
            n_regressors = len([x.strip() for x in regressors_string.split(' + ')])
            #Create a DataFrame with the avaluation metrics
            y_true = (
                df_test.groupby('session')['choice_num']
                .apply(lambda x: x.iloc[n_regressors:])
                .reset_index(drop=True)  # Flatten to a single Series
            )  # True binary outcomes
            predictions = []
            for session, group in df_test.groupby('session'):
                # Get predictions for this session only
                session_pred = mM_logit.predict(group)[n_regressors:]
                predictions.append(session_pred)
                
            y_pred_prob = pd.concat(predictions)  # Predicted probabilities (change this tot the test set)
            y_pred_class = (y_pred_prob >= 0.5).astype(int)
            np.random.seed(42) 
            y_pred_class_mult = (np.random.rand(len(y_pred_prob)) < y_pred_prob).astype(int) # We may use the multinomial here to choose with probability (sampling)
            if mice == 'B3' and n_back in [2,10] and i == 0:
                print('alert')
            metrics_dict = {
                # Log-likelihood
                "log_likelihood": mM_logit.llf,
                "log_likelihood_per_obs": mM_logit.llf / len(y_true),
                
                # Information criteria
                "AIC": mM_logit.aic,
                "BIC": mM_logit.bic,
                
                # Pseudo R-squared
                "pseudo_r2_mcfadden": mM_logit.prsquared,  # McFadden's pseudo R²
                "pseudo_r2_cox_snell": 1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true)),  # Cox-Snell
                "pseudo_r2_nagelkerke": (1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true))) / 
                                    (1 - np.exp(2 * mM_logit.llnull / len(y_true))),  # Nagelkerke
                
                # Classification metrics (threshold=0.5)
                "accuracy": accuracy_score(y_true, y_pred_class),
                "precision": precision_score(y_true, y_pred_class),
                "recall": recall_score(y_true, y_pred_class),
                "f1_score": f1_score(y_true, y_pred_class),
                "accuracy_bis": accuracy_score(y_true, y_pred_class_mult),
                "precision_bis": precision_score(y_true, y_pred_class_mult),
                "recall_bis": recall_score(y_true, y_pred_class_mult),
                "f1_score_bis": f1_score(y_true, y_pred_class_mult),
                
                # Probability-based metrics
                "roc_auc": roc_auc_score(y_true, y_pred_prob),
                "brier_score": brier_score_loss(y_true, y_pred_prob),
            }
            GLM_metrics = pd.DataFrame([metrics_dict])
            GLM_metrics['subject'] = mice
            GLM_metrics['split'] = i
            all_metrics.append(GLM_metrics)
            # Extract p-values for each regressor (including intercept)
            for i, reg in enumerate(mM_logit.params.index):
                if 'r_plus' in reg:
                    lag = int(reg.split('_')[-1])
                    cv_plus_pvalues_per_lag[lag].append(mM_logit.pvalues[i])
                elif 'r_minus' in reg:
                    lag = int(reg.split('_')[-1])
                    cv_minus_pvalues_per_lag[lag].append(mM_logit.pvalues[i])
                elif reg == 'Intercept':
                    cv_intercept_pvalues.append(mM_logit.pvalues[i])
            # Combine p-values across CV folds using median for this mouse
        for lag in cv_plus_pvalues_per_lag:
            plus_pvalues_per_lag[lag].append(np.median(cv_plus_pvalues_per_lag[lag]))
        for lag in cv_minus_pvalues_per_lag:
            minus_pvalues_per_lag[lag].append(np.median(cv_minus_pvalues_per_lag[lag]))
        intercept_pvalues.append(np.median(cv_intercept_pvalues))

    #if the path exists, remove it
    if os.path.exists(f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_glm_prob_r_{n_back}.csv'):
        os.remove(f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_glm_prob_r_{n_back}.csv')
    # Save combined GLM metrics to CSV
    combined_glm_metrics = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_glm_prob_r_{n_back}.csv'
    combined_metrics = pd.concat(all_metrics,ignore_index=True,axis=0)
    combined_metrics.to_csv(combined_glm_metrics, index=False)
    df_GLM_data = pd.concat(GLM_data)
    df_GLM_data = df_GLM_data.reset_index(drop=True)
    df_GLM_df = df_GLM_data.groupby(['regressor','subject'])['coefficient'].mean().reset_index()
    
    # Combine p-values using Stuffer's method for each lag and intercept
    fisher_plus = {}
    fisher_minus = {}
    for lag in plus_pvalues_per_lag:
        fisher_plus[lag] = scipy.stats.combine_pvalues(plus_pvalues_per_lag[lag], method='fisher')[1]
    for lag in minus_pvalues_per_lag:
        fisher_minus[lag] = scipy.stats.combine_pvalues(minus_pvalues_per_lag[lag], method='fisher')[1]
    
    # Combine intercept p-values
    if intercept_pvalues:  # Only if intercepts were found
        fisher_intercept = scipy.stats.combine_pvalues(intercept_pvalues, method='fisher')[1]
    
    for i, mice in enumerate(mice_list):
        r_plus = df_GLM_df[df_GLM_df['regressor'].str.contains('r_plus') & (df_GLM_df['subject'] == mice)]['coefficient'].values
        r_minus = df_GLM_df[df_GLM_df['regressor'].str.contains('r_minus') & (df_GLM_df['subject'] == mice)]['coefficient'].values
        intercept = df_GLM_df[(df_GLM_df['regressor'] == 'Intercept') & (df_GLM_df['subject'] == mice)]['coefficient'].values
        ax.plot(np.arange(len(r_plus))+1, r_plus, 'o-', color=plus_color,label = mice, alpha=alphas[i])
        ax.plot(np.arange(len(r_minus))+1, r_minus, 's--', color=minus_color, label = mice, alpha=alphas[i])
        # Plot intercept
        ax.plot(0, intercept, 'o-', color=intercept_color,label = mice, alpha=alphas[i])
    
    # Add significance markers for lags (same as before)
    y_max = ax.get_ylim()[1]
    for lag in sorted(fisher_plus.keys()):
        p = fisher_plus[lag]
        if p < 0.001:
            ax.text(lag, y_max * 0.95, '***', ha='center', fontsize=30, color=plus_color)
        elif p < 0.01:
            ax.text(lag, y_max * 0.95, '**', ha='center', fontsize=30, color=plus_color)
        elif p < 0.05:
            ax.text(lag, y_max * 0.95, '*', ha='center', fontsize=30, color=plus_color)
        else:
            ax.text(lag, y_max * 0.95, 'ns', ha='center', fontsize=30, color=plus_color)
    
    for lag in sorted(fisher_minus.keys()):
        p = fisher_minus[lag]
        if p < 0.001:
            ax.text(lag, y_max * 0.90, '***', ha='center', fontsize=30, color=minus_color)
        elif p < 0.01:
            ax.text(lag, y_max * 0.90, '**', ha='center', fontsize=30, color=minus_color)
        elif p < 0.05:
            ax.text(lag, y_max * 0.90, '*', ha='center', fontsize=30, color=minus_color)
        else:
            ax.text(lag, y_max * 0.90, 'ns', ha='center', fontsize=30, color=minus_color)
    
    # Add intercept significance (if applicable)
    if intercept_pvalues:
        if fisher_intercept < 0.001:
            ax.text(0, y_max * 0.85, '***', ha='center', fontsize=25, color=intercept_color)
        elif fisher_intercept < 0.01:
            ax.text(0, y_max * 0.85, '**', ha='center', fontsize=25, color=intercept_color)
        elif fisher_intercept < 0.05:
            ax.text(0, y_max * 0.85, '*', ha='center', fontsize=25, color=intercept_color)
        else:
            ax.text(0, y_max * 0.85, 'ns', ha='center', fontsize=25, color=intercept_color)
    
    # Add reference line, labels, and legends
    ax.axhline(y=0, color='black', linestyle='--', linewidth=3, alpha=0.5)
    ax.set_title('GLM Weights with Fisher-Combined Significance', pad=20)
    ax.set_ylabel('GLM Weight', labelpad=20)
    ax.set_xlabel('Previous Trials', labelpad=20)
    ax.grid(True, linestyle=':', alpha=0.3)
    plt.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label=r'$\beta_{t}^+$', markerfacecolor=plus_color, markersize=10),
            Line2D([0], [0], marker='s', color='w', label=r'$\beta_{t}^-$', markerfacecolor=minus_color, markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Intercept', markerfacecolor=intercept_color, markersize=10)
        ],
        loc='best', fontsize=20, framealpha=0.5, edgecolor='black', facecolor='white', bbox_to_anchor=(0.95, 0.5),
    )
    # Make legend frame visible 
    legend = ax.get_legend()
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    plt.tight_layout()
    plt.show()

def glm(df):
    mice_counter = 0
    n_subjects = len(df['subject'].unique())
    avaluate = 0
    for j in [2,3,4,7,10]:
        j = 10
        plot_all_mice_glm_combined(df,n_back = j, figsize=(46.8, 33.1))
    if not avaluate:
        n_cols = int(np.ceil(n_subjects / 2))
        f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
        f1, axes1 = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
        exponents = np.zeros((n_subjects,2))
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
                mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_80).fit()
                GLM_df = pd.DataFrame({
                    'coefficient': mM_logit.params,
                    'std_err': mM_logit.bse,
                    'z_value': mM_logit.tvalues,
                    'p_value': mM_logit.pvalues,
                    'conf_Interval_Low': mM_logit.conf_int()[0],
                    'conf_Interval_High': mM_logit.conf_int()[1]
                })
                exponents[mice_counter] = exp_regressors(GLM_df)
                # subplot title with name of mouse
                ax = axes[mice_counter//n_cols, mice_counter%n_cols]
                ax1 = axes1[mice_counter//n_cols, mice_counter%n_cols]

                ax.set_title(f'GLM weights: {mice}')
                ax1.set_title(f'Psychometric Function: {mice}')
                plot_GLM(ax, GLM_df)
                #data_label can be either 'choice_num' or 'probability_r'
                psychometric_data(ax1, df_20, GLM_df, regressors_string,'choice_num')
                ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.set_xlabel('Evidence')
                ax1.set_ylabel('Prob of going right')
                ax1.legend(loc='upper left')
                mice_counter += 1
        plt.tight_layout()
        plt.show()
        print(exponents)
        print(sum(exponents)/9)
    else:
        unique_subjects = df['subject'][df['subject'] != 'A10'].unique()
        n_back_vect = [3,5,7,9]
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
    new_df = parsing(df,trained,0)
    glm(new_df)
    #performance(new_df)
