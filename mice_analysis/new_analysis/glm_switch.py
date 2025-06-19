import pandas as pd
import numpy as np
import scipy
from typing import Tuple
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
import os
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss,
)


from extra_plotting import *
from parsing import *


def obt_regressors(df,n) -> Tuple[pd.DataFrame, str]:
    """
    Summary:
    This function processes the data needed to obtain the regressors and derives the 
    formula for the glm.

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

    #prepare the switch regressor
    new_df['choice_1'] = new_df.groupby('session')['choice'].shift(1)
    new_df.loc[(new_df['choice'] == new_df['choice_1']), 'switch_num'] = 0
    new_df.loc[(new_df['choice'] != new_df['choice_1']), 'switch_num'] = 1


    # Last trial reward
    new_df['last_trial'] = new_df.groupby('session')['outcome_bool'].shift(1)

    # build the regressors for previous trials
    rss_plus = ''
    rss_minus = ''
    rds_plus = ''
    rds_minus = ''
    for i in range(2, n + 1):
        new_df[f'choice_{i}'] = new_df.groupby('session')['choice'].shift(i)
        new_df[f'outcome_bool_{i}'] = new_df.groupby('session')['outcome_bool'].shift(i)
        
        #prepare the data for the error_switch regressor rss_-
        new_df.loc[(new_df[f'choice_{i}'] == new_df['choice_1']) & (new_df[f'outcome_bool_{i}'] == 0), f'rss_minus{i}'] = 1
        new_df.loc[(new_df[f'choice_{i}'] == new_df['choice_1']) & (new_df[f'outcome_bool_{i}'] == 1), f'rss_minus{i}'] = 0
        new_df.loc[new_df[f'choice_{i}'] != new_df['choice_1'], f'rss_minus{i}'] = 0
        new_df[f'rss_minus{i}'] = pd.to_numeric(new_df[f'rss_minus{i}'], errors='coerce')
        #prepare the data for the error_switch regressor rss_-
        new_df.loc[(new_df[f'choice_{i}'] == new_df['choice_1']) & (new_df[f'outcome_bool_{i}'] == 1), f'rss_plus{i}'] = 1
        new_df.loc[(new_df[f'choice_{i}'] == new_df['choice_1']) & (new_df[f'outcome_bool_{i}'] == 0), f'rss_plus{i}'] = 0
        new_df.loc[new_df[f'choice_{i}'] != new_df['choice_1'], f'rss_plus{i}'] = 0
        new_df[f'rss_plus{i}'] = pd.to_numeric(new_df[f'rss_plus{i}'], errors='coerce')
        rss_plus += f'rss_plus{i} + '
        rss_minus += f'rss_minus{i} + '
        #prepare the data for the error_switch regressor rds_-
        # new_df.loc[(new_df[f'choice_{i}'] != new_df['choice_1']) & (new_df[f'outcome_bool_{i}'] == 0), f'rds_minus{i}'] = 1
        # new_df.loc[(new_df[f'choice_{i}'] != new_df['choice_1']) & (new_df[f'outcome_bool_{i}'] == 1), f'rds_minus{i}'] = 0
        # new_df.loc[new_df[f'choice_{i}'] == new_df['choice_1'], f'rds_minus{i}'] = 0
        # new_df[f'rss_minus{i}'] = pd.to_numeric(new_df[f'rss_minus{i}'], errors='coerce')

        #prepare the data for the error_switch regressor rds_+
        new_df.loc[(new_df[f'choice_{i}'] != new_df['choice_1']) & (new_df[f'outcome_bool_{i}'] == 1), f'rds_plus{i}'] = 1
        new_df.loc[(new_df[f'choice_{i}'] != new_df['choice_1']) & (new_df[f'outcome_bool_{i}'] == 0), f'rds_plus{i}'] = 0
        new_df.loc[new_df[f'choice_{i}'] == new_df['choice_1'], f'rds_plus{i}'] = 0
        new_df[f'rss_plus{i}'] = pd.to_numeric(new_df[f'rss_plus{i}'], errors='coerce')
        rds_plus += f'rds_plus{i} + '
    regressors_string = rss_plus + rss_minus + rds_plus + 'last_trial'
    new_df = new_df.copy()

    return new_df, regressors_string


def plot_all_mice_glm_combined(df, n_back, figsize=(46.8, 33.1)):
    """
    Plot ALL mice's GLM weights with significance testing using Fisher's method,
    including the intercept's combined p-value.
    
    Parameters:
    - df: DataFrame containing the data
    - n_back: Number of previous trials to consider
    - figsize: Tuple specifying figure dimensions
    """
    
    # Set plot styling parameters
    plt.rcParams.update({
        'axes.titlesize': 50,
        'axes.labelsize': 50,
        'xtick.labelsize': 35,
        'ytick.labelsize': 35,
        'legend.fontsize': 25,
        'lines.linewidth': 4,
        'lines.markersize': 15
    })

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique mouse IDs, excluding 'A10'
    mice_list = np.unique(df['subject'])
    mice_list = mice_list[mice_list!= 'A10']
    
    # Create alpha values for fading effect across mice
    alphas = np.linspace(0.3, 1, len(mice_list))
    
    # Define colors for different regressor types
    rssp_color = '#d62728'  # Red for RSS+
    rds_color = '#ff7f0e'   # Orange for RDS+
    rssm_color = '#1f77b4'  # Blue for RSS-
    neutral_color = '#7f7f7f'  # Gray for neutral
    intercept_color = '#2ca02c'  # Green for intercept

    # Initialize dictionaries to store p-values for each lag
    rssp_pvalues_per_lag = defaultdict(list)
    rssm_pvalues_per_lag = defaultdict(list)
    rdsp_pvalues_per_lag = defaultdict(list)
    lt_pvalues = []  # For last trial p-values
    intercept_pvalues = []  # Store intercept p-values
    
    # Initialize lists to store GLM data and metrics
    GLM_data = []
    all_metrics = []

    # Process each mouse's data
    for mice in mice_list:
        # Filter data for current mouse
        df_mice = df.loc[df['subject'] == mice]
        
        # Get regressors and prepare GLM data
        df_glm_mice, regressors_string = obt_regressors(df=df_mice, n=n_back)
        
        # Implement 5-fold cross-validation
        df_cv = select_train_sessions(df_glm_mice)
        
        # Initialize storage for CV p-values
        cv_rssp_pvalues_per_lag = defaultdict(list)
        cv_rssm_pvalues_per_lag = defaultdict(list)
        cv_rdsp_pvalues_per_lag = defaultdict(list)
        cv_intercept_pvalues = []
        cv_lt = []

        # Perform cross-validation
        for i in range(5):
            # Split into train and test sets
            df_80 = df_cv[df_cv[f'split_{i}'] == f'train']
            df_test = df_cv[df_cv[f'split_{i}'] == f'test']
            
            # Fit GLM model
            mM_logit = smf.logit(formula='switch_num ~ ' + regressors_string, data=df_80).fit()
            
            # Store GLM coefficients and p-values
            GLM_df = pd.DataFrame({
                'coefficient': mM_logit.params,
                'p_value': mM_logit.pvalues
            })
            GLM_df['subject'] = mice
            GLM_df['split'] = i
            df_reset = GLM_df.reset_index()
            df_reset = df_reset.rename(columns={'index': 'regressor'})
            GLM_data.append(df_reset)
            
            # Make predictions on test set
            df_test['pred_prob'] = mM_logit.predict(df_test)
            n_regressors = len([x.strip() for x in regressors_string.split(' + ')])
            
            # Prepare true labels and predictions for evaluation
            y_true = (
                df_test.groupby('session')['switch_num']
                .apply(lambda x: x.iloc[n_regressors:])
                .reset_index(drop=True)
            )
            
            # Get session-wise predictions
            predictions = []
            for session, group in df_test.groupby('session'):
                session_pred = mM_logit.predict(group)[n_regressors:]
                predictions.append(session_pred)
                
            y_pred_prob = pd.concat(predictions)
            y_pred_class = (y_pred_prob >= 0.5).astype(int)
            np.random.seed(42) 
            y_pred_class_mult = (np.random.rand(len(y_pred_prob)) < y_pred_prob).astype(int)

            # Calculate various evaluation metrics
            metrics_dict = {
                # Log-likelihood
                "log_likelihood": mM_logit.llf,
                "log_likelihood_per_obs": mM_logit.llf / len(y_true),
                
                # Information criteria
                "AIC": mM_logit.aic,
                "BIC": mM_logit.bic,
                
                # Pseudo R-squared
                "pseudo_r2_mcfadden": mM_logit.prsquared,
                "pseudo_r2_cox_snell": 1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true)),
                "pseudo_r2_nagelkerke": (1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true))) / 
                                    (1 - np.exp(2 * mM_logit.llnull / len(y_true))),
                
                # Classification metrics
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
            
            # Store metrics
            GLM_metrics = pd.DataFrame([metrics_dict])
            GLM_metrics['subject'] = mice
            GLM_metrics['split'] = i
            all_metrics.append(GLM_metrics)
            
            # Extract p-values for each regressor type
            for i, reg in enumerate(mM_logit.params.index):
                if 'rss_plus' in reg:
                    lag = int(reg.split('plus')[-1])
                    cv_rssp_pvalues_per_lag[lag].append(mM_logit.pvalues[i])
                elif 'rss_minus' in reg:
                    lag = int(reg.split('minus')[-1])
                    cv_rssm_pvalues_per_lag[lag].append(mM_logit.pvalues[i])
                elif 'rds_plus' in reg:
                    lag = int(reg.split('plus')[-1])
                    cv_rdsp_pvalues_per_lag[lag].append(mM_logit.pvalues[i])
                elif reg == 'Intercept':
                    cv_intercept_pvalues.append(mM_logit.pvalues[i])
                elif reg == 'last_trial':
                    cv_lt.append(mM_logit.pvalues[i])

        # Combine p-values across CV folds using median
        for lag in cv_rssp_pvalues_per_lag:
            rssp_pvalues_per_lag[lag].append(np.median(cv_rssp_pvalues_per_lag[lag]))
        for lag in cv_rssm_pvalues_per_lag:
            rssm_pvalues_per_lag[lag].append(np.median(cv_rssm_pvalues_per_lag[lag]))
        for lag in cv_rdsp_pvalues_per_lag:
            rdsp_pvalues_per_lag[lag].append(np.median(cv_rdsp_pvalues_per_lag[lag]))
        intercept_pvalues.append(np.median(cv_intercept_pvalues))
        lt_pvalues.append(np.median(cv_lt))    

    # Save metrics to CSV (remove existing file first)
    if os.path.exists(f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_glm_prob_switch_{n_back}.csv'):
        os.remove(f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_glm_prob_switch_{n_back}.csv')
    
    combined_glm_metrics = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_glm_prob_switch_{n_back}.csv'
    combined_metrics = pd.concat(all_metrics,ignore_index=True,axis=0)
    combined_metrics.to_csv(combined_glm_metrics, index=False)
    
    # Process GLM data to get mean coefficients per regressor per subject
    df_GLM_data = pd.concat(GLM_data)
    df_GLM_data = df_GLM_data.reset_index(drop=True)
    df_GLM_df = df_GLM_data.groupby(['subject','regressor'])['coefficient'].mean().reset_index()
    
    # Combine p-values across mice using Fisher's method
    fisher_rssp = {}
    fisher_rssm = {}
    fisher_rdsp = {}
    for lag in rssp_pvalues_per_lag:
        fisher_rssp[lag] = scipy.stats.combine_pvalues(rssp_pvalues_per_lag[lag], method='fisher')[1]
    for lag in rssm_pvalues_per_lag:
        fisher_rssm[lag] = scipy.stats.combine_pvalues(rssm_pvalues_per_lag[lag], method='fisher')[1]
    for lag in rdsp_pvalues_per_lag:
        fisher_rdsp[lag] = scipy.stats.combine_pvalues(rdsp_pvalues_per_lag[lag], method='fisher')[1]
    
    # Combine intercept and last trial p-values
    if intercept_pvalues:
        fisher_intercept = scipy.stats.combine_pvalues(intercept_pvalues, method='fisher')[1]
    if lt_pvalues:
        fisher_lt = scipy.stats.combine_pvalues(lt_pvalues, method='fisher')[1]
    
    # Plot coefficients for each mouse
    for i, mice in enumerate(mice_list):
        # Get coefficients for each regressor type
        rss_plus = df_GLM_df[df_GLM_df['regressor'].str.contains('rss_plus') & (df_GLM_df['subject'] == mice)]['coefficient'].values
        rds_plus = df_GLM_df[df_GLM_df['regressor'].str.contains('rds_plus') & (df_GLM_df['subject'] == mice)]['coefficient'].values
        rss_minus = df_GLM_df[df_GLM_df['regressor'].str.contains('rss_minus') & (df_GLM_df['subject'] == mice)]['coefficient'].values
        
        # Reorder coefficients (move first to last)
        rss_plus = np.append(rss_plus[1:], rss_plus[0])
        rds_plus = np.append(rds_plus[1:], rds_plus[0])
        rss_minus = np.append(rss_minus[1:], rss_minus[0])

        # Get intercept and last trial coefficients
        intercept = df_GLM_df[(df_GLM_df['regressor'] == 'Intercept') & (df_GLM_df['subject'] == mice)]['coefficient'].values
        last_trial = df_GLM_df[(df_GLM_df['regressor'] == 'last_trial') & (df_GLM_df['subject'] == mice)]['coefficient'].values
        
        # Plot coefficients with appropriate markers and colors
        ax.plot(np.arange(len(rss_plus))+2, rss_plus, 'o-', color=rssp_color,label = mice, alpha=alphas[i])
        ax.plot(np.arange(len(rss_minus))+2, rss_minus, 's--', color=rssm_color, label = mice, alpha=alphas[i])
        ax.plot(np.arange(len(rds_plus))+2, rds_plus, 'o--', color=rds_color, label = mice, alpha=alphas[i])
        ax.plot(0, intercept, 'o-', color=intercept_color,label = mice, alpha=alphas[i])
        ax.plot(1, last_trial, 'o-', color=neutral_color,label = mice, alpha=alphas[i])  

    # Add significance markers for each lag
    y_max = ax.get_ylim()[1]
    for lag in sorted(fisher_rssp.keys()):
        p = fisher_rssp[lag]
        if p < 0.001:
            ax.text(lag, y_max * 0.95, '***', ha='center', fontsize=30, color=rssp_color)
        elif p < 0.01:
            ax.text(lag, y_max * 0.95, '**', ha='center', fontsize=30, color=rssp_color)
        elif p < 0.05:
            ax.text(lag, y_max * 0.95, '*', ha='center', fontsize=30, color=rssp_color)
        else:
            ax.text(lag, y_max * 0.95, 'ns', ha='center', fontsize=30, color=rssp_color)
    
    # Repeat for other regressor types
    for lag in sorted(fisher_rssm.keys()):
        p = fisher_rssm[lag]
        if p < 0.001:
            ax.text(lag, y_max * 0.85, '***', ha='center', fontsize=30, color=rssm_color)
        elif p < 0.01:
            ax.text(lag, y_max * 0.85, '**', ha='center', fontsize=30, color=rssm_color)
        elif p < 0.05:
            ax.text(lag, y_max * 0.85, '*', ha='center', fontsize=30, color=rssm_color)
        else:
            ax.text(lag, y_max * 0.85, 'ns', ha='center', fontsize=30, color=rssm_color)

    for lag in sorted(fisher_rdsp.keys()):
        p = fisher_rdsp[lag]
        if p < 0.001:
            ax.text(lag, y_max * 0.75, '***', ha='center', fontsize=30, color=rds_color)
        elif p < 0.01:
            ax.text(lag, y_max * 0.75, '**', ha='center', fontsize=30, color=rds_color)
        elif p < 0.05:
            ax.text(lag, y_max * 0.75, '*', ha='center', fontsize=30, color=rds_color)
        else:
            ax.text(lag, y_max * 0.75, 'ns', ha='center', fontsize=30, color=rds_color)
    
    # Add significance markers for intercept and last trial
    if intercept_pvalues:
        if fisher_intercept < 0.001:
            ax.text(0, y_max * 0.80, '***', ha='center', fontsize=25, color=intercept_color)
        elif fisher_intercept < 0.01:
            ax.text(0, y_max * 0.80, '**', ha='center', fontsize=25, color=intercept_color)
        elif fisher_intercept < 0.05:
            ax.text(0, y_max * 0.80, '*', ha='center', fontsize=25, color=intercept_color)
        else:
            ax.text(0, y_max * 0.80, 'ns', ha='center', fontsize=25, color=intercept_color)
    if lt_pvalues:
        if fisher_lt < 0.001:
            ax.text(1, y_max * 0.75, '***', ha='center', fontsize=25, color=neutral_color)
        elif fisher_lt < 0.01:
            ax.text(1, y_max * 0.75, '**', ha='center', fontsize=25, color=neutral_color)
        elif fisher_lt < 0.05:
            ax.text(1, y_max * 0.75, '*', ha='center', fontsize=25, color=neutral_color)
        else:
            ax.text(1, y_max * 0.75, 'ns', ha='center', fontsize=25, color=neutral_color)   
    
    # Final plot formatting
    ax.axhline(y=0, color='black', linestyle='--', linewidth=3, alpha=0.5)
    ax.set_ylabel('GLM Weight', labelpad=20)
    ax.set_xlabel('Previous Trials', labelpad=20)
    ax.grid(True, linestyle=':', alpha=0.3)
    
    # Create custom legend
    plt.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label=r'$\beta_{t}^+$', markerfacecolor=rssp_color, markersize=10),
            Line2D([0], [0], marker='s', color='w', label=r'$\beta_{t}^-$', markerfacecolor=rssm_color, markersize=10),
            Line2D([0], [0], marker='o', color='w', label=r'$\alpha_{t}$', markerfacecolor=rds_color, markersize=10),
            Line2D([0], [0], marker='o', color='w', label=r'$\gamma$', markerfacecolor=neutral_color, markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Intercept', markerfacecolor=intercept_color, markersize=10)
        ],
        loc='lower right', fontsize=20, framealpha=0.5, edgecolor='black', facecolor='white'
    )
    
    # Format legend frame
    legend = ax.get_legend()
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    
    # Final adjustments and display
    plt.tight_layout()
    plt.show()

def glm(df):
    """
    Plot inference model results either combined across all mice or separately for each mouse.
    Performs logistic regression with cross-validation for each mouse and visualizes coefficients.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing behavioral data for all mice with columns:
        - subject: mouse identifier
        - session: session identifier
        - switch_num: dependent variable (binary)
        - other regressor columns
        
    Returns:
    --------
    None (displays plot with subplots for each mouse)
    """
    # Calculate subplot layout (2 rows x n_cols columns)
    n_subjects = len(df['subject'].unique())
    n_cols = int(np.ceil(n_subjects / 2))  # Ensure enough columns for all mice
    
    # Create figure with dynamic sizing based on number of mice
    f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
    mice_counter = 0  # Counter to track mouse position in subplot grid
    
    # Color scheme for different regressor types
    regressor_colors = {
        'Intercept': '#2ca02c',  # Green for intercept
        'rss_plus': '#d62728',   # Red for RSS+ 
        'rss_minus': '#1f77b4',  # Blue for RSS-
        'rds_plus': '#ff7f0e',   # Orange for RDS+
    }
    
    # Storage for aggregated results across all mice
    all_results = []
    
    # Process each mouse individually
    for mice in df['subject'].unique():
        if mice != 'A10':  # Exclude specific mouse if needed
            # Filter data for current mouse
            df_mice = df.loc[df['subject'] == mice]
            
            # Filter sessions - only keep those with >50 trials
            session_counts = df_mice['session'].value_counts()
            mask = df_mice['session'].isin(session_counts[session_counts > 50].index)
            df_mice['sign_session'] = 0  # Initialize flag column
            df_mice.loc[mask, 'sign_session'] = 1  # Mark valid sessions
            new_df_mice = df_mice[df_mice['sign_session'] == 1]  # Filter to valid sessions
            print(f"Processing mouse: {mice}")
            
            # Set number of previous trials to consider (could be parameterized)
            n_back = 10  # Using 10 previous trials for analysis
            
            # Compute inference values and select training sessions
            df_values_new, regressor_string = obt_regressors(new_df_mice, n_back)
            df_cv = select_train_sessions(df_values_new)
            
            # Get current subplot axis based on mouse counter
            ax = axes[mice_counter//n_cols, mice_counter%n_cols]
            
            GLM_data = []  # Store GLM results across all CV splits
            
            # Perform 5-fold cross-validation
            for i in range(5):  
                # Split data into train (80%) and test (20%) sets
                df_80 = df_cv[df_cv[f'split_{i}'] == f'train']
                df_test = df_cv[df_cv[f'split_{i}'] == f'test']
                
                # Fit logistic regression model
                mM_logit = smf.logit(formula='switch_num ~ ' + regressor_string, 
                                    data=df_80).fit(disp=0)  # Suppress output
                
                # Store model results in DataFrame
                GLM_df = pd.DataFrame({
                    'coefficient': mM_logit.params,        # Regression coefficients
                    'std_err': mM_logit.bse,              # Standard errors
                    'z_value': mM_logit.tvalues,          # Z-scores
                    'p_value': mM_logit.pvalues,          # P-values
                    'conf_interval_low': mM_logit.conf_int()[0],  # CI lower bound
                    'conf_interval_high': mM_logit.conf_int()[1], # CI upper bound
                })
                # Add metadata
                GLM_df['subject'] = mice
                GLM_df['split'] = i
                GLM_df = GLM_df.reset_index().rename(columns={'index': 'regressor'})
                GLM_data.append(GLM_df)
                
                # Store predictions on test set
                df_test['pred_prob'] = mM_logit.predict(df_test)
            
            # Combine results across all CV splits for this mouse
            df_GLM_data = pd.concat(GLM_data).reset_index(drop=True)
            
            # Calculate median coefficients and p-values across splits (robust to outliers)
            median_results = df_GLM_data.groupby('regressor').agg({
                'coefficient': 'median',
                'p_value': 'median'
            }).reset_index()
            
            # Set subplot title and reference line at y=0
            ax.set_title(f'Mouse {mice}', fontsize=12)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            
            # Get y-axis limits for proper significance marker placement
            y_min, y_max = ax.get_ylim()
            sig_height = y_max * 0.95  # Position for significance markers
            
            # Initialize list for x-tick positions
            x_ticks_vect = []
            
            # Plot each regressor's coefficient
            for i, (_, row) in enumerate(median_results.iterrows()):
                reg = row['regressor']
                coef = row['coefficient']
                pval = row['p_value']
                
                # Determine color based on regressor type
                if 'rss_plus' in reg:
                    color = regressor_colors.get('rss_plus', '#d62728')  # Red
                elif 'rss_minus' in reg:
                    color = regressor_colors.get('rss_minus', '#1f77b4')  # Blue
                elif 'rds_plus' in reg:
                    color = regressor_colors.get('rds_plus', '#ff7f0e')  # Orange
                elif 'Intercept' in reg:
                    color = regressor_colors.get('Intercept', '#2ca02c')  # Green
                else:
                    color = '#7f7f7f'  # Gray for other regressors
                
                # Adjust x-position based on regressor type
                if '10' in reg:  # Special handling for lag 10
                    i += 8
                elif ('Intercept' in reg) or ('last_trial' in reg):
                    i = i  # Keep position for intercept and last trial
                else:
                    i -= 1  # Adjust position for other lags
                
                # Determine x-tick label based on regressor type
                if 'last_trial' in reg:
                    j = 1
                elif 'Intercept' in reg:
                    j = 0
                # Extract numeric lag value from regressor name
                elif 'rss_plus' in reg:
                    j = int(reg.split('plus')[-1])
                elif 'rss_minus' in reg:
                    j = int(reg.split('minus')[-1])
                elif 'rds_plus' in reg:
                    j = int(reg.split('plus')[-1])
                
                # Adjust x-tick positions for better visualization
                if j == 10:
                    j = 2  # Special case for lag 10
                elif j > 1:
                    j += 1  # Shift other lags
                
                x_ticks_vect.append(j)  # Store x-tick position
                
                # Plot coefficient as a point
                ax.plot(i, coef, 'o', markersize=6, color=color, alpha=0.8)
                
                # Add significance marker above point
                if pval < 0.001:
                    sig_marker = '***'
                elif pval < 0.01:
                    sig_marker = '**'
                elif pval < 0.05:
                    sig_marker = '*'
                else:
                    sig_marker = 'ns'
                
                ax.text(i, sig_height, sig_marker, ha='center', fontsize=10, color=color)
                
                # Store results for potential population-level analysis
                all_results.append({
                    'subject': mice,
                    'regressor': reg,
                    'coefficient': coef,
                    'p_value': pval
                })
            
            # Configure subplot labels and ticks
            ax.set_xlabel('Regressor', fontsize=10)
            ax.set_ylabel('Coefficient', fontsize=10)
            ax.set_xticks(range(len(median_results)))
            ax.set_xticklabels(x_ticks_vect)  # Use custom x-tick labels
            mice_counter += 1  # Move to next subplot position

    # Adjust layout to prevent overlapping and display plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials1.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # 1 for analisis of trained mice, 0 for untrained
    print(df['task'].unique())
    trained = 1
    new_df = parsing(df,trained,0)
    glm(new_df)
    # Flag to control whether to plot mice separately or combined
    separate_mice = True
    # Combined plot for all mice
    if not separate_mice:
        n_subjects = len(df['subject'].unique())
        for j in [2,3,4,7,10]:
            plot_all_mice_glm_combined(new_df,n_back = j, figsize=(46.8, 33.1))
        
    # Individual plots for each mouse
    if separate_mice:
        glm(new_df)