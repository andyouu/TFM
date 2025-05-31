import pandas as pd
import numpy as np
import scipy
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
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

def manual_computation(df: pd.DataFrame, n_back: int, hist:bool) -> pd.DataFrame:
    #A column with the choice of the mice will now be constructed
    df= df.reset_index(drop=True)
    new_df = df.copy()
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'right'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'left'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'left'), 'choice'] = 1
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'right'), 'choice'] = 1
    #Choide-reward will be created
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 0), 'choice_rwd'] = '00'    
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 1), 'choice_rwd'] = '01'  
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 0), 'choice_rwd'] = '10'  
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 1), 'choice_rwd'] = '11'  
    new_df['choice_rwd'] = new_df['choice_rwd'].fillna(' ')
    new_df = new_df.dropna(subset=['probability_r']).reset_index(drop=True)
    new_df['sequence'] = ''
    new_df['right_active'] = 0
    new_df['left_active'] = 0
    new_df.loc[(new_df['probability_r'] > 0.5), 'right_active'] = 1
    new_df.loc[(new_df['probability_r'] < 0.5), 'left_active'] = 1
        # Create shifted columns for n_back previous trials and build sequence string
    for i in range(n_back):
        new_df[f'choice_rwd{i+1}'] = new_df.groupby('session')['choice_rwd'].shift(i+1)
        new_df['sequence'] = new_df['sequence'] + new_df[f'choice_rwd{i+1}']
    
    # Remove rows with incomplete sequences
    new_df = new_df.dropna(subset=['sequence']).reset_index(drop=True)
    
    # Create active side indicators based on probability
    new_df['right_active'] = 0  # 1 if right side is more probable
    new_df['left_active'] = 0   # 1 if left side is more probable
    new_df.loc[(new_df['probability_r'] > 0.5), 'right_active'] = 1
    new_df.loc[(new_df['probability_r'] < 0.5), 'left_active'] = 1
    
    # Shift active indicators to align with next trial's outcome
    # new_df['right_outcome'] = new_df['right_active'].shift(-1)
    # new_df['left_outcome'] = new_df['left_active'].shift(-1)
    new_df['right_outcome'] = new_df['right_active']
    new_df['left_outcome'] = new_df['left_active']
    
    # Calculate probability of right/left outcomes for each sequence pattern
    new_df['prob_right'] = new_df.groupby('sequence')['right_outcome'].transform('mean')
    new_df['prob_left'] = new_df.groupby('sequence')['left_outcome'].transform('mean')
    
    # Compute value difference between right and left options
    new_df['V_t'] = (new_df['prob_right'] - new_df['prob_left'])
    #plot histogram of the V_t
    if hist :
        #count the different values and print them
        print(new_df['V_t'].value_counts())
        print(new_df['sequence'].value_counts())
        plt.hist(new_df['V_t'], bins=100)
        plt.show()

    new_df = new_df.dropna(subset=['choice']).reset_index(drop=True)
    new_df['choice_1'] = new_df.groupby('session')['choice'].shift(-1)
    new_df.loc[(new_df['choice_1'] == 0), 'side_num'] = -1
    new_df.loc[(new_df['choice_1'] == 1), 'side_num'] = 1
    #return all but last bc it contanis a nan
    return new_df
def manual_computation_v2(df: pd.DataFrame, p_SW: float, p_RWD: float, hist: bool) -> pd.DataFrame:
    """
    Compute V_t based on the recursive equations for R_t and the given parameters.
    
    Args:
        df: Input DataFrame containing trial data
        p_SW: Probability of switching from active to inactive state
        p_RWD: Probability of reward in active state
        p_RW0: Base probability parameter for V_t computation
        
    Returns:
        DataFrame with computed V_t values and intermediate calculations
    """
    
    # Reset index and create new working dataframe
    df = df.reset_index(drop=True)
    new_df = df.copy()
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'right'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'left'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'left'), 'choice'] = 1
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'right'), 'choice'] = 1

    # Prepare columns for side and choice tracking
    new_df['choice_1'] = new_df.groupby('session')['choice'].shift(-1)
    new_df['choice_2'] = new_df.groupby('session')['choice'].shift(-2)
    new_df.loc[(new_df['choice_1'] == 0), 'side_num'] = -1
    new_df.loc[(new_df['choice_1'] == 1), 'side_num'] = 1
    
    # Initialize variables for the recursive computation
    new_df['R_t'] = 0.0
    new_df['V_t'] = 0.0
    new_df['same_site'] = (new_df['choice'] == new_df['choice_1']).astype(int)
    new_df['same_site_2'] = (new_df['choice'] == new_df['choice_2']).astype(int)
    
    new_df.loc[0, 'same_site'] = 0  # First trial has no previous site
    
    # Compute rho parameter
    rho = 1 / ((1 - p_SW) * (1 - p_RWD))
    
    # Iterate through trials to compute R_t and V_t
    for t in range(len(new_df)):
        if t == 0 or new_df.at[t, 'choice_1'] is None:
            # First trial starts with R_t = 0
            new_df.at[t, 'R_t'] = 0.0
        else:
            if new_df.at[t, 'outcome_bool']:
                # Reward resets R_t to 0
                new_df.at[t, 'R_t'] = 0.0
            else:
                if new_df.at[t, 'same_site']:
                    # Same site: apply the recursive equation
                    prev_R = new_df.at[t-1, 'R_t']
                    new_df.at[t, 'R_t'] = rho * (prev_R + p_SW)
                else:
                    # This correspond to the unrewarded exploratory switches or returns to the side after that switch.
                    #if it returns to the same site after a switch, we apply the recursive equation (not needed, the following covers it)
                    # if t > 2 and new_df.at[t, 'same_site_2'] == 1:
                    #     prev_R = new_df.at[t-2, 'R_t']
                    #     new_df.at[t, 'R_t'] = rho * (prev_R + p_SW)
                    #for unsuccessful exploratory switches, we apply the equation with the previous R_t using a marker (R_t cannot be equal)
                    new_df.at[t, 'R_t'] = new_df.at[t-1, 'R_t']
        
        # Compute V_t from R_t
        R_t = new_df.at[t, 'R_t']
        if R_t == 0:
            new_df.at[t, 'V_t'] = new_df.at[t, 'side_num'] * p_RWD
        elif R_t == new_df.at[t-1, 'R_t']:
            #after exploratory switch, we use the previous R_t and return to the rewarding side
            new_df.at[t, 'V_t'] = - new_df.at[t, 'side_num'] * p_RWD
        else:
            new_df.at[t, 'V_t'] = p_RWD * (1 - 2 / (R_t**(-new_df.at[t, 'side_num']) + 1))
        if new_df.at[t, 'V_t']  > 1:
            print(f"Warning: V_t exceeds 1 at trial {t}: {new_df.at[t, 'V_t']}")
    #plot histogram of V_t
    if hist:
        # Count the different values and print them
        print(new_df['V_t'].value_counts())
        plt.hist(new_df['V_t'], bins=100)
        plt.show()
    
    return new_df[:-1]

def plot_all_mice_correct_inf_combined(df, n_back, figsize=(46.8, 33.1)):
    """
    Plot ALL mice's correct inference weights in a SINGLE figure with:
    - Custom colors for coefficient types
    - Mice differentiated by alpha levels
    - A0 poster sizing
    """
    # Set your custom colors
    beta_color = '#d62728'  # Red for beta (V_t)
    side_color = '#1f77b4'  # Blue for side bias
    intercept_color = '#2ca02c'  # Green for intercept
    
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
    plus_pvalues_per_lag = []
    minus_pvalues_per_lag = []
    intercept_pvalues = []  # Store intercept p-values
    GLM_data = []
    all_metrics = []
    v2 = 0
    # Plot each mouse's coefficients
    for i, mice in enumerate(mice_list):
        df_mice = df.loc[df['subject'] == mice]
        hist = False
        if n_back >5: hist = True
        #if precomputed priors
        #if inference like vertechi
        if v2==1:
            df_values_new = manual_computation_v2(df_mice, p_SW=0.01, p_RWD=0.8,hist=hist)
        else:
            df_values_new = manual_computation(df_mice, n_back= n_back,hist=hist)
        df_cv = select_train_sessions(df_values_new)
        cv_plus_pvalues_per_lag = []
        cv_minus_pvalues_per_lag = []
        cv_intercept_pvalues = []  # Store intercept p-values

        for i in range(5):
            df_80 = df_cv[df_cv[f'split_{i}'] == f'train']
            df_test = df_cv[df_cv[f'split_{i}'] == f'test']
            try:
                #model_0
                if v2==2:
                    mM_logit = smf.logit(formula='choice ~ side_num', data=df_80).fit()
                else:
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
            })
            GLM_df['subject'] = mice
            GLM_df['split'] = i
            df_reset = GLM_df.reset_index()
            df_reset = df_reset.rename(columns={'index': 'regressor'})
            GLM_data.append(df_reset)
            df_test['pred_prob'] = mM_logit.predict(df_test)
            n_regressors = 2
            #Create a DataFrame with the avaluation metrics
            y_true = (
                df_test.groupby('session')['choice']
                .apply(lambda x: x.iloc[n_regressors:-1])
                .reset_index(drop=True)  # Flatten to a single Series
            )  # True binary outcomes
            predictions = []
            for session, group in df_test.groupby('session'):
                # Get predictions for this session only
                session_pred = mM_logit.predict(group[:-1])[n_regressors:]
                if(session_pred.isna().any()):
                    print(np.where(session_pred.isna()))
                predictions.append(session_pred)
                
            y_pred_prob = pd.concat(predictions)  # Predicted probabilities (change this tot the test set)
            y_pred_class = (y_pred_prob >= 0.5).astype(int)
            np.random.seed(42) 
            y_pred_class_mult = (np.random.rand(len(y_pred_prob)) < y_pred_prob).astype(int) # We may use the multinomial here to choose with probability (sampling)

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
            for i, reg in enumerate(mM_logit.params.index):
                if 'V' in reg:
                    cv_plus_pvalues_per_lag.append(mM_logit.pvalues[i])
                elif 'side' in reg:
                    cv_minus_pvalues_per_lag.append(mM_logit.pvalues[i])
                elif reg == 'Intercept':
                    cv_intercept_pvalues.append(mM_logit.pvalues[i])
        plus_pvalues_per_lag.append(np.median(cv_plus_pvalues_per_lag))
        minus_pvalues_per_lag.append(np.median(cv_minus_pvalues_per_lag))
        intercept_pvalues.append(np.median(cv_intercept_pvalues))  

    #if the path exists, remove it
    if v2==1:
        file_name = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_inference_based_v2_{n_back}.csv'
    elif v2==2:
        file_name = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_model_0_{n_back}.csv'
    else:
        file_name = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_inference_based_{n_back}.csv'
    if os.path.exists(file_name):
        os.remove(file_name)
    # Save combined GLM metrics to CSV
    combined_glm_metrics = file_name
    combined_metrics = pd.concat(all_metrics,ignore_index=True,axis=0)
    combined_metrics.to_csv(combined_glm_metrics, index=False)
    df_GLM_data = pd.concat(GLM_data)
    df_GLM_data = df_GLM_data.reset_index(drop=True)
    df_GLM_df = df_GLM_data.groupby(['regressor','subject'])['coefficient'].mean().reset_index()
    df_GLM_data = pd.concat(GLM_data)
    df_GLM_data = df_GLM_data.reset_index(drop=True)
    df_GLM_df = df_GLM_data.groupby(['regressor'])['coefficient'].mean().reset_index()
    fisher_plus = scipy.stats.combine_pvalues(plus_pvalues_per_lag, method='stouffer')[1]
    fisher_minus = scipy.stats.combine_pvalues(minus_pvalues_per_lag, method='stouffer')[1]
    
    # Combine intercept p-values
    if intercept_pvalues:  # Only if intercepts were found
        fisher_intercept = scipy.stats.combine_pvalues(intercept_pvalues, method='stouffer')[1]
    
    i = 1
    y_max = ax.get_ylim()[1]
    if v2 != 2:
        beta = df_GLM_df.loc[df_GLM_df['regressor'].str.contains('V_t'), 'coefficient'].values[0]
        ax.bar(i, beta, width=0.66, color=beta_color, alpha=1, label=f'β' if i == 0 else "")
        p = fisher_plus
        if p < 0.001:
            ax.text(i, y_max * 0.95, '***', ha='center', fontsize=30, color=beta_color)
        elif p < 0.01:
            ax.text(i, y_max * 0.95, '**', ha='center', fontsize=30, color=beta_color)
        elif p < 0.05:
            ax.text(i, y_max * 0.95, '*', ha='center', fontsize=30, color=beta_color)
        else:
            ax.text(i, y_max * 0.95, 'ns', ha='center', fontsize=30, color=beta_color)
    side = df_GLM_df.loc[df_GLM_df['regressor'].str.contains('side_num'), 'coefficient'].values[0]
    intercept = df_GLM_df[(df_GLM_df['regressor'] == 'Intercept')]['coefficient'].values
    ax.bar(i+0.67, side, width=0.66, color=side_color, alpha=1, label=f' side' if i == 0 else "")
    # Add intercept bar
    if intercept.size > 0:
        ax.bar(0.33, intercept[0], width=0.66, color=intercept_color, alpha=1, label=f'{mice} Intercept' if i == 0 else "")
    

    p = fisher_minus
    if p < 0.001:
        ax.text(i + 0.67, y_max * 0.90, '***', ha='center', fontsize=30, color=side_color)
    elif p < 0.01:
        ax.text(i + 0.67, y_max * 0.90, '**', ha='center', fontsize=30, color=side_color)
    elif p < 0.05:
        ax.text(i + 0.67, y_max * 0.90, '*', ha='center', fontsize=30, color=side_color)
    else:
        ax.text(i + 0.67, y_max * 0.90, 'ns', ha='center', fontsize=30, color=side_color)
    
    # Add intercept significance (if applicable)
    if intercept_pvalues:
        if fisher_intercept < 0.001:
            ax.text(0.33, y_max * 0.85, '***', ha='center', fontsize=25, color=intercept_color)
        elif fisher_intercept < 0.01:
            ax.text(0.33, y_max * 0.85, '**', ha='center', fontsize=25, color=intercept_color)
        elif fisher_intercept < 0.05:
            ax.text(0.33, y_max * 0.85, '*', ha='center', fontsize=25, color=intercept_color)
        else:
            ax.text(0.33, y_max * 0.85, 'ns', ha='center', fontsize=25, color=intercept_color)
    # Add reference line and styling
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.grid(True, axis='y', linestyle=':', alpha=0.3)
    #put the x-axis in blank
    ax.set_xticks([])
    
    # Create simplified legends
    # Legend 1: Coefficient types with your colors
    coeff_handles = [
        mpatches.Patch(color=beta_color, label=r'$\beta^V$'),
        mpatches.Patch(color=side_color, label=r'$\beta^S$'),
        mpatches.Patch(color=intercept_color, label='Intercept')
    ]
    legend1 = ax.legend(handles=coeff_handles, title='Coefficient Types',
                       loc='upper left', bbox_to_anchor=(1.01, 1))
    
    # Legend 2: Mice with alpha gradient
    mice_handles = []
    for i, mice in enumerate(mice_list):
        mice_handles.append(mpatches.Patch(color='gray', alpha=alphas[i], label=mice))
    
    # Add the first legend back
    ax.add_artist(legend1)
    
    # Adjust layout
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(right=0.75, bottom=0.2)  # Make space for legends and x-labels
    plt.show()


def inference_plot(prob_switch,prob_rwd,df):
    mice_counter = 0
    for j in [1,2,3,4,5]:
        j = 3
        plot_all_mice_correct_inf_combined(df,n_back=j, figsize=(46.8, 33.1))
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
