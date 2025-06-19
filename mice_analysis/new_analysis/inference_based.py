import pandas as pd
import numpy as np
import scipy
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)
import matplotlib.patches as mpatches
from extra_plotting import *
from parsing import *



def manual_computation(df: pd.DataFrame, n_back: int, hist: bool) -> pd.DataFrame:
    """
    Processes mouse choice behavior data to compute value differences and sequence patterns.
    
    Args:
        df: Input DataFrame containing trial-by-trial behavioral data
        n_back: Number of previous trials to consider for sequence patterns
        hist: Boolean flag to plot histogram of computed value differences
        
    Returns:
        Processed DataFrame with computed features including choice sequences and value differences
    """
    
    df = df.reset_index(drop=True)
    new_df = df.copy()
    
    # Encode choices (0/1) based on side and outcome
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'right'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'left'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'left'), 'choice'] = 1
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'right'), 'choice'] = 1
    
    # Create choice-reward codes ('00', '01', '10', '11')
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 0), 'choice_rwd'] = '00'    
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 1), 'choice_rwd'] = '01'  
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 0), 'choice_rwd'] = '10'  
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 1), 'choice_rwd'] = '11'  
    new_df['choice_rwd'] = new_df['choice_rwd'].fillna(' ')
    
    # Clean data and init columns
    new_df = new_df.dropna(subset=['probability_r']).reset_index(drop=True)
    new_df['sequence'] = ''
    
    # Mark active side (higher probability)
    new_df['right_active'] = (new_df['probability_r'] > 0.5).astype(int)
    new_df['left_active'] = (new_df['probability_r'] < 0.5).astype(int)
    
    # Build n-back sequences
    for i in range(n_back):
        new_df[f'choice_rwd{i+1}'] = new_df.groupby('session')['choice_rwd'].shift(i+1)
        new_df['sequence'] += new_df[f'choice_rwd{i+1}']
    
    new_df = new_df.dropna(subset=['sequence']).reset_index(drop=True)
    
    # Compute value difference (V_t)
    new_df['prob_right'] = new_df.groupby('sequence')['right_active'].transform('mean')
    new_df['prob_left'] = new_df.groupby('sequence')['left_active'].transform('mean')
    new_df['V_t'] = new_df['prob_right'] - new_df['prob_left']
    
    # Optional histogram
    if hist:
        print(new_df['V_t'].value_counts())
        plt.hist(new_df['V_t'], bins=100)
        plt.show()
    
    # Prepare next trial data
    new_df = new_df.dropna(subset=['choice'])
    new_df['choice_1'] = new_df.groupby('session')['choice'].shift(-1)
    new_df['side_num'] = new_df['choice_1'].map({0: -1, 1: 1})
    
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
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing behavioral data for all mice
    n_back : int
        Number of previous trials to consider for inference calculation
    figsize : tuple, optional
        Figure size in inches (default is A0 poster size 46.8x33.1)
        
    Returns:
    --------
    None (displays plot)
    """
    
    # Custom color scheme for different coefficient types
    beta_color = '#d62728'  # Red for beta (V_t - value coefficient)
    side_color = '#1f77b4'  # Blue for side bias coefficient
    intercept_color = '#2ca02c'  # Green for intercept
    
    # Set global styling parameters for poster presentation
    plt.rcParams.update({
        'axes.titlesize': 50,        # Large title size
        'axes.labelsize': 50,         # Large axis labels
        'xtick.labelsize': 35,       # Large x-tick labels
        'ytick.labelsize': 35,       # Large y-tick labels
        'legend.fontsize': 25,       # Large legend font
        'lines.linewidth': 4,        # Thick lines
        'lines.markersize': 15       # Large markers
    })
    
    # Create figure with specified size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get list of mice, excluding A10
    mice_list = [m for m in df['subject'].unique() if m != 'A10']
    n_mice = len(mice_list)
    
    # Create alpha gradient for mice (lighter to darker)
    alphas = np.linspace(0.3, 1, n_mice)
    
    # Initialize lists to store p-values and data across CV splits
    plus_pvalues_per_lag = []
    minus_pvalues_per_lag = []
    intercept_pvalues = []  
    GLM_data = []
    all_metrics = []
    v2 = 0  # Version flag for different computation methods
    
    # Process each mouse's data
    for i, mice in enumerate(mice_list):
        df_mice = df.loc[df['subject'] == mice]
        
        # Determine if using history (for n_back > 5)
        hist = False
        if n_back >5: 
            hist = True
            
        # Compute inference values (two possible methods)
        if v2==1:
            df_values_new = manual_computation_v2(df_mice, p_SW=0.01, p_RWD=0.8,hist=hist)
        else:
            df_values_new = manual_computation(df_mice, n_back= n_back,hist=hist)
            
        # Select training sessions using cross-validation
        df_cv = select_train_sessions(df_values_new)
        
        # Initialize CV lists for this mouse
        cv_plus_pvalues_per_lag = []
        cv_minus_pvalues_per_lag = []
        cv_intercept_pvalues = []

        # 5-fold cross-validation
        for i in range(5):
            df_80 = df_cv[df_cv[f'split_{i}'] == f'train']
            df_test = df_cv[df_cv[f'split_{i}'] == f'test']
            
            try:
                # Fit logistic regression model (two possible versions)
                if v2==2:
                    mM_logit = smf.logit(formula='choice ~ side_num', data=df_80).fit()
                else:
                    mM_logit = smf.logit(formula='choice ~ V_t + side_num', data=df_80).fit()
            except Exception as e:
                print(f"Model fitting failed for {mice}: {str(e)}")
                continue  # Skip this mouse if model fails

            # Store GLM results in DataFrame
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
            
            # Make predictions on test set
            df_test['pred_prob'] = mM_logit.predict(df_test)
            n_regressors = 2
            
            # Calculate evaluation metrics
            y_true = (
                df_test.groupby('session')['choice']
                .apply(lambda x: x.iloc[n_regressors:-1])
                .reset_index(drop=True)
            )
            
            # Get predictions per session
            predictions = []
            for session, group in df_test.groupby('session'):
                session_pred = mM_logit.predict(group[:-1])[n_regressors:]
                if(session_pred.isna().any()):
                    print(np.where(session_pred.isna()))
                predictions.append(session_pred)
                
            y_pred_prob = pd.concat(predictions)
            y_pred_class = (y_pred_prob >= 0.5).astype(int)
            np.random.seed(42) 
            y_pred_class_mult = (np.random.rand(len(y_pred_prob)) < y_pred_prob).astype(int)

            # Comprehensive metrics dictionary
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
            
            # Store p-values by regressor type
            for i, reg in enumerate(mM_logit.params.index):
                if 'V' in reg:
                    cv_plus_pvalues_per_lag.append(mM_logit.pvalues[i])
                elif 'side' in reg:
                    cv_minus_pvalues_per_lag.append(mM_logit.pvalues[i])
                elif reg == 'Intercept':
                    cv_intercept_pvalues.append(mM_logit.pvalues[i])
                    
        # Store median p-values across CV splits for this mouse
        plus_pvalues_per_lag.append(np.median(cv_plus_pvalues_per_lag))
        minus_pvalues_per_lag.append(np.median(cv_minus_pvalues_per_lag))
        intercept_pvalues.append(np.median(cv_intercept_pvalues))  

    # Save metrics to CSV (handling different versions)
    if v2==1:
        file_name = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_inference_based_v2_{n_back}.csv'
    elif v2==2:
        file_name = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_model_0_{n_back}.csv'
    else:
        file_name = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_inference_based_{n_back}.csv'
        
    # Remove existing file if present
    if os.path.exists(file_name):
        os.remove(file_name)
        
    # Save combined metrics
    combined_glm_metrics = file_name
    combined_metrics = pd.concat(all_metrics,ignore_index=True,axis=0)
    combined_metrics.to_csv(combined_glm_metrics, index=False)
    
    # Process GLM data
    df_GLM_data = pd.concat(GLM_data)
    df_GLM_data = df_GLM_data.reset_index(drop=True)
    df_GLM_df = df_GLM_data.groupby(['regressor','subject'])['coefficient'].mean().reset_index()
    
    # Combine p-values across mice using Stouffer's method
    fisher_plus = scipy.stats.combine_pvalues(plus_pvalues_per_lag, method='stouffer')[1]
    fisher_minus = scipy.stats.combine_pvalues(minus_pvalues_per_lag, method='stouffer')[1]
    
    # Combine intercept p-values if available
    if intercept_pvalues:
        fisher_intercept = scipy.stats.combine_pvalues(intercept_pvalues, method='stouffer')[1]
    
    # Plot coefficients with significance markers
    i = 1
    y_max = ax.get_ylim()[1]
    
    # Plot beta coefficient (if not model 0)
    if v2 != 2:
        beta = df_GLM_df.loc[df_GLM_df['regressor'].str.contains('V_t'), 'coefficient'].values[0]
        ax.bar(i, beta, width=0.66, color=beta_color, alpha=1, label=f'β' if i == 0 else "")
        
        # Add significance markers
        p = fisher_plus
        if p < 0.001:
            ax.text(i, y_max * 0.95, '***', ha='center', fontsize=30, color=beta_color)
        elif p < 0.01:
            ax.text(i, y_max * 0.95, '**', ha='center', fontsize=30, color=beta_color)
        elif p < 0.05:
            ax.text(i, y_max * 0.95, '*', ha='center', fontsize=30, color=beta_color)
        else:
            ax.text(i, y_max * 0.95, 'ns', ha='center', fontsize=30, color=beta_color)
    
    # Plot side bias coefficient
    side = df_GLM_df.loc[df_GLM_df['regressor'].str.contains('side_num'), 'coefficient'].values[0]
    intercept = df_GLM_df[(df_GLM_df['regressor'] == 'Intercept')]['coefficient'].values
    ax.bar(i+0.67, side, width=0.66, color=side_color, alpha=1, label=f' side' if i == 0 else "")
    
    # Plot intercept if available
    if intercept.size > 0:
        ax.bar(0.33, intercept[0], width=0.66, color=intercept_color, alpha=1, label=f'{mice} Intercept' if i == 0 else "")
    
    # Add significance markers for side bias
    p = fisher_minus
    if p < 0.001:
        ax.text(i + 0.67, y_max * 0.90, '***', ha='center', fontsize=30, color=side_color)
    elif p < 0.01:
        ax.text(i + 0.67, y_max * 0.90, '**', ha='center', fontsize=30, color=side_color)
    elif p < 0.05:
        ax.text(i + 0.67, y_max * 0.90, '*', ha='center', fontsize=30, color=side_color)
    else:
        ax.text(i + 0.67, y_max * 0.90, 'ns', ha='center', fontsize=30, color=side_color)
    
    # Add intercept significance if available
    if intercept_pvalues:
        if fisher_intercept < 0.001:
            ax.text(0.33, y_max * 0.85, '***', ha='center', fontsize=25, color=intercept_color)
        elif fisher_intercept < 0.01:
            ax.text(0.33, y_max * 0.85, '**', ha='center', fontsize=25, color=intercept_color)
        elif fisher_intercept < 0.05:
            ax.text(0.33, y_max * 0.85, '*', ha='center', fontsize=25, color=intercept_color)
        else:
            ax.text(0.33, y_max * 0.85, 'ns', ha='center', fontsize=25, color=intercept_color)
    
    # Add reference line and grid
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.grid(True, axis='y', linestyle=':', alpha=0.3)
    ax.set_xticks([])  # Remove x-ticks
    
    # Create legends
    # Legend 1: Coefficient types
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
    
    # Add the first legend back (since adding second would remove it)
    ax.add_artist(legend1)
    
    # Adjust layout for better spacing
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(right=0.75, bottom=0.2)  # Make space for legends and x-labels
    plt.show()

def inference_plot(df):
    """
    Plot inference model results either combined across all mice or separately for each mouse.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing behavioral data for all mice
        
    Returns:
    --------
    None (displays plot)
    """
    
    # Flag to control whether to plot mice separately or combined
    separate_mice = True
    
    # Combined plot for all mice
    if not separate_mice:
        # Loop through possible n_back values (though j is hardcoded to 3)
        for j in [1,2,3,4,5]:
            j = 3  # Hardcoded override - always uses n_back=3
            # Call plotting function with A0 poster size
            plot_all_mice_correct_inf_combined(df, n_back=j, figsize=(46.8, 33.1))
        
    # Individual plots for each mouse
    if separate_mice:
        # Calculate subplot layout (2 rows x n_cols columns)
        n_subjects = len(df['subject'].unique())
        n_cols = int(np.ceil(n_subjects / 2))
        
        # Create figure with dynamic sizing
        f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
        mice_counter = 0  # Counter to track mouse position in subplot grid
        
        # Color scheme for different regressors
        regressor_colors = {
            'Intercept': '#1f77b4',  # Blue
            'side_num': '#ff7f0e',   # Orange
            'V_t': '#2ca02c',        # Green
        }
        
        # Storage for aggregated results
        all_results = []
        
        # Process each mouse
        for mice in df['subject'].unique():
            if mice != 'A10':  # Exclude specific mouse if needed
                # Filter and preprocess data for current mouse
                df_mice = df.loc[df['subject'] == mice]
                
                # Only keep sessions with >50 trials
                session_counts = df_mice['session'].value_counts()
                mask = df_mice['session'].isin(session_counts[session_counts > 50].index)
                df_mice['sign_session'] = 0
                df_mice.loc[mask, 'sign_session'] = 1
                new_df_mice = df_mice[df_mice['sign_session'] == 1]
                print(f"Processing mouse: {mice}")
                
                # Set n_back value (could be parameterized)
                n_back = 3  
                
                # Compute inference values and select training sessions
                df_values_new = manual_computation(new_df_mice, n_back, hist=False)
                df_cv = select_train_sessions(df_values_new)
                
                # Get current subplot axis
                ax = axes[mice_counter//n_cols, mice_counter%n_cols]
                
                GLM_data = []  # Store GLM results across CV splits
                
                # 5-fold cross-validation
                for i in range(5):  
                    # Split data
                    df_80 = df_cv[df_cv[f'split_{i}'] == f'train']
                    df_test = df_cv[df_cv[f'split_{i}'] == f'test']
                    
                    # Fit logistic regression model
                    mM_logit = smf.logit(formula='choice ~ V_t + side_num', data=df_80).fit(disp=0)
                    
                    # Store model results
                    GLM_df = pd.DataFrame({
                        'coefficient': mM_logit.params,
                        'std_err': mM_logit.bse,
                        'z_value': mM_logit.tvalues,
                        'p_value': mM_logit.pvalues,
                        'conf_interval_low': mM_logit.conf_int()[0],
                        'conf_interval_high': mM_logit.conf_int()[1],
                    })
                    GLM_df['subject'] = mice
                    GLM_df['split'] = i
                    GLM_df = GLM_df.reset_index().rename(columns={'index': 'regressor'})
                    GLM_data.append(GLM_df)
                    
                    # Store predictions
                    df_test['pred_prob'] = mM_logit.predict(df_test)
                
                # Combine results across all CV splits
                df_GLM_data = pd.concat(GLM_data).reset_index(drop=True)
                
                # Calculate median coefficients and p-values across splits
                median_results = df_GLM_data.groupby('regressor').agg({
                    'coefficient': 'median',
                    'p_value': 'median'
                }).reset_index()
                
                # Set subplot title and reference line
                ax.set_title(f'Mouse {mice}', fontsize=12)
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                
                # Get y-axis limits for significance marker placement
                y_min, y_max = ax.get_ylim()
                sig_height = y_max * 0.95  # Position for significance markers
                
                # Plot each regressor's coefficient
                for i, (_, row) in enumerate(median_results.iterrows()):
                    reg = row['regressor']
                    coef = row['coefficient']
                    pval = row['p_value']
                    
                    # Get regressor-specific color (default to gray)
                    color = regressor_colors.get(reg, '#7f7f7f')
                    
                    # Plot coefficient point
                    ax.plot(i, coef, 'o', markersize=8, color=color, alpha=0.8)
                    
                    # Add significance marker
                    if pval < 0.001:
                        ax.text(i, sig_height, '***', ha='center', fontsize=20, color=color)
                    elif pval < 0.01:
                        ax.text(i, sig_height, '**', ha='center', fontsize=20, color=color)
                    elif pval < 0.05:
                        ax.text(i, sig_height, '*', ha='center', fontsize=20, color=color)
                    else:
                        ax.text(i, sig_height, 'ns', ha='center', fontsize=15, color='gray')
                    
                    # Store results for population analysis
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
                ax.set_xticklabels(
                    median_results['regressor'], 
                    rotation=45 if len(median_results) > 3 else 0
                )
                
                mice_counter += 1  # Move to next subplot position
    
    # Adjust layout and display plot
    plt.tight_layout()
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
    opto_yes = 0
    new_df = parsing(new_df, trained,opto_yes)
    inference_plot(new_df)
