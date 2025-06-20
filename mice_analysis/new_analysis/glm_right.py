import pandas as pd
import numpy as np
from typing import Tuple
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)
import statsmodels.formula.api as smf
import os
import matplotlib.patches as mpatches
import scipy
from extra_plotting import *
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

def plot_all_mice_glm_combined(df, n_back, figsize=(46.8, 33.1)):
    """
    Plot ALL mice's GLM weights with significance testing using Fisher's method,
    including the intercept's combined p-value.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing behavioral data for all mice with columns:
        - subject: mouse identifier
        - session: session identifier
        - choice_num: dependent variable (binary choice)
        - other regressor columns
        
    n_back : int
        Number of previous trials to consider in the GLM
        
    figsize : tuple, optional
        Figure dimensions (width, height) in inches (default is very large for visibility)
        
    Returns:
    --------
    None (displays plot)
    """
    
    # Define color scheme for different regressor types
    plus_color = '#d62728'  # Red for r_plus regressors
    minus_color = '#1f77b4'  # Blue for r_minus regressors
    intercept_color = 'green'  # Green for intercept
    
    # Configure plot styling parameters for large, publication-quality figure
    plt.rcParams.update({
        'axes.titlesize': 50,    # Large title size
        'axes.labelsize': 50,    # Large axis label size
        'xtick.labelsize': 35,   # Large x-tick labels
        'ytick.labelsize': 35,   # Large y-tick labels
        'legend.fontsize': 25,   # Large legend font
        'lines.linewidth': 4,    # Thick lines
        'lines.markersize': 15   # Large markers
    })

    # Create figure with specified size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get list of unique mice, excluding mouse 'A10'
    mice_list = np.unique(df['subject'])
    mice_list = mice_list[mice_list != 'A10']
    
    # Create alpha values for fading effect across mice (darker for later mice)
    alphas = np.linspace(0.3, 1, len(mice_list))
    
    # Initialize storage for p-values (one list per lag + intercept)
    plus_pvalues_per_lag = defaultdict(list)  # For r_plus regressors
    minus_pvalues_per_lag = defaultdict(list)  # For r_minus regressors
    intercept_pvalues = []  # For intercept p-values
    
    # Initialize lists to store GLM results and metrics
    GLM_data = []
    all_metrics = []
    
    # Process each mouse individually
    for mice in mice_list:
        # Filter data for current mouse
        df_mice = df.loc[df['subject'] == mice]
        
        # Get regressors and prepare GLM data
        df_glm_mice, regressors_string = obt_regressors(df=df_mice, n=n_back)
        
        # Implement 5-fold cross-validation
        df_cv = select_train_sessions(df_glm_mice)
        
        # Initialize storage for CV p-values
        cv_plus_pvalues_per_lag = defaultdict(list)
        cv_minus_pvalues_per_lag = defaultdict(list)
        cv_intercept_pvalues = []
        
        # Perform 5-fold cross-validation
        for i in range(5):
            # Split into training and test sets
            df_80 = df_cv[df_cv[f'split_{i}'] == f'train']
            df_test = df_cv[df_cv[f'split_{i}'] == f'test']
            
            # Fit logistic regression model
            mM_logit = smf.logit(formula='choice_num ~ ' + regressors_string, data=df_80).fit()
            
            # Store model coefficients and p-values
            GLM_df = pd.DataFrame({
                'coefficient': mM_logit.params,  # Regression coefficients
                'p_value': mM_logit.pvalues      # P-values
            })
            GLM_df['subject'] = mice
            GLM_df['split'] = i
            df_reset = GLM_df.reset_index()
            df_reset = df_reset.rename(columns={'index': 'regressor'})
            GLM_data.append(df_reset)
            
            # Make predictions on test set
            df_test['pred_prob'] = mM_logit.predict(df_test)
            n_regressors = len([x.strip() for x in regressors_string.split(' + ')])
            
            # Prepare true labels and predictions for evaluation metrics
            y_true = (
                df_test.groupby('session')['choice_num']
                .apply(lambda x: x.iloc[n_regressors:])  # Skip first n_back trials per session
                .reset_index(drop=True)  # Flatten to single Series
            )
            
            # Get session-wise predictions
            predictions = []
            for session, group in df_test.groupby('session'):
                session_pred = mM_logit.predict(group)[n_regressors:]  # Skip first n_back trials
                predictions.append(session_pred)
                
            y_pred_prob = pd.concat(predictions)  # Combined predicted probabilities
            y_pred_class = (y_pred_prob >= 0.5).astype(int)  # Binary predictions at 0.5 threshold
            
            # Alternative probabilistic predictions (multinomial sampling)
            np.random.seed(42)  # For reproducibility
            y_pred_class_mult = (np.random.rand(len(y_pred_prob)) < y_pred_prob).astype(int)
            
            # Debugging check for specific mouse and n_back
            if mice == 'B3' and n_back in [2,10] and i == 0:
                print('alert')
            
            # Calculate comprehensive evaluation metrics
            metrics_dict = {
                # Log-likelihood measures
                "log_likelihood": mM_logit.llf,
                "log_likelihood_per_obs": mM_logit.llf / len(y_true),
                
                # Information criteria
                "AIC": mM_logit.aic,
                "BIC": mM_logit.bic,
                
                # Pseudo R-squared variants
                "pseudo_r2_mcfadden": mM_logit.prsquared,  # McFadden's
                "pseudo_r2_cox_snell": 1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true)),
                "pseudo_r2_nagelkerke": (1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true))) / 
                                    (1 - np.exp(2 * mM_logit.llnull / len(y_true))),
                
                # Classification metrics (threshold=0.5)
                "accuracy": accuracy_score(y_true, y_pred_class),
                "precision": precision_score(y_true, y_pred_class),
                "recall": recall_score(y_true, y_pred_class),
                "f1_score": f1_score(y_true, y_pred_class),
                
                # Probabilistic classification metrics
                "accuracy_bis": accuracy_score(y_true, y_pred_class_mult),
                "precision_bis": precision_score(y_true, y_pred_class_mult),
                "recall_bis": recall_score(y_true, y_pred_class_mult),
                "f1_score_bis": f1_score(y_true, y_pred_class_mult),
                
                # Probability-based metrics
                "roc_auc": roc_auc_score(y_true, y_pred_prob),
                "brier_score": brier_score_loss(y_true, y_pred_prob),
            }
            
            # Store metrics with mouse and split info
            GLM_metrics = pd.DataFrame([metrics_dict])
            GLM_metrics['subject'] = mice
            GLM_metrics['split'] = i
            all_metrics.append(GLM_metrics)
            
            # Extract p-values for each regressor type
            for i, reg in enumerate(mM_logit.params.index):
                if 'r_plus' in reg:
                    lag = int(reg.split('_')[-1])  # Extract lag number
                    cv_plus_pvalues_per_lag[lag].append(mM_logit.pvalues[i])
                elif 'r_minus' in reg:
                    lag = int(reg.split('_')[-1])  # Extract lag number
                    cv_minus_pvalues_per_lag[lag].append(mM_logit.pvalues[i])
                elif reg == 'Intercept':
                    cv_intercept_pvalues.append(mM_logit.pvalues[i])
        
        # Combine p-values across CV folds using median (robust to outliers)
        for lag in cv_plus_pvalues_per_lag:
            plus_pvalues_per_lag[lag].append(np.median(cv_plus_pvalues_per_lag[lag]))
        for lag in cv_minus_pvalues_per_lag:
            minus_pvalues_per_lag[lag].append(np.median(cv_minus_pvalues_per_lag[lag]))
        intercept_pvalues.append(np.median(cv_intercept_pvalues))

    # Save combined metrics to CSV (remove existing file first)
    metrics_path = f'/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_glm_prob_r_{n_back}.csv'
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    
    # Combine all metrics and save to CSV
    combined_metrics = pd.concat(all_metrics, ignore_index=True, axis=0)
    combined_metrics.to_csv(metrics_path, index=False)
    
    # Process GLM coefficients across all mice and splits
    df_GLM_data = pd.concat(GLM_data)
    df_GLM_data = df_GLM_data.reset_index(drop=True)
    
    # Calculate mean coefficients per regressor per mouse
    df_GLM_df = df_GLM_data.groupby(['regressor','subject'])['coefficient'].mean().reset_index()
    
    # Combine p-values across mice using Fisher's method for each lag
    fisher_plus = {}  # For r_plus regressors
    fisher_minus = {}  # For r_minus regressors
    
    for lag in plus_pvalues_per_lag:
        fisher_plus[lag] = scipy.stats.combine_pvalues(plus_pvalues_per_lag[lag], method='fisher')[1]
    for lag in minus_pvalues_per_lag:
        fisher_minus[lag] = scipy.stats.combine_pvalues(minus_pvalues_per_lag[lag], method='fisher')[1]
    
    # Combine intercept p-values across mice
    if intercept_pvalues:  # Only if intercepts were found
        fisher_intercept = scipy.stats.combine_pvalues(intercept_pvalues, method='fisher')[1]
    
    # Plot coefficients for each mouse
    for i, mice in enumerate(mice_list):
        # Get coefficients for each regressor type
        r_plus = df_GLM_df[df_GLM_df['regressor'].str.contains('r_plus') & 
                          (df_GLM_df['subject'] == mice)]['coefficient'].values
        r_minus = df_GLM_df[df_GLM_df['regressor'].str.contains('r_minus') & 
                           (df_GLM_df['subject'] == mice)]['coefficient'].values
        intercept = df_GLM_df[(df_GLM_df['regressor'] == 'Intercept') & 
                            (df_GLM_df['subject'] == mice)]['coefficient'].values
        
        # Plot coefficients with appropriate markers and colors
        ax.plot(np.arange(len(r_plus))+1, r_plus, 'o-', color=plus_color, 
               label=mice, alpha=alphas[i])  # r_plus as circles with solid line
        ax.plot(np.arange(len(r_minus))+1, r_minus, 's--', color=minus_color, 
               label=mice, alpha=alphas[i])  # r_minus as squares with dashed line
        ax.plot(0, intercept, 'o-', color=intercept_color, 
               label=mice, alpha=alphas[i])  # Intercept at x=0
    
    # Add significance markers for each lag
    y_max = ax.get_ylim()[1]  # Get current y-axis limit
    
    # Add significance markers for r_plus regressors (top of plot)
    for lag in sorted(fisher_plus.keys()):
        p = fisher_plus[lag]
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        ax.text(lag, y_max * 0.95, marker, ha='center', fontsize=30, color=plus_color)
    
    # Add significance markers for r_minus regressors (slightly below r_plus)
    for lag in sorted(fisher_minus.keys()):
        p = fisher_minus[lag]
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        ax.text(lag, y_max * 0.90, marker, ha='center', fontsize=30, color=minus_color)
    
    # Add intercept significance marker (below the others)
    if intercept_pvalues:
        p = fisher_intercept
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        ax.text(0, y_max * 0.85, marker, ha='center', fontsize=25, color=intercept_color)
    
    # Add reference line at y=0, labels, and grid
    ax.axhline(y=0, color='black', linestyle='--', linewidth=3, alpha=0.5)
    ax.set_title('GLM Weights with Fisher-Combined Significance', pad=20)
    ax.set_ylabel('GLM Weight', labelpad=20)
    ax.set_xlabel('Previous Trials', labelpad=20)
    ax.grid(True, linestyle=':', alpha=0.3)
    
    # Create custom legend
    plt.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label=r'$\beta_{t}^+$', 
                  markerfacecolor=plus_color, markersize=10),  # r_plus
            Line2D([0], [0], marker='s', color='w', label=r'$\beta_{t}^-$', 
                  markerfacecolor=minus_color, markersize=10),  # r_minus
            Line2D([0], [0], marker='o', color='w', label='Intercept', 
                  markerfacecolor=intercept_color, markersize=10)  # Intercept
        ],
        loc='best', 
        fontsize=20, 
        framealpha=0.5, 
        edgecolor='black', 
        facecolor='white', 
        bbox_to_anchor=(0.95, 0.5),  # Position legend
    )
    
    # Format legend frame
    legend = ax.get_legend()
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    
    # Final layout adjustments and display
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
        - switch_num: dependent variable (binary switch behavior)
        - Other regressor columns containing past trial information
        
    Returns:
    --------
    None (displays plot with subplots for each mouse showing GLM coefficients)
    """
    
    # =============================================
    # 1. SET UP PLOT LAYOUT AND VISUAL PARAMETERS
    # =============================================
    
    # Calculate grid layout for subplots (2 rows x enough columns for all mice)
    n_subjects = len(df['subject'].unique())  # Count unique mice
    n_cols = int(np.ceil(n_subjects / 2))    # Calculate needed columns
    
    # Create figure with dynamic sizing based on number of mice
    # Width adjusts based on columns, fixed height of 8 inches
    f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
    
    mice_counter = 0  # Tracks current mouse's position in subplot grid
    
    # Color scheme for different regressor types
    regressor_colors = {
        'Intercept': '#2ca02c',  # Green for intercept term
        'r_plus': '#d62728',    # Red for positive reward predictors  
        'r_minus': '#1f77b4',   # Blue for negative reward predictors
    }
    
    # Storage for aggregated results across all mice (for potential meta-analysis)
    all_results = []
    
    # =============================================
    # 2. PROCESS EACH MOUSE INDIVIDUALLY
    # =============================================
    
    for mice in df['subject'].unique():
        if mice != 'A10':  # Skip excluded mouse if specified
            
            # 2.1 DATA FILTERING AND PREPARATION
            # ----------------------------------
            df_mice = df.loc[df['subject'] == mice]  # Filter to current mouse
            
            # Only keep sessions with sufficient trials (>50)
            session_counts = df_mice['session'].value_counts()
            valid_sessions = session_counts[session_counts > 50].index
            df_mice['sign_session'] = 0  # Initialize validity flag
            df_mice.loc[df_mice['session'].isin(valid_sessions), 'sign_session'] = 1
            new_df_mice = df_mice[df_mice['sign_session'] == 1]  # Final filtered data
            
            print(f"Processing mouse: {mice}")  # Progress tracking
            
            # Set analysis parameters
            n_back = 10  # Number of previous trials to consider in model
            
            # 2.2 MODEL SPECIFICATION AND CROSS-VALIDATION
            # --------------------------------------------
            
            # Prepare regressors and select training sessions
            df_values_new, regressor_string = obt_regressors(new_df_mice, n_back)
            df_cv = select_train_sessions(df_values_new)
            
            # Get current subplot axis based on grid position
            ax = axes[mice_counter//n_cols, mice_counter%n_cols]
            
            GLM_data = []  # Store results across all CV splits
            
            # Perform 5-fold cross-validation
            for i in range(5):  
                
                # 2.2.1 DATA SPLITTING
                # ---------------------
                df_80 = df_cv[df_cv[f'split_{i}'] == f'train']  # 80% training
                df_test = df_cv[df_cv[f'split_{i}'] == f'test']  # 20% testing
                
                # 2.2.2 MODEL FITTING
                # --------------------
                mM_logit = smf.logit(
                    formula='choice_num ~ ' + regressor_string, 
                    data=df_80
                ).fit(disp=0)  # Suppress convergence output
                
                # 2.2.3 STORE MODEL RESULTS
                # -------------------------
                GLM_df = pd.DataFrame({
                    'coefficient': mM_logit.params,        # Beta weights
                    'std_err': mM_logit.bse,              # Standard errors
                    'z_value': mM_logit.tvalues,          # Z-statistics
                    'p_value': mM_logit.pvalues,          # Significance
                    'conf_interval_low': mM_logit.conf_int()[0],  # 95% CI lower
                    'conf_interval_high': mM_logit.conf_int()[1], # 95% CI upper
                })
                
                # Add metadata for tracking
                GLM_df['subject'] = mice
                GLM_df['split'] = i
                GLM_df = GLM_df.reset_index().rename(columns={'index': 'regressor'})
                GLM_data.append(GLM_df)
                
                # Generate and store predictions on test set
                df_test['pred_prob'] = mM_logit.predict(df_test)
            
            # 2.3 AGGREGATE RESULTS ACROSS CV SPLITS
            # --------------------------------------
            df_GLM_data = pd.concat(GLM_data).reset_index(drop=True)
            
            # Use median across splits (robust to outliers)
            median_results = df_GLM_data.groupby('regressor').agg({
                'coefficient': 'median',
                'p_value': 'median'
            }).reset_index()
            
            # =============================================
            # 3. VISUALIZATION FOR CURRENT MOUSE
            # =============================================
            
            # 3.1 BASIC PLOT SETUP
            # --------------------
            ax.set_title(f'Mouse {mice}', fontsize=12)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Reference line
            
            # Get y-axis limits for significance marker placement
            y_min, y_max = ax.get_ylim()
            sig_height = y_max * 0.95  # Position above coefficients
            
            # Initialize list for x-tick positions
            x_ticks_vect = []
            
            # 3.2 PLOT EACH REGRESSOR'S COEFFICIENT
            # --------------------------------------
            for i, (_, row) in enumerate(median_results.iterrows()):
                reg = row['regressor']
                coef = row['coefficient']
                pval = row['p_value']
                
                # Determine color based on regressor type
                if 'r_plus' in reg:
                    color = '#d62728'  # Red for positive reward
                elif 'r_minus' in reg:
                    color = '#1f77b4'  # Blue for negative reward
                elif 'Intercept' in reg:
                    color = '#2ca02c'  # Green for intercept
                else:
                    color = '#7f7f7f'  # Gray for other terms
                
                # Determine x-tick label based on regressor type
                if 'last_trial' in reg:
                    j = 1  # Special position for last trial
                elif 'Intercept' in reg:
                    j = 0  # Intercept at position 0
                else:
                    # Extract lag number from regressor name
                    j = int(reg.split('_')[-1])  # Gets number after r_plus/minus_X
                
                x_ticks_vect.append(j)  # Store position for x-ticks
                
                # 3.2.1 PLOT COEFFICIENT POINT
                ax.plot(i, coef, 'o', markersize=6, color=color, alpha=0.8)
                
                # 3.2.2 ADD SIGNIFICANCE MARKER
                if pval < 0.001:
                    sig_marker = '***'
                elif pval < 0.01:
                    sig_marker = '**'
                elif pval < 0.05:
                    sig_marker = '*'
                else:
                    sig_marker = 'ns'  # Not significant
                
                ax.text(i, sig_height, sig_marker, 
                       ha='center', fontsize=10, color=color)
                
                # 3.3 STORE RESULTS FOR AGGREGATE ANALYSIS
                all_results.append({
                    'subject': mice,
                    'regressor': reg,
                    'coefficient': coef,
                    'p_value': pval
                })
            
            # 3.4 FINAL SUBPLOT FORMATTING
            # -----------------------------
            ax.set_xlabel('Regressor', fontsize=10)
            ax.set_ylabel('Coefficient', fontsize=10)
            ax.set_xticks(range(len(median_results)))
            ax.set_xticklabels(x_ticks_vect)  # Custom x-tick labels
            
            mice_counter += 1  # Advance to next subplot position

    # =============================================
    # 4. FINAL PLOT ADJUSTMENTS AND DISPLAY
    # =============================================
    
    # Adjust spacing between subplots to prevent overlap
    plt.tight_layout()
    
    # Display the complete figure
    plt.show()


if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials1.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # 1 for analisis of trained mice, 0 for untrained
    print(df['task'].unique())
    trained = 1
    new_df = parsing(df,trained,0)
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
