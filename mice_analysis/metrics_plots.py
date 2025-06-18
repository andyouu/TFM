import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib import patches as mpatches
import matplotlib.lines as mlines



def plot_metrics_comparison_grouped_by_models(blocks, metrics_data, model_names):
    """
    Creates comparative visualizations of model performance metrics grouped by models.
    Generates four separate plots for different evaluation metrics.
    
    Args:
        blocks: List of probability blocks defining experimental conditions
                (e.g., [[0.2,0.8], [0.3,0.7]] where each sublist represents reward probabilities)
                
        metrics_data: Nested dictionary containing evaluation metrics for each model
                     Structure: {model_name: {metric_name: [values_across_blocks]}}
                     
        model_names: List of strings identifying models to compare
                    (e.g., ['glm_prob_switch', 'glm_prob_r'])
    
    Plots Generated:
    1. Log Likelihood
    2. Number of Trials (n_trials)
    3. BIC (Bayesian Information Criterion)
    4. Log Likelihood per Observation
    5. Accuracy
    
    Visualization Features:
    - Consistent color scheme across all plots
    - Large, readable fonts for presentations
    - Automatic layout adjustment
    - Model-grouped comparison
    """
    
    # Create color palette with 2 extra colors beyond what's needed for blocks
    # (The +2 suggests anticipation of additional conditions)
    palette = sns.color_palette("viridis", len(blocks)+2)
    
    # Set font sizes for publication-quality plots
    title_fontsize = 50    # Large title size 
    label_fontsize = 50    # Axis labels
    legend_fontsize = 20   # Smaller to fit inside plot
    tick_fontsize = 25     # Tick labels
    
    # Plot 1: Log Likelihood (raw)
    plt.figure(figsize=(10, 6))
    plot_metric_grouped_by_models(
        blocks, metrics_data, model_names, 
        'log_likelihood', 'log_likelihood',
        palette, title_fontsize, label_fontsize, 
        legend_fontsize, tick_fontsize
    )
    
    # Plot 2: Number of Trials
    plt.figure(figsize=(10, 6))
    plot_metric_grouped_by_models(
        blocks, metrics_data, model_names,
        'n_trials', 'n_trials',
        palette, title_fontsize, label_fontsize,
        legend_fontsize, tick_fontsize
    )
    
    # Plot 3: BIC
    plt.figure(figsize=(10, 6))
    plot_metric_grouped_by_models(
        blocks, metrics_data, model_names,
        'BIC', 'BIC',
        palette, title_fontsize, label_fontsize,
        legend_fontsize, tick_fontsize
    )
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Normalized Log Likelihood
    plt.figure(figsize=(10, 6))
    plot_metric_grouped_by_models(
        blocks, metrics_data, model_names,
        'log_likelihood_per_obs', 'Log Likelihood per Obs',
        palette, title_fontsize, label_fontsize,
        legend_fontsize, tick_fontsize
    )
    plt.tight_layout()
    plt.show()
    
    # Plot 5: Accuracy
    plt.figure(figsize=(10, 6))
    plot_metric_grouped_by_models(
        blocks, metrics_data, model_names,
        'accuracy', 'Accuracy',
        palette, title_fontsize, label_fontsize,
        legend_fontsize, tick_fontsize
    )
    plt.tight_layout()
    plt.show()


def plot_metric_grouped_by_models(blocks, metrics_data, model_names, metric, ylabel, 
                                palette, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize):
    """
    Creates a grouped comparison plot for a specific metric across models and blocks.
    
    Args:
        blocks: List of probability blocks/conditions (e.g., [[0.2,0.8], [0.3,0.7]])
        metrics_data: Nested dictionary {model: {metric: [values_per_block]}}
        model_names: List of model identifiers
        metric: The specific metric to plot (key in metrics_data)
        ylabel: Label for the y-axis
        palette: Color palette for different blocks
        title_fontsize: Font size for title (currently unused)
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legend text
        tick_fontsize: Font size for axis ticks
    
    Produces:
        A boxplot with overlaid stripplots showing:
        - X-axis: Different models
        - Y-axis: Metric values
        - Hue: Probability blocks/conditions
        - Colors: Distinct for each block
    """
    
    # ===== DATA PREPARATION =====
    plot_data = []
    for block_idx, block in enumerate(blocks):
        for model_idx, model in enumerate(model_names):
            values = metrics_data[model][metric][block_idx]
            
            # Special case handling for model-specific formatting
            if model != 'inference_based':
                if block_idx == 3:
                    block_label = f'{block_idx+3}'
                    color_idx = block_idx+2
                elif block_idx == 4:
                    block_label = f'{block_idx+5}'
                    color_idx = block_idx+2
                else:
                    block_label = f'{block_idx+1}'
                    color_idx = block_idx
            elif model == 'model_0':
                block_label = f'{block_idx+1}'
                color_idx = 1
            else:
                block_label = f'{block_idx+1}'
                color_idx = block_idx
            
            # Append all values for this block-model combination
            for val in values:
                plot_data.append({
                    'Block': block_label,
                    'Model': model,
                    'Value': val,
                    'Color': palette[color_idx]
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    # ===== PLOT CREATION =====
    plt.figure(figsize=(12, 8))  # Larger figure for better legend spacing
    ax = plt.gca()
    
    # -- BOXPLOT --
    sns.boxplot(
        x='Model', 
        y='Value', 
        hue='Block',
        data=df_plot,
        palette=palette,
        width=0.6,
        linewidth=1.5,
        fliersize=3  # Size of outlier markers
    )
    
    # -- STRIPPLOT --
    sns.stripplot(
        x='Model',
        y='Value',
        hue='Block',
        data=df_plot,
        dodge=True,
        palette=palette,
        alpha=0.5,
        edgecolor='gray',
        linewidth=0.5,
        jitter=0.2,  # Reduced jitter for better alignment
        size=4,
        legend=False
    )
    
    # ===== PLOT FORMATTING =====
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlabel('Model', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Adjust x-tick labels if needed
    ax.set_xticklabels([m.replace('_', ' ').title() for m in model_names])
    
    # ===== LEGEND CUSTOMIZATION =====
    # Get unique handles/labels while preserving order
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    
    ax.legend(
        unique_handles, 
        unique_labels, 
        title='Trials Back',
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),  # Further right to prevent overlap
        framealpha=1,
        edgecolor='black'
    )
    
    # ===== FINAL TOUCHES =====
    plt.grid(True, alpha=0.2, linestyle='--')
    sns.despine(ax=ax)
    plt.tight_layout()


if __name__ == '__main__':
    # ===== EXPERIMENT PARAMETERS =====
    # Path and file configuration
    main_folder = '/home/marcaf/TFM(IDIBAPS)/rrns2/networks'
    results_base = '/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_'
    
    # Timing parameters (currently unused in visualization)
    timing_params = {
        'w_factor': 0.01,
        'mean_ITI': 400,
        'max_ITI': 800,
        'fix_dur': 100,
        'dec_dur': 100,
        'blk_dur': 38
    }
    
    # Model configuration
    n_regressors = 4
    n_back = 3
    
    # ===== DATA STRUCTURE INITIALIZATION =====
    # Define models to analyze (comment/uncomment as needed)
    models_to_analyze = [
        'glm_prob_switch', 
        'glm_prob_r', 
        'inference_based'
        # 'inference_based_v2',
        # 'model_0'
    ]
    
    # Block definitions for different model types
    block_config = {
        'standard_models': [2, 3, 4, 7, 10],    # For glm_prob_switch and glm_prob_r
        'special_models': [1, 2, 3, 4, 5]       # For inference_based and variants
    }
    
    # Initialize metrics data structure
    metrics_data = {
        model: {
            'log_likelihood_per_obs': [],
            'log_likelihood': [],
            'n_trials': [],
            'BIC': [],
            'AIC': [],
            'accuracy': []
        } 
        for model in models_to_analyze
    }
    
    # ===== DATA LOADING =====
    for model in models_to_analyze:
        # Determine which blocks to use for this model
        blocks = block_config['special_models'] if model in ['inference_based', 'inference_based_v2', 'model_0'] else block_config['standard_models']
        
        for block in blocks:
            file_path = f"{results_base}{model}_{block}.csv"
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Use median of each subject's results (comment out to see all data points)
                    df = df.groupby('subject').median()
                    
                    # Store metrics
                    metrics_data[model]['log_likelihood_per_obs'].append(df['log_likelihood_per_obs'].values)
                    metrics_data[model]['log_likelihood'].append(df['log_likelihood'].values)
                    
                    # Calculate trials as likelihood / likelihood_per_obs
                    with np.errstate(divide='ignore', invalid='ignore'):
                        n_trials = np.where(
                            df['log_likelihood_per_obs'] != 0,
                            df['log_likelihood'] / df['log_likelihood_per_obs'],
                            0
                        )
                    metrics_data[model]['n_trials'].append(n_trials)
                    
                    metrics_data[model]['BIC'].append(df['BIC'].values)
                    metrics_data[model]['AIC'].append(df['AIC'].values)
                    metrics_data[model]['accuracy'].append(df['accuracy'].values)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    # Append empty array if processing fails
                    for metric in metrics_data[model]:
                        metrics_data[model][metric].append(np.array([]))
            else:
                print(f"Metrics file not found: {file_path}")
                # Append empty arrays if file is missing
                for metric in metrics_data[model]:
                    metrics_data[model][metric].append(np.array([]))
    
    # ===== VISUALIZATION =====
    # Define which models to plot (can be different from analysis set)
    models_to_plot = ['inference_based', 'glm_prob_switch', 'glm_prob_r']
    
    # Generate comparative visualizations
    try:
        print("Generating grouped comparison plots...")
        plot_metrics_comparison_grouped_by_models(
            block_config['standard_models'],  # Using standard blocks for visualization
            metrics_data, 
            models_to_plot
        )
    except Exception as e:
        print(f"Error generating plots: {str(e)}")