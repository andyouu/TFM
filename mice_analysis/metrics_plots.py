import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib import patches as mpatches
import matplotlib.lines as mlines



def plot_metrics_comparison(blocks, metrics_data, model_names):
    """
    Create separate plots for BIC and Accuracy with legends inside the plots
    
    Args:
        blocks: Array of probability blocks (e.g., [[0.2,0.8], [0.3,0.7]])
        metrics_data: Dictionary containing metrics for each model
        model_names: List of model names (e.g., ['glm_prob_switch', 'glm_prob_r'])
    """
    # Create a consistent color palette
    palette = sns.color_palette("husl", len(model_names))
    
    # Convert blocks to readable labels
    block_labels = [p for p in blocks]
    
    # Set font sizes
    title_fontsize = 50
    label_fontsize = 50
    legend_fontsize = 20  # Slightly smaller for inside placement
    tick_fontsize = 25
    
    # Plot BIC
    plt.figure(figsize=(10, 6))  # Slightly smaller for single plot
    plot_metric(blocks, metrics_data, model_names, 'BIC', 'BIC', 
               palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plot_metric(blocks, metrics_data, model_names, 'accuracy', 'Accuracy', 
               palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()
    
    #Plot Log Likelihood
    plt.figure(figsize=(10, 6))
    plot_metric(blocks, metrics_data, model_names, 'log_likelihood_per_obs', 'Log Likelihood per Obs',
               palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()

def plot_metric(blocks, metrics_data, model_names, metric, ylabel, 
                palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize):
    """Helper function to plot a single metric with internal legend"""
    # Prepare data
    plot_data = []
    for model_idx, model in enumerate(model_names):
        for block_idx, block in enumerate(blocks):
            values = metrics_data[model][metric][block_idx]
            for val in values:
                plot_data.append({
                    'Model': model,
                    'Probability': block_labels[block_idx],
                    'Value': val,
                    'Color': palette[model_idx]
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create plot
    ax = plt.gca()
    
    # Create boxplot
    sns.boxplot(
        x='Probability', 
        y='Value', 
        hue='Model',
        data=df_plot,
        palette=palette,
        width=0.6,
        linewidth=1.5
    )
    
    # Add individual data points
    sns.stripplot(
        x='Probability',
        y='Value',
        hue='Model',
        data=df_plot,
        dodge=True,
        palette=palette,
        alpha=0.5,
        edgecolor='gray',
        linewidth=0.5,
        jitter=True,
        size=4  # Slightly smaller points
    )
    
    # Format plot
    #ax.set_title(ylabel, fontsize=title_fontsize, pad=10)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Customize legend - placed inside plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        labels, 
        title='Model',
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc='best',  # Changed to upper right inside
        framealpha=1,
        edgecolor='black'
    )
    
    # Add grid and clean borders
    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)

def plot_metrics_comparison_grouped_by_models(blocks, metrics_data, model_names):
    """
    Create separate plots for BIC, Accuracy, and Log Likelihood grouped by models
    
    Args:
        blocks: Array of probability blocks (e.g., [[0.2,0.8], [0.3,0.7]])
        metrics_data: Dictionary containing metrics for each model
        model_names: List of model names (e.g., ['glm_prob_switch', 'glm_prob_r'])
    """
    # Create a consistent color palette
    palette = sns.color_palette("husl", len(blocks)+2)
    
    # Set font sizes
    title_fontsize = 50
    label_fontsize = 50
    legend_fontsize = 20
    tick_fontsize = 25
    
    # Plot BIC
    plt.figure(figsize=(10, 6))  # Slightly smaller for single plot
    plot_metric_grouped_by_models(blocks, metrics_data, model_names, 'log_likelihood', 'log_likelihood',
                                    palette, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.figure(figsize=(10, 6)) 
    plot_metric_grouped_by_models(blocks, metrics_data, model_names, 'n_trials', 'n_trials',
                                    palette, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.figure(figsize=(10, 6))
    plot_metric_grouped_by_models(blocks, metrics_data, model_names, 'BIC', 'BIC', 
                                 palette, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()
    # Plot Log Likelihood
    plt.figure(figsize=(10, 6))
    plot_metric_grouped_by_models(blocks, metrics_data, model_names, 'log_likelihood_per_obs', 'Log Likelihood per Obs',
                                 palette, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plot_metric_grouped_by_models(blocks, metrics_data, model_names, 'accuracy', 'Accuracy', 
                                 palette, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()
    


def plot_metric_grouped_by_models(blocks, metrics_data, model_names, metric, ylabel, 
                                palette, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize):
    """Helper function to plot a single metric grouped by models with internal legend"""
    # Prepare data
    plot_data = []
    for block_idx, block in enumerate(blocks):
        for model_idx, model in enumerate(model_names):
            values = metrics_data[model][metric][block_idx]
            for val in values:
                if (model != 'inference_based') and (block_idx == 3):
                    plot_data.append({
                        'Block': f'{block_idx+3}',
                        'Model': model,
                        'Value': val,
                        'Color': palette[block_idx+2]
                    })
                elif (model != 'inference_based') and (block_idx == 4):
                    plot_data.append({
                        'Block': f'{block_idx+5}',
                        'Model': model,
                        'Value': val,
                        'Color': palette[block_idx+2]
                    })
                elif model == 'model_0':
                    plot_data.append({
                        'Block': f'{block_idx+1}',
                        'Model': model,
                        'Value': val,
                        'Color': palette[1]
                    })
                else:
                    plot_data.append({
                        'Block': f'{block_idx+1}',
                        'Model': model,
                        'Value': val,
                        'Color': palette[block_idx]
                    })
                
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create plot
    ax = plt.gca()
    
    # Create boxplot
    sns.boxplot(
        x='Model', 
        y='Value', 
        hue='Block',
        data=df_plot,
        palette=palette,
        width=0.6,
        linewidth=1.5
    )
    #put labels depending on the color using mpatches

    # Add individual data points
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
        jitter=True,
        size=4,
        legend=False
    )
    
    # Format plot
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlabel('Model', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Customize legend - placed inside plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        labels, 
        title='Trials_back',
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc='upper left',          # Anchor point for the legend
        bbox_to_anchor=(1.02, 1), 
        framealpha=1,
        edgecolor='black'
    )
    
    # Add grid and clean borders
    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)


if __name__ == '__main__':
    main_folder = '/home/marcaf/TFM(IDIBAPS)/rrns2/networks'
    w_factor = 0.01
    mean_ITI = 400
    max_ITI = 800
    fix_dur = 100
    dec_dur = 100
    blk_dur = 38
    n_regressors = 4
    n_back = 3
    blocks = np.array([
        [0, 0.9],[0.2, 0.8],[0.3, 0.7],[0.4, 0.6]#,[2,2]
    ])
    blocks = np.array(['Mice'])
    # Store metrics for each model
    metrics_data = {
        'glm_prob_switch': {
            'log_likelihood_per_obs': [],
            'log_likelihood': [],
            'n_trials': [],
            'BIC': [],
            'AIC': [],
            'accuracy': []
        },
        'glm_prob_r': {
            'log_likelihood_per_obs': [],
            'log_likelihood': [],
            'n_trials': [],
            'BIC': [],
            'AIC': [],
            'accuracy': []
        },
        'inference_based': {
            'log_likelihood_per_obs': [],
            'log_likelihood': [],
            'n_trials': [],
            'BIC': [],
            'AIC': [],
            'accuracy': []
        },
        # 'inference_based_v2': { 
        #     'log_likelihood_per_obs': [],
        #     'log_likelihood': [],
        #     'n_trials': [],
        #     'BIC': [],
        #     'AIC': [],
        #     'accuracy': []
        # },
        # 'model_0': {
        #     'log_likelihood_per_obs': [],
        #     'log_likelihood': [],
        #     'n_trials': [],
        #     'BIC': [],
        #     'AIC': [],
        #     'accuracy': []
        # }
    }
    for model in ['glm_prob_switch', 'glm_prob_r', 'inference_based']:# 'model_0','inference_based_v2']: #
        if model == 'inference_based' or model == 'inference_based_v2' or model == 'model_0':
            blocks = [1,2,3,4,5]
        else:
            blocks = [2,3,4,7,10]
        for block in blocks:
            folder = ('/home/marcaf/TFM(IDIBAPS)/codes/data/all_subjects_glm_metrics_') + model + f'_{block}.csv'
            if os.path.exists(folder):
                df = pd.read_csv(folder)
                #To just plot one point for each seed, comment to see all cross-validation cases
                df = df.groupby('subject').median()
                # Store metrics
                metrics_data[model]['log_likelihood_per_obs'].append(df['log_likelihood_per_obs'].values)
                metrics_data[model]['log_likelihood'].append(df['log_likelihood'].values)
                metrics_data[model]['n_trials'].append(df['log_likelihood'].values/df['log_likelihood_per_obs'].values)
                metrics_data[model]['BIC'].append(df['BIC'].values)
                metrics_data[model]['AIC'].append(df['AIC'].values)
                metrics_data[model]['accuracy'].append(df['accuracy'].values)
            else:
                print(f"Metrics file not found: {folder}")
                # Append empty arrays if data is missing
                metrics_data[model]['log_likelihood_per_obs'].append(np.array([]))
                metrics_data[model]['log_likelihood'].append(np.array([]))
                metrics_data[model]['BIC'].append(np.array([]))
                metrics_data[model]['AIC'].append(np.array([]))
                metrics_data[model]['accuracy'].append(np.array([]))
        
        # Generate plots
        # Combine data for plotting

    plot_metrics_comparison_grouped_by_models(blocks, metrics_data, ['inference_based','glm_prob_switch', 'glm_prob_r'])#,'inference_based_v2','model_0']) #maybe add model_0
    plot_metrics_comparison(blocks, metrics_data, ['glm_prob_switch', 'glm_prob_r', 'inference_based'])