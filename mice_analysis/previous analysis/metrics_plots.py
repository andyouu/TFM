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
from matplotlib import rcParams
from extra_plotting import *
from model_avaluation import *
from parsing import *

def plot_performance_comparison(df_data):    
    # Set up figure for A0 poster
    plt.figure(figsize=(24, 16))
    
    # Set font sizes
    title_fontsize = 55
    label_fontsize = 55
    tick_fontsize = 30
    legend_fontsize = 30
    
    good_perfs = []
    bad_perfs = []
    boxplot_labels = ['Mice']
    good_block_perfs = []
    bad_block_perfs = []
    
    subjects = np.unique(df_data['subject'])
    for subj in subjects:
        data_s = df_data[df_data['subject'] == subj]
        mean_perf = np.sum(data_s['perf']) / len(data_s)
        print(f'AVERAGE performance: {mean_perf}')
        
        if mean_perf < 0.55:
            bad_block_perfs.append(mean_perf)
        else:
            good_block_perfs.append(mean_perf)
    
    # Append to lists   
    good_perfs.append(good_block_perfs)
    bad_perfs.append(bad_block_perfs)
    
    # Create boxplot for good networks
    bp = plt.boxplot(good_perfs, positions=[1],
                    patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor='skyblue', alpha=0.6))  # Added color and alpha
    
    # Style other elements
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=2)
    
    # Add scattered points for bad networks
    if bad_perfs[0]:  # Only plot if there are bad performers
        x_pos = np.random.normal(1, 0.05, size=len(bad_perfs[0]))
        plt.scatter(x_pos, good_perfs[0], color='blue', alpha=0.7, 
                   s=100, edgecolor='black', linewidth=1.5,
                   label='Poor Performers')
    
    # Add chance level line
    plt.axhline(0.55, color='gray', linestyle='--', linewidth=3, alpha=0.7)
    
    # Customize plot
    plt.xticks(np.arange(1, 2), boxplot_labels, fontsize=tick_fontsize)
    plt.ylabel('Average Performance', fontsize=label_fontsize, labelpad=20)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylim(0.4, 0.85)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='skyblue', alpha=0.6, label='Good Performers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=15, markeredgecolor='black', label='Poor Performers'),
        plt.Line2D([0], [0], color='gray', linestyle='--', 
                  label='Discrimination Threshold', linewidth=3),
    ]
    
    plt.legend(handles=legend_elements, fontsize=legend_fontsize,
              loc='upper right', framealpha=0.9)
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_perf_psychos(combined_data):
    data = combined_data
    
    # Set up figure for A0 poster with constrained layout
    plt.figure(figsize=(20, 18), constrained_layout=False)
    
    # Set font sizes
    title_fontsize = 28
    label_fontsize = 50
    legend_fontsize = 30
    tick_fontsize = 30
    linewidth = 5
    markersize = 14
    
    # Get all training blocks and sort them
    subjects = np.sort(np.unique(data['subject']))
    colors = sns.color_palette("viridis", len(subjects))
    # Initialize alpha value
    base_alpha = 1.0
    alpha_step = 0.15
    
    # Create colormap for different training blocks
    avg_perf = []
    for i, subj in enumerate(subjects):
        data_s = data[data['subject'] == subj]
        perf = np.array(data_s['perf'])
        perf = perf[perf != -1]
        block_values = np.unique(data_s['probability_r'])
        block_values = block_values[block_values!=0.5]
        all_perf_by_block = {blk: [] for blk in np.unique(block_values)}
        for blk in block_values:
            mask = (data_s['probability_r'] == blk)[:len(perf)]
            perf_cond = perf[mask]
            mean_perf_cond = np.mean(perf_cond) if len(perf_cond) > 0 else np.nan
            transformed_perf = mean_perf_cond if blk > 0.5 else 1 - mean_perf_cond
            all_perf_by_block[blk].append(transformed_perf)
    
        block_probs = sorted(all_perf_by_block.keys())
        mean_perfs = [np.mean(all_perf_by_block[blk]) for blk in block_probs]
        
        avg_perf.append(mean_perfs)
        
        
        plt.plot(block_probs, mean_perfs, 'o-', 
                color=colors[i], 
                alpha=0.7,
                linewidth=linewidth, 
                markersize=markersize,
                label=subj,
                markeredgecolor='black',
                markeredgewidth=1.5)
        
    plt.plot(block_probs, np.mean(avg_perf, axis=0), 'o-',
            color='black', 
            alpha=0.9,
            linewidth=linewidth, 
            markersize=markersize,
            label='Average',
            markeredgecolor='black',
            markeredgewidth=1.5)
    plt.plot(block_probs, block_probs, 'k--', alpha=0.5, linewidth=linewidth)
    # Add chance level line
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, 
               linewidth=3)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, 
            linewidth=3)
    # Customize plot appearance with padding
    # plt.title('Average Probability Right Across Probability Blocks', 
    #          fontsize=title_fontsize, pad=25)  # Increased pad
    plt.xlabel('Block Prob of reward (Right)', 
              fontsize=label_fontsize, labelpad=20)  # Increased labelpad
    plt.ylabel('Average Prob of right', 
              fontsize=label_fontsize, labelpad=20)
    
    # Customize legend
    legend = plt.legend(loc='best', 
                      fontsize=legend_fontsize,
                      framealpha=0.9,
                      edgecolor='black',
                      facecolor='white',
                      bbox_to_anchor=(1.01, 1.01),  # 1.01 moves it right of the axes, 0 aligns with bottom
                      borderaxespad=0.5)
    
    # Make legend frame visible
    legend.get_frame().set_linewidth(2)

    
    # Customize ticks and grid
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.grid(True, alpha=0.3, linewidth=1.5)
    plt.yticks(fontsize=tick_fontsize)
    
    # Adjust layout to ensure everything fits
    plt.tight_layout(pad=4.0)  # Increased padding
    plt.show()

def outcome_block_change(df,window,switchy=0):
    subjects = np.unique(df['subject'])
    #keep only the right-left actions
    df = df[df['choice'] >= 0]
    
    if switchy == 1:
        df['choice_1'] = df['choice'].shift(1)
        df['switch_num'] = np.where(df['choice'] == df['choice_1'], 0, 1)
    else:
        df['probability_r_1'] = df['probability_r'].shift(1)
        df['switch_num'] = np.where(df['probability_r'] == df['probability_r_1'], 0, 1)
    
    # Calculate fraction of correct responses
    df['fraction_of_correct_responses'] = np.where(
        ((df['probability_r'] >= 0.5) & (df['choice'] == 1)) |
        ((df['probability_r'] < 0.5) & (df['choice'] == 0)),
        1, 0
    )
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(24, 16))
    title_fontsize = 50
    label_fontsize = 50
    legend_fontsize = 30
    tick_fontsize = 30
    
    # Define colors for different subjects
    colors = plt.cm.viridis(np.linspace(0, 1, len(subjects)))
    
    # Initialize lists to store all aligned data
    all_outcome = []
    all_switches = []
    
    for i, subj in enumerate(subjects):
        subj_df = df[df['subject'] == subj].reset_index(drop=True)
        
        # ----- Switch indices -----
        switch_idx = subj_df.index[subj_df['switch_num'] == 1]
        
        # ----- Switch-triggered outcome -----
        for idx in switch_idx:
            if idx - window < 0 or idx + window >= len(subj_df):
                continue

            outcome_vals = subj_df.iloc[idx - window:idx + window + 1]['fraction_of_correct_responses'].values
            switch_vals = subj_df.iloc[idx - window:idx + window + 1]['switch_num'].values
            
            for lag in range(-window, window + 1):
                all_outcome.append({'subject': subj, 'lag': lag, 
                                   'fraction_of_correct_responses': outcome_vals[lag + window]})
                all_switches.append({'subject': subj, 'lag': lag, 
                                    'switch': switch_vals[lag + window]})
    
    # Convert to DataFrames
    outcome_df = pd.DataFrame(all_outcome)
    switch_df = pd.DataFrame(all_switches)
    
    # Plot each subject's data
    for i, subj in enumerate(subjects):
        subj_outcome = outcome_df[outcome_df['subject'] == subj]
        outcome_mean = subj_outcome.groupby('lag')['fraction_of_correct_responses'].mean()
        outcome_sem = subj_outcome.groupby('lag')['fraction_of_correct_responses'].sem()
        
        ax.errorbar(outcome_mean.index, outcome_mean.values, yerr=outcome_sem.values,
                    fmt='o-', capsize=3, color=colors[i],
                    label=f'Subject {subj}', linewidth=5)
    
    # Reference lines and labels
    if switchy == 1:
        ax.set_xlabel('Trials from switch', size=label_fontsize)
        ax.axvline(0, linestyle='--', color='black', label='Switch')
    else:
        ax.set_xlabel('Trials from block transition', size=label_fontsize)
        ax.axvline(0, linestyle='--', color='black', label='Block change')
    
    ax.axhline(0.5, linestyle='--', color='gray', label='Chance')
    ax.set_ylabel('Average Performance', size=label_fontsize)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(
            loc='lower left', 
            fontsize=legend_fontsize,
            bbox_to_anchor=(1.01, 0),  # 1.01 moves it right of the axes, 0 aligns with bottom
            frameon=True,  # Keep the legend frame visible
            borderaxespad=0.  # Remove padding between legend and axes
        )
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials1.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # 1 for analisis of trained mice, 0 for untrained
    print(df['task'].unique())
    trained = 1
    new_df = parsing(df,trained,0)
    new_df['outcome_bool'] = np.where(new_df['outcome'] == "correct", 1, 0)
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'right'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'left'), 'choice'] = 0
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['side'] == 'left'), 'choice'] = 1
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['side'] == 'right'), 'choice'] = 1
    new_df['perf'] = 0
    new_df.loc[(new_df['probability_r'] > 0.5) & (new_df['choice'] == 1), 'perf'] = 1
    new_df.loc[(new_df['probability_r'] < 0.5) & (new_df['choice'] == 0), 'perf'] = 1
    #plot_performance_comparison(new_df)
    plot_perf_psychos(new_df)
    outcome_block_change(new_df,switchy=0,window=10)