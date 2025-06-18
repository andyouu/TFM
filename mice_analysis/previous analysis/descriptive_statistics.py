import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from extra_plotting import *
from model_avaluation import *
from parsing import *

def plot_performance_comparison(df_data, chance_level=0.55):
    """
    Visualize performance comparison between good and poor performers with boxplots.
    
    Args:
        df_data: DataFrame containing performance data with columns:
            - subject: subject identifier
            - perf: performance values (0-1)
        chance_level: Threshold to separate good/poor performers (default: 0.55)
        save_path: Optional path to save the figure (default: None)
    
    Produces:
        A boxplot visualization showing:
        - Boxplot of good performers (above chance_level)
        - Individual points for poor performers
        - Reference line at chance_level
    """
    # ===== VISUALIZATION PARAMETERS =====
    params = {
        'figure.figsize': (24, 16),
        'font.size': 30,
        'axes.titlesize': 55,
        'axes.labelsize': 55,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30,
        'legend.framealpha': 0.9,
        'grid.alpha': 0.3
    }
    plt.rcParams.update(params)
    
    # ===== DATA PROCESSING =====
    subjects = np.unique(df_data['subject'])
    good_perfs = []
    poor_perfs = []
    
    for subj in subjects:
        subj_data = df_data[df_data['subject'] == subj]
        mean_perf = np.mean(subj_data['perf'])
        
        if mean_perf >= chance_level:
            good_perfs.append(mean_perf)
        else:
            poor_perfs.append(mean_perf)
    
    # Print summary statistics
    print(f"\nPerformance Summary:")
    print(f"Good performers (>={chance_level:.2f}): {len(good_perfs)} subjects")
    print(f"Poor performers (<{chance_level:.2f}): {len(poor_perfs)} subjects")
    if good_perfs:
        print(f"Good performer range: {min(good_perfs):.2f}-{max(good_perfs):.2f}")
    if poor_perfs:
        print(f"Poor performer range: {min(poor_perfs):.2f}-{max(poor_perfs):.2f}")
    
    # ===== VISUALIZATION =====
    fig, ax = plt.subplots()
    
    # Create boxplot for good performers
    if good_perfs:
        bp = ax.boxplot([good_perfs], positions=[1], patch_artist=True,
                       widths=0.5, showfliers=False)
        
        # Style boxplot elements
        plt.setp(bp['boxes'], facecolor='skyblue', alpha=0.6)
        plt.setp(bp['whiskers'], color='black', linewidth=2)
        plt.setp(bp['caps'], color='black', linewidth=2)
        plt.setp(bp['medians'], color='darkblue', linewidth=3)
    
    # Add jittered points for poor performers
    if poor_perfs:
        x_pos = np.random.normal(1, 0.05, size=len(poor_perfs))
        ax.scatter(x_pos, poor_perfs, color='red', alpha=0.7, s=150,
                  edgecolor='black', linewidth=1.5, zorder=3)
    
    # Add individual points for good performers (with slight jitter)
    if good_perfs:
        x_pos = np.random.normal(1, 0.05, size=len(good_perfs))
        ax.scatter(x_pos, good_perfs, color='blue', alpha=0.5, s=100,
                  edgecolor='black', linewidth=1, zorder=3)
    
    # ===== PLOT FORMATTING =====
    # Reference lines and labels
    ax.axhline(chance_level, color='gray', linestyle='--', 
               linewidth=3, alpha=0.7, label='Chance Level')
    
    # Axis configuration
    ax.set_xticks([1])
    ax.set_xticklabels(['Mice'])
    ax.set_ylabel('Average Performance', labelpad=20)
    ax.set_ylim(0.4, 0.85)
    
    # Create custom legend
    legend_elements = [
        mpatches.Patch(facecolor='skyblue', alpha=0.6, 
                      label=f'Good Performers (≥{chance_level})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=15, markeredgecolor='black', 
                  label=f'Poor Performers (<{chance_level})'),
        plt.Line2D([0], [0], color='gray', linestyle='--', 
                  label='Chance Level', linewidth=3)
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add statistical annotation if both groups exist
    if good_perfs and poor_perfs:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(good_perfs, poor_perfs)
        ax.text(0.5, 0.42, f'Mann-Whitney U p = {pval:.3f}', 
               fontsize=25, ha='center')
    
    # Final layout adjustments
    plt.tight_layout()
    
    plt.show()

def plot_performance_psychometric(combined_data):
    """
    Plot psychometric curves showing performance across different probability blocks.
    
    Args:
        combined_data: DataFrame containing behavioral data with columns:
            - subject: subject identifier
            - perf: performance values (0-1)
            - probability_r: probability of reward for right choice
        save_path: Optional path to save the figure (default: None)
    
    Produces:
        A psychometric plot showing:
        - Individual subject curves
        - Group average curve
        - Ideal performance line
        - Chance level references
    """
    # ===== VISUALIZATION PARAMETERS =====
    params = {
        'figure.figsize': (20, 18),
        'axes.titlesize': 28,
        'axes.labelsize': 50,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30,
        'grid.alpha': 0.3,
        'grid.linewidth': 1.5
    }
    plt.rcParams.update(params)
    
    # Plot styling
    linewidth = 5
    markersize = 14
    chance_style = {'color': 'red', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 3}
    
    # ===== DATA PROCESSING =====
    subjects = np.sort(np.unique(combined_data['subject']))
    colors = sns.color_palette("viridis", len(subjects))
    all_perfs = []
    block_probs = None
    
    # Prepare figure
    fig, ax = plt.subplots(constrained_layout=True)
    
    # Process each subject's data
    for i, subj in enumerate(subjects):
        subj_data = combined_data[combined_data['subject'] == subj]
        perf = np.array(subj_data['perf'])
        perf = perf[perf != -1]  # Remove invalid values
        
        # Get unique probability blocks (excluding 0.5 if present)
        blocks = np.unique(subj_data['probability_r'])
        blocks = blocks[blocks != 0.5]
        
        # Calculate performance by block
        perf_by_block = {}
        for blk in blocks:
            mask = (subj_data['probability_r'] == blk)[:len(perf)]
            perf_cond = perf[mask]
            
            # Transform left-side probabilities to right-equivalents
            transformed_perf = np.mean(perf_cond) if blk > 0.5 else 1 - np.mean(perf_cond)
            perf_by_block[blk] = transformed_perf
        
        if block_probs is None:
            block_probs = np.sort(list(perf_by_block.keys()))
        
        # Store sorted performances for averaging
        sorted_perfs = [perf_by_block[blk] for blk in block_probs]
        all_perfs.append(sorted_perfs)
        
        # Plot individual subject curve
        ax.plot(block_probs, sorted_perfs, 'o-',
               color=colors[i],
               alpha=0.7,
               linewidth=linewidth,
               markersize=markersize,
               label=subj,
               markeredgecolor='black',
               markeredgewidth=1.5,
               zorder=2)
    
    # ===== GROUP AVERAGE =====
    if len(all_perfs) > 1:
        mean_perfs = np.nanmean(all_perfs, axis=0)
        ax.plot(block_probs, mean_perfs, 'o-',
               color='black',
               alpha=0.9,
               linewidth=linewidth,
               markersize=markersize,
               label='Average',
               markeredgecolor='black',
               markeredgewidth=1.5,
               zorder=3)
    
    # ===== REFERENCE LINES =====
    # Ideal performance line
    ax.plot(block_probs, block_probs, 'k--', 
           alpha=0.5, 
           linewidth=linewidth,
           label='Ideal Performance',
           zorder=1)
    
    # Chance level references
    ax.axhline(0.5, **chance_style)
    ax.axvline(0.5, **chance_style)
    
    # ===== PLOT FORMATTING =====
    ax.set_xlabel('Probability of Reward (Right)', labelpad=20)
    ax.set_ylabel('Probability of Choosing Right', labelpad=20)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    
    # Custom legend
    legend = ax.legend(loc='upper left',
                     bbox_to_anchor=(1.01, 1),
                     framealpha=0.9,
                     edgecolor='black',
                     facecolor='white')
    legend.get_frame().set_linewidth(2)
    
    # Add text annotation for chance level
    ax.text(0.02, 0.52, 'Chance Level', color='red', 
           fontsize=25, alpha=0.7)
    
    plt.show()

def outcome_block_change(df, window, switch_type=0):
    """
    Analyze and visualize performance around block changes or choice switches.
    
    Args:
        df: DataFrame containing behavioral data with columns:
            - subject: subject identifier
            - choice: subject's choice (0=left, 1=right)
            - probability_r: probability of reward for right choice
        window: Number of trials to show before/after the event
        switch_type: 0=block transitions (probability changes), 
                    1=choice switches
    
    Produces:
        A plot showing performance aligned to switch events with:
        - Individual subject traces
        - SEM error bars
        - Reference lines for event timing and chance performance
    """
    # ===== PARAMETERS =====
    plot_params = {
        'figsize': (24, 16),
        'title_fontsize': 50,
        'label_fontsize': 50,
        'legend_fontsize': 30,
        'tick_fontsize': 30,
        'linewidth': 5,
        'capsize': 3,
        'chance_line_style': {'color': 'gray', 'linestyle': '--'},
        'event_line_style': {'color': 'black', 'linestyle': '--'}
    }
    
    # ===== DATA PREPROCESSING =====
    # Filter valid choices and create working copy
    df = df[df['choice'] >= 0].copy()
    subjects = np.unique(df['subject'])
    
    # Calculate switch points
    if switch_type == 1:
        # Choice switches
        df['choice_1'] = df['choice'].shift(1)
        df['switch_num'] = np.where(df['choice'] == df['choice_1'], 0, 1)
        event_label = 'Switch'
    else:
        # Block transitions (probability changes)
        df['probability_r_1'] = df['probability_r'].shift(1)
        df['switch_num'] = np.where(df['probability_r'] == df['probability_r_1'], 0, 1)
        event_label = 'Block change'
    
    # Calculate trial correctness
    df['correct'] = (
        ((df['probability_r'] >= 0.5) & (df['choice'] == 1)) |
        ((df['probability_r'] < 0.5) & (df['choice'] == 0)))
    
    # ===== DATA AGGREGATION =====
    all_outcomes = []
    
    for subj in subjects:
        subj_df = df[df['subject'] == subj].reset_index(drop=True)
        switch_idx = subj_df.index[subj_df['switch_num'] == 1]
        
        for idx in switch_idx:
            # Skip events too close to edges
            if idx - window < 0 or idx + window >= len(subj_df):
                continue
            
            # Extract window around event
            window_df = subj_df.iloc[idx - window:idx + window + 1]
            
            for lag in range(-window, window + 1):
                all_outcomes.append({
                    'subject': subj,
                    'lag': lag,
                    'correct': window_df.iloc[lag + window]['correct'],
                    'is_event': (lag == 0)  # Mark the actual event time
                })
    
    outcome_df = pd.DataFrame(all_outcomes)
    
    # ===== VISUALIZATION =====
    fig, ax = plt.subplots(figsize=plot_params['figsize'])
    
    # Create color map for subjects
    colors = plt.cm.viridis(np.linspace(0, 1, len(subjects)))
    
    # Plot each subject's data
    for i, subj in enumerate(subjects):
        subj_data = outcome_df[outcome_df['subject'] == subj]
        mean_perf = subj_data.groupby('lag')['correct'].mean()
        sem_perf = subj_data.groupby('lag')['correct'].sem()
        
        ax.errorbar(
            x=mean_perf.index,
            y=mean_perf.values,
            yerr=sem_perf.values,
            fmt='o-',
            color=colors[i],
            label=f'Subject {subj}',
            linewidth=plot_params['linewidth'],
            capsize=plot_params['capsize']
        )
    
    # ===== PLOT FORMATTING =====
    # Reference lines
    ax.axhline(0.5, **plot_params['chance_line_style'], label='Chance')
    ax.axvline(0, **plot_params['event_line_style'], label=event_label)
    
    # Axis labels and limits
    ax.set_xlabel(f'Trial lag from {event_label.lower()}', 
                 size=plot_params['label_fontsize'])
    ax.set_ylabel('Fraction correct', size=plot_params['label_fontsize'])
    ax.set_ylim(0, 1)
    
    # Tick parameters
    ax.tick_params(axis='both', labelsize=plot_params['tick_fontsize'])
    
    # Legend - placed outside on right
    ax.legend(
        loc='lower left',
        fontsize=plot_params['legend_fontsize'],
        bbox_to_anchor=(1.01, 0),
        frameon=True,
        borderaxespad=0.
    )
    
    plt.tight_layout()
    plt.show()
    
    return outcome_df  # Return aggregated data for further analysis

def plot_blocks(df):
    """
    Plot the probability of reward in the right side across trials with presentation-friendly formatting.
    Includes chance level reference and optimized for slide readability.

    Args:
        df (DataFrame): Input dataframe containing trial data
    """
    # ===== VISUALIZATION PARAMETERS =====
    params = {
        'figure.figsize': (16, 8),          # Slightly wider aspect ratio for slides
        'axes.titlesize': 24,               # Larger title
        'axes.labelsize': 22,               # Larger axis labels
        'xtick.labelsize': 18,              # Larger x-tick labels
        'ytick.labelsize': 18,              # Larger y-tick labels
        'legend.fontsize': 18,              # Larger legend
        'grid.alpha': 0.3,                  # Slightly more visible grid
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0               # Slightly thicker axis lines
    }
    plt.rcParams.update(params)
    
    # ===== DATA PREPARATION =====
    df['trial'] = np.arange(len(df))
    df1 = df[(df['session'] == 129) & (df['subject'] == 'B1')]
    
    # Create figure and axis with constrained layout
    fig, ax = plt.subplots(constrained_layout=True)
    
    # Set pure white background
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    
    # ===== PLOT DATA =====
    # Main data line (thicker and more vibrant for visibility)
    ax.plot(df1['trial'].values, df1['probability_r'].values, 
            color='#1f77b4', linewidth=4, alpha=0.9, 
            label='Right reward probability')
    
    # Chance level reference line (more visible)
    ax.axhline(0.5, color='#7f7f7f', linestyle='--', linewidth=3, 
               alpha=0.8, label='Chance level')
    
    # ===== FORMATTING =====
    # Labels and title with increased padding
    ax.set_xlabel('Trial Number', labelpad=15)
    ax.set_ylabel('Reward Probability', labelpad=15)
    ax.set_title('Reward Probability Dynamics (Right Side)', pad=25)
    
    # Axis limits and ticks
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    # More visible grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.4)
    
    # Legend with stronger framing
    ax.legend(framealpha=1, facecolor='white', edgecolor='#cccccc', frameon=True)
    
    # Make spines slightly thicker for visibility
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)
    
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
    plot_blocks(new_df)
    plot_performance_comparison(new_df)
    plot_performance_psychometric(new_df)
    outcome_block_change(new_df,switch_type=1,window=10)