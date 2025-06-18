#Mice Analysis

##Descriptive Statistics
###Script Overview

The script (`descriptive_statistics.py`) processes behavioral data from a CSV file and generates four types of visualizations to analyze performance across subjects (mice) in a reward-based task. The main functions are:
#### `plot_performance_comparison`

- **Purpose**: Compares performance between good and poor performers using a boxplot for good performers and scattered points for poor performers.
- **Inputs**:
  - `df_data`: DataFrame with columns `subject` (subject identifier) and `perf` (performance values, 0-1).
  - `chance_level`: Threshold to classify good/poor performers (default: 0.55).
- **Features**:
  - Boxplot for good performers (performance ≥ chance_level) with styled whiskers, caps, and medians.
  - Jittered scatter points for both good and poor performers, with distinct colors (blue for good, red for poor).
  - Reference line at the chance level.
  - Custom legend and statistical annotation (Mann-Whitney U test p-value) if both groups exist.
  - Large fonts and high contrast for presentation readability.
  
#### `plot_performance_psychometric`

- **Purpose**: Plots psychometric curves showing performance as a function of reward probability for each subject and the group average.
- **Inputs**:
  - `combined_data`: DataFrame with columns `subject`, `perf`, and `probability_r` (probability of reward for right choice).
- **Features**:
  - Individual subject curves with performance plotted against reward probability, excluding the 0.5 probability condition.
  - Group average curve in black for emphasis.
  - Ideal performance line (y=x) and chance level references (horizontal and vertical at 0.5).
  - `viridis` color palette for subjects, with a custom legend placed outside the plot.
  - Handles invalid performance values (`-1`) and transforms left-side probabilities to right-equivalent for consistency.
  
#### `outcome_block_change`

- **Purpose**: Analyzes and visualizes performance around block transitions (changes in reward probability) or choice switches, showing trial-by-trial correctness.
- **Inputs**:
  - `df`: DataFrame with columns `subject`, `choice` (0=left, 1=right), and `probability_r`.
  - `window`: Number of trials to show before/after the event.
  - `switch_type`: 0 for block transitions, 1 for choice switches.
- **Features**:
  - Plots performance (fraction correct) aligned to switch events for each subject, with SEM error bars.
  - Identifies switch points based on changes in probability (`switch_type=0`) or choice (`switch_type=1`).
  - Includes reference lines for the event (vertical at lag 0) and chance level (horizontal at 0.5).
  - Uses `viridis` colormap for subjects, with a legend placed outside the plot.
  - Returns an aggregated DataFrame for further analysis.
  
#### `plot_blocks`

- **Purpose**: Visualizes the reward probability for the right choice across trials for a specific session and subject.
- **Inputs**:
  - `df`: DataFrame with columns `session`, `subject`, `trial`, and `probability_r`.
- **Features**:
  - Plots reward probability dynamics for a single session (hardcoded: session 129, subject 'B1').
  - Includes a chance level reference line at 0.5.
  - Optimized for slides with a clean design: Arial font, thicker lines, visible grid, and no top/right spines.
  - High-contrast colors and large fonts for readability.

#### Main Execution

- **Data Loading**:
  - Reads a CSV file (`global_trials1.csv`) with behavioral data, using `;` as the separator and specifying `iti_duration` as float.
  - Applies a custom `parsing` function to filter data (e.g., for trained mice, `trained=1`).
- **Data Preprocessing**:
  - Creates a `choice` column based on `outcome` and `side` (correct/incorrect and left/right).
  - Computes a `perf` column (1 for correct choices based on `probability_r`, 0 otherwise).
- **Visualization**:
  - Calls all four plotting functions in sequence: `plot_blocks`, `plot_performance_comparison`, `plot_performance_psychometric`, and `outcome_block_change` (with `switch_type=1` and `window=10`).
  
### Usage

1. Ensure the required CSV file (`global_trials1.csv`) is available at the specified path.
2. Verify that the `extra_plotting`, `model_avaluation`, and `parsing` modules are accessible.
3. Update the `data_path` variable if needed.
5. The script will generate and display four plots:
   - Reward probability dynamics for a specific session/subject.
   - Performance comparison between good and poor performers.
   - Psychometric curves for all subjects.
   - Performance around choice switches.
### Notes

- The script assumes a specific CSV structure with columns like `subject`, `session`, `outcome`, `side`, `probability_r`, and `iti_duration`. Ensure your data matches this format.
- The `plot_blocks` function is hardcoded for session 129 and subject 'B1'; modify the filtering condition (`df[(df['session'] == 129) & (df['subject'] == 'B1')]`) for other sessions/subjects.
- The `parsing` function is not provided; it must handle data filtering for trained/untrained mice.
- Visualization parameters (e.g., font sizes, figure dimensions) are optimized for presentations but can be adjusted in the `params` dictionaries.


##Model Performance Visualization
###Script overview
The script (`metrics_plots.py`) is designed to load previously stored model performance data from CSV files and create comparative visualizations grouped by models. It includes two main functions:

####`plot_metrics_comparison_grouped_by_models`

- **Purpose**: Generates five separate plots to compare model performance across different metrics: Log Likelihood, Number of Trials, BIC (Bayesian Information Criterion), Log Likelihood per Observation, and Accuracy.
- **Inputs**:
  - `blocks`: A list of experimental conditions (e.g., probability blocks like `[[0.2,0.8], [0.3,0.7]]`).
  - `metrics_data`: A nested dictionary containing metric values for each model (e.g., `{model_name: {metric_name: [values_across_blocks]}}`).
  - `model_names`: A list of model identifiers (e.g., `['glm_prob_switch', 'glm_prob_r']`).
- **Features**:
  - Uses a consistent color scheme (`viridis` palette) across all plots.
  - Employs large, readable fonts suitable for presentations.
  - Automatically adjusts layout to prevent overlap.
  - Each plot visualizes one metric, grouping data by models and coloring by experimental conditions.

#### `plot_metric_grouped_by_models`

- **Purpose**: Creates a single grouped comparison plot for a specific metric, using boxplots and stripplots to show distribution and individual data points.
- **Inputs**:
  - `blocks`: Same as above.
  - `metrics_data`: Same as above.
  - `model_names`: Same as above.
  - `metric`: The specific metric to plot (e.g., `'log_likelihood'`).
  - `ylabel`: Y-axis label for the plot.
  - `palette`: Color palette for different blocks.
  - Font size parameters for title, labels, legend, and ticks.
- **Features**:
  - Boxplots show the distribution of metric values per model and block.
  - Stripplots overlay individual data points with slight jitter for clarity.
  - Custom legend placement outside the plot to avoid overlap.
  - Special handling for model-specific block labeling (e.g., different block indices for `inference_based` vs. other models).
  - Grid and despined axes for clean visualization.
  
####Main Execution

- **Data Loading**:
  - Reads CSV files containing model metrics (e.g., `all_subjects_glm_metrics_{model}_{block}.csv`).
  - Aggregates data by taking the median across subjects for each metric.
  - Handles missing files or errors by appending empty arrays.
- **Model Configuration**:
  - Supports multiple models (e.g., `glm_prob_switch`, `glm_prob_r`, `inference_based`).
  - Uses different block configurations for standard and special models.
- **Visualization**:
  - Calls `plot_metrics_comparison_grouped_by_models` to generate plots for selected models.
  - Includes error handling to report issues during plot generation.

### Usage

1. Place the script in a directory containing the required CSV files with model metrics.

2. Update the `main_folder` and `results_base` paths in the script to match your file locations.

3. Modify `models_to_analyze` and `models_to_plot` to include the desired models.

4. The script will generate and display five plots comparing the specified metrics across models.

### Notes

- Ensure CSV files follow the expected format with columns for `subject`, `log_likelihood`, `log_likelihood_per_obs`, `BIC`, `AIC`, and `accuracy`.
- The script assumes median aggregation per subject; modify the `df.groupby('subject').median()` line to use all data points if needed.

Font sizes and figure dimensions are optimized for presentations but can be adjusted in the function parameters.

##Inference-Based Script
### Script Overview

The script (`inference-based.py`) processes behavioral data from a CSV file and includes functions to compute inference-based features (`V_t`) and visualize logistic regression coefficients for predicting mouse choices. The main functions are:

#### `manual_computation`

- **Purpose**: Processes trial-by-trial behavioral data to compute value differences (`V_t`) and choice-reward sequences based on past trials.
- **Inputs**:
  - `df`: DataFrame with columns like `subject`, `session`, `outcome`, `side`, `probability_r`.
  - `n_back`: Number of previous trials to consider for sequence patterns.
  - `hist`: Boolean flag to plot a histogram of computed `V_t` values.
- **Features**:
  - Encodes choices (0=left, 1=right) based on `outcome` and `side`.
  - Creates choice-reward codes (`00`, `01`, `10`, `11`) combining choice and reward outcome.
  - Builds sequences of `n_back` previous choice-reward pairs.
  - Computes `V_t` as the difference between the mean probability of the right side being active (`prob_right`) and the left side being active (`prob_left`) for each sequence.
  - Optionally plots a histogram of `V_t` values.
  - Prepares next-trial choice data for modeling.
- **Returns**: Processed DataFrame with computed features (`choice`, `sequence`, `V_t`, etc.).

#### `manual_computation_v2`

- **Purpose**: Computes `V_t` using recursive equations based on reward history and given probabilities, suitable for an alternative inference model.
- **Inputs**:
  - `df`: DataFrame with trial data.
  - `p_SW`: Probability of switching from active to inactive state.
  - `p_RWD`: Probability of reward in the active state.
  - `hist`: Boolean flag to plot a histogram of `V_t` values.
- **Features**:
  - Encodes choices similarly to `manual_computation`.
  - Tracks same-site choices across trials.
  - Computes `R_t` (reward history) recursively:
    - Resets to 0 on reward.
    - Updates using `rho = 1 / ((1 - p_SW) * (1 - p_RWD))` for same-site unrewarded trials.
    - Maintains previous `R_t` for exploratory switches.
  - Computes `V_t` based on `R_t`, `side_num`, and `p_RWD`, with warnings for values exceeding 1.
  - Optionally plots a histogram of `V_t` values.
- **Returns**: DataFrame with computed `R_t`, `V_t`, and other intermediate columns.

#### `plot_all_mice_correct_inf_combined`

- **Purpose**: Creates a single plot showing logistic regression coefficients for all mice, optimized for A0 poster presentation.
- **Inputs**:
  - `df`: DataFrame with behavioral data for all mice.
  - `n_back`: Number of previous trials for sequence patterns.
  - `figsize`: Figure size (default: A0 poster size, 46.8x33.1 inches).
- **Features**:
  - Excludes mouse 'A10' from analysis.
  - Uses 5-fold cross-validation to fit logistic regression models (`choice ~ V_t + side_num` or `choice ~ side_num` for `v2=2`).
  - Computes coefficients, standard errors, p-values, and confidence intervals.
  - Calculates comprehensive metrics (log-likelihood, AIC, BIC, pseudo R-squared, accuracy, precision, recall, F1, ROC AUC, Brier score).
  - Saves metrics to a CSV file (path based on `v2` and `n_back`).
  - Plots coefficients (`β^V`, `β^S`, Intercept) with distinct colors and significance markers (`***`, `**`, `*`, `ns`) based on Stouffer's combined p-values.
  - Uses an alpha gradient to differentiate mice in the legend.
  - Optimized for poster presentation with large fonts and thick lines.
  
#### `inference_plot`

- **Purpose**: Plots logistic regression coefficients either combined across all mice or separately for each mouse.
- **Inputs**:
  - `df`: DataFrame with behavioral data.
- **Features**:
  - Supports two modes:
    - Combined: Calls `plot_all_mice_correct_inf_combined` with `n_back=3` (hardcoded).
    - Separate: Creates a subplot grid (2 rows, dynamic columns) for individual mice.
  - For separate plots:
    - Filters sessions with >50 trials.
    - Uses `manual_computation` with `n_back=3` to compute `V_t`.
    - Fits logistic regression (`choice ~ V_t + side_num`) for each mouse using 5-fold cross-validation.
    - Plots median coefficients (Intercept, `side_num`, `V_t`) with significance markers.
    - Uses distinct colors for regressors and rotates x-labels if needed.
  - Shares y-axis across subplots for consistency.
  - Optimized for readability with dynamic subplot sizing.
  
#### Main Execution

- **Data Loading**:
  - Reads `global_trials1.csv` with `;` separator, specifying `iti_duration` as float.
  - Shifts `iti_duration` to the next trial and excludes `subject='manual'`.
- **Data Preprocessing**:
  - Selects relevant columns (`subject`, `session`, `outcome`, `side`, `iti_duration`, `probability_r`, `task`, `date`).
  - Encodes `outcome_bool` (1 for correct, 0 otherwise).
  - Applies a custom `parsing` function for trained mice (`trained=1`, `opto_yes=0`).
- **Visualization**:
  - Calls `inference_plot` to generate plots (default: separate plots for each mouse).
  
### Usage

1. Ensure the required CSV file (`global_trials1.csv`) is available at the specified path.
2. Verify that the `extra_plotting`, `model_avaluation`, and `parsing` modules are accessible.
3. Update the `data_path` variable if needed.
4. The script will generate and display either:
   - A single A0-sized plot with combined coefficients for all mice (if `separate_mice=False`).
   - A subplot grid with coefficients for each mouse (if `separate_mice=True`).

### Notes

- The script assumes a specific CSV structure with columns like `subject`, `session`, `outcome`, `side`, `probability_r`, `iti_duration`, `task`, and `date`.
- The `parsing` and `select_train_sessions` functions are not provided; they must handle data filtering and cross-validation split creation.
- The `v2` flag in `plot_all_mice_correct_inf_combined` controls the computation method (`manual_computation` or `manual_computation_v2`) and model formula.
- The `n_back=3` is hardcoded in `inference_plot` for combined plots; adjust as needed.
- Visualization parameters (e.g., font sizes, figure dimensions) are optimized for presentations/posters but can be modified in `plt.rcParams` or function arguments.
- The script excludes mouse 'A10' and sessions with ≤50 trials in `inference_plot`.


