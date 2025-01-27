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
from extra_plotting import *





if __name__ == '__main__':
    data_path = '/home/marcaf/TFM(IDIBAPS)/codes/data/global_trials_maybe_updated.csv'
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype={'iti_duration': float})
    # shift iti_duration to the next trial
    df['iti_duration'] = df['iti_duration'].shift(1)
    # get only trials with iti
    #df = df[df['task'] != 'S4']
    df = df[df['subject'] != 'manual']
    prob_rwd = 0.99
    prob_switch = 0.5
    new_df = df[['subject','session', 'outcome', 'side', 'iti_duration','probability_r']]
    new_df = new_df.copy()
    new_df['outcome_bool'] = np.where(new_df['outcome'] == "correct", 1, 0)

    inference_plot(prob_switch,prob_rwd,new_df)