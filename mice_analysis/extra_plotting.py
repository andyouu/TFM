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


def psychometric(x):
    y = np.exp(x)
    return y/(1+y)


def plotting(evidence):
    plt.hist(evidence, bins=30, edgecolor='black')  # 30 bins for the histogram
    plt.title('Histogram of the evidence in 100 bins')
    plt.xlabel('Trial bins')
    plt.ylabel('log(Odds)')
    plt.show()
    psychometric_values = psychometric(evidence)
    plt.plot(evidence, psychometric_values)
    plt.show()



def psychometric_data(df_glm_mice, GLM_df,regressors_string):
    #we will first compute the evidence:
    regressors_vect = regressors_string.split(' + ')
    coefficients = GLM_df['coefficient']
    n = len(df_glm_mice['r_plus_1'])
    evidence = np.zeros(n)
    df_glm_mice['evidence'] = 0
    for j in range(len(regressors_vect)):
        df_glm_mice['evidence']+= coefficients[regressors_vect[j]]*df_glm_mice[regressors_vect[j]]
    plotting(df_glm_mice['evidence'])
    

