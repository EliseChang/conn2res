# -*- coding: utf-8 -*-
"""
MEA functional connectome-informed reservoir (Echo-State Network)
=================================================
"""

# %% Imports

# from reservoirpy.nodes import Reservoir, Ridge
import os
from bct.algorithms.degree import strengths_und
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, Ridge
from conn2res import reservoir, coding, plotting, iodata, task
import pandas as pd
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
from datetime import date
plt.ioff()  # turn off interactive mode
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %% Set global variables

# Metaparameters
import_dataframe = 0
dataframe_name = ''  # name of previously generated .csv dataframe
plot_diagnostics = 0
plot_perf_curves = 1

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_Mona_stim_dataset_conn2res"
MEA_DATA = os.path.join(PROJ_DIR, 'data', 'MEA', dataset)
# METADATA = f"{dataset}.xlsx"

# Today's date for saving objects
today = date.today()

# # Import metadata
# metadata = pd.read_excel(os.path.join(PROJ_DIR, 'data', METADATA),
#                          sheet_name="Sheet2", engine="openpyxl")
# names = metadata["File name"]
# ages = metadata["Age"]
# genotypes = metadata["Genotype"]

# Get trial-based dataset for task
frac_train = 0.8

task_name = 'Mona_classification'

n_patterns = 2
n_pres = 60  # number of presentations of each pattern

kwargs = {}

output_nodes = np.array([24, 23, 27, 44, 47, 48])
n_output_nodes = len(output_nodes)

n_trials = n_patterns*n_pres
n_training_trials = round(n_pres*frac_train)  # for each pattern
n_testing_trials = round(n_pres*(1-frac_train))  # for each pattern

n_run = 100 # random training-testing split

# Metrics to evaluate the regression model for RC output
metrics = ['score']

fig_num = 1
figsize = (19.2, 9.43)

file = "OWT220207_1H_DIV57"

# %% Stimulation

df_sample_ls = []  # initialise list to contain dictionaries for each alpha trial

# define model for RC ouput
model = RidgeClassifier(alpha=1e-8, fit_intercept=True)

for run in range(0,n_run):

        # fetch data
        net_response_0 = pd.read_csv(os.path.join(MEA_DATA, f'{file}_HUB45_3UA_cSpikes_L0_RP2.csv'),header=None).to_numpy()
        net_response_1 = pd.read_csv(os.path.join(MEA_DATA, f'{file}_PER72_3UA_cSpikes_L0_RP2.csv'),header=None).to_numpy()
        
        # randomly select testing trials
        testing_trial_idx_0 = random.sample(range(n_pres), n_testing_trials)
        testing_trials_0 = net_response_0[testing_trial_idx_0,:]

        testing_trial_idx_1 = random.sample(range(n_pres), n_testing_trials)
        testing_trials_1 = net_response_1[testing_trial_idx_1,:]

        rs_test = np.concatenate((testing_trials_0, testing_trials_1), axis=0)
        y_test = rs_test[:,-1]
        rs_test = rs_test[:,output_nodes]

        training_trials_0 = np.delete(net_response_0, testing_trial_idx_0, axis=0)
        training_trials_1 = np.delete(net_response_1, testing_trial_idx_1, axis=0)
        rs_train = np.concatenate((training_trials_0, training_trials_1), axis=0)
        y_train = rs_train[:,-1]
        rs_train = rs_train[:,output_nodes]

# for idx, file in names.items():

#     # sample_id = samples[idx]

#     # Import connectivity matrix
#     print(
#         f'\n*************** file = {file} ***************')

#     sample_dict = {
#         'name': file,
#         'age': ages[idx],
#         'genotype': genotypes[idx]}

# TESTING

# scores = {m: [] for m in metrics}

        df_stim, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                        target=(y_train, y_test),
                                        model=model,
                                        metric=metrics)
        df_stim['run'] = run
        df_stim['group'] = 'stimulation'
        df_sample_ls.append(df_stim)

# df_sample = pd.DataFrame(df_sample_ls)
# df_sample.to_csv(
#     f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_randomised.csv')
# print("Dataframe saved.")

# %% Baseline

        # fetch control data
        rs_control = pd.read_csv(os.path.join(MEA_DATA, f'{file}_BASELINE_cSpikes_L0_RP2.csv'),header=None).to_numpy()
        np.random.shuffle(rs_control)

        rs_control = rs_control[:, output_nodes]
        rs_train,rs_test = iodata.split_dataset((rs_control), frac_train=frac_train)

# for idx, file in names.items():

#     # sample_id = samples[idx]

#     # Import connectivity matrix
#     print(
#         f'\n*************** file = {file} ***************')

#     sample_dict = {
#         'name': file,
#         'age': ages[idx],
#         'genotype': genotypes[idx]}

        # TESTING

        # scores = {m: [] for m in metrics}

        df_control, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                        target=(y_train, y_test),
                                        model=model,
                                        metric=metrics)
        df_control['run'] = run
        df_control['group'] = 'control'
        df_sample_ls.append(df_control)

df_sample = pd.concat(df_sample_ls)
df_sample.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_{file}.csv')
print("Dataframe saved.")

# %% Import dataframe
if import_dataframe:
    df_sample = pd.read_csv(os.path.join(
        PROJ_DIR, 'dataframes', dataframe_name))

# %% Plotting performance

if plot_perf_curves:
    
    figsize = (19.2, 9.43)
    
    # plot performance over trials for each sample at each alpha value
    for m in metrics:

        plotting.boxplot('group', m, df_sample, width=0.5, figsize=(4, 9), savefig=True, ylim=[0,1], ylabel="Classification accuracy", chance_perf=0.5)