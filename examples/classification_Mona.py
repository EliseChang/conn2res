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
import_dataframe = 1
dataframe_name = 'Mona_classification_MEA-Mecp2_Mona_stim_dataset_conn2res_2023-03-29_spike_rates_all_treatments_all_output_nodes.csv'  # name of previously generated .csv dataframe
plot_diagnostics = 0
plot_perf_curves = 1

state_var = 'spike_rates'

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_Mona_stim_dataset_conn2res"
MEA_DATA = os.path.join(PROJ_DIR, 'data', 'MEA', dataset, state_var)
METADATA = f"{dataset}.xlsx"

# Today's date for saving objects
today = date.today()

# Import metadata
metadata = pd.read_excel(os.path.join(PROJ_DIR, 'data', METADATA),
                         sheet_name=state_var, engine="openpyxl")
names = metadata["Stimulation pair name"]
sample_ids = metadata["Sample ID"]
baseline_recs = metadata["Baseline file name"]
pattern_0_recs = metadata["Hub stimulation file name"]
pattern_1_recs = metadata["Peripheral stimulation file name"]
ages = metadata["Age"]
treatment = metadata["Treatment aggregated"]

# Get trial-based dataset for task
frac_train = 0.8

task_name = 'Mona_classification'

# Get pattern data
n_patterns = 2
n_pres = 60  # number of presentations of each pattern
pattern_0_inputs = metadata["Hub input node index"]
pattern_1_inputs = metadata["Peripheral input node index"]
kwargs = {}

output_nodes = np.array([35, 40, 43, 44]) # [0,1,5,6]
n_output_nodes = len(output_nodes)
n_trials = n_patterns*n_pres
n_training_trials = round(n_pres*frac_train)  # for each pattern
n_testing_trials = round(n_pres*(1-frac_train))  # for each pattern
n_channels = 60

n_run = 100 # random training-testing split

# Metrics to evaluate the regression model for RC output

fig_num = 1
figsize = (19.2, 9.43)

# %% Main

df_sample_ls = []  # initialise list to contain dictionaries for each alpha trial

# define model for RC ouput
model = RidgeClassifier(alpha=1e-8, fit_intercept=True)

# file = "OWT220207_1H_DIV57_3UA_STIM"
# idx = 0
for idx,file in names.items():
            
    # Exclude stimulation and reference nodes
    exclude_nodes = [pattern_0_inputs[idx], pattern_1_inputs[idx], 14]
    rs_nodes = [n for n in range(n_channels) if n not in exclude_nodes]

# for channel in range(59):
        
        # if channel not in exclude_nodes:

    print(
    f'\n*************** file = {file} ***************')

    # fetch data
    net_response_0 = pd.read_csv(os.path.join(MEA_DATA, f'{pattern_0_recs[idx]}.csv'),header=None).to_numpy()
    net_response_1 = pd.read_csv(os.path.join(MEA_DATA, f'{pattern_1_recs[idx]}.csv'),header=None).to_numpy()
    # net_response_0 = net_response_0[:,output_nodes]
    # net_response_1 = net_response_1[:,output_nodes]

    rs_control = pd.read_csv(os.path.join(MEA_DATA, f'{baseline_recs[idx]}.csv'),header=None).to_numpy()    
    # rs_control = rs_control[:,output_nodes]
    rs_control = np.delete(rs_control, exclude_nodes, axis=1)

    stim_weights = np.zeros((n_run,np.size(rs_control,axis=1)))
    control_weights = np.zeros((n_run,np.size(rs_control,axis=1)))

    for run in range(0,n_run):
            
        # randomly select testing trials
        testing_trial_idx_0 = random.sample(range(n_pres), n_testing_trials)
        testing_trials_0 = net_response_0[testing_trial_idx_0,:]

        testing_trial_idx_1 = random.sample(range(n_pres), n_testing_trials)
        testing_trials_1 = net_response_1[testing_trial_idx_1,:]

        rs_test = np.concatenate((testing_trials_0, testing_trials_1), axis=0)
        y_test = np.concatenate((np.zeros(n_testing_trials,dtype='int64'), np.ones(n_testing_trials,dtype='int64')), axis=0)
        rs_test = np.delete(rs_test, exclude_nodes, axis=1)

        training_trials_0 = np.delete(net_response_0, testing_trial_idx_0, axis=0)
        training_trials_1 = np.delete(net_response_1, testing_trial_idx_1, axis=0)
        rs_train = np.concatenate((training_trials_0, training_trials_1), axis=0)
        y_train = np.concatenate((np.zeros(n_training_trials,dtype='int64'), np.ones(n_training_trials,dtype='int64')), axis=0)
        rs_train = np.delete(rs_train, exclude_nodes, axis=1)

        print("Stimulation")
        df_stim, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                        target=(y_train, y_test),
                                        model=model,
                                        metric=['score'])
        df_stim['name'] = file
        df_stim['sample'] = sample_ids[idx]
        df_stim['age'] = ages[idx]
        df_stim['Treatment'] = treatment[idx]
        df_stim['run'] = run
        df_stim['group'] = "stimulation"
        df_sample_ls.append(df_stim)

        stim_weights[run,:] = modelout.coef_

        # Fetch control data
        # rs_control = rs_control[:, output_nodes]
        
        np.random.shuffle(rs_control)
        rs_train,rs_test = iodata.split_dataset((rs_control), frac_train=frac_train)

        print("Control")
        df_control, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                        target=(y_train, y_test),
                                        model=model,
                                        metric=['score'])
        df_control['name'] = f'{file}_CONTROL'
        df_control['sample'] = sample_ids[idx]
        df_control['age'] = ages[idx]
        df_control['Treatment'] = treatment[idx]
        df_control['run'] = run
        df_control['group'] = "control"
        df_sample_ls.append(df_control)

        control_weights[run,:] = modelout.coef_

    # Plot diagnostics
    if plot_diagnostics:

        # Channel weights
        if state_var == 'spike_rates':
            plotting.barplot(stim_weights,rs_nodes,fig_num,figsize=figsize, subplot=(2,1,1),
            savefig=False)
            plotting.barplot(control_weights,rs_nodes,fig_num,figsize=figsize, subplot=(2,1,2),
            savefig=True,name=f'{file}_model_weights')

        # Time-step weights
        if state_var == 'spike_counts':
            plotting.plot_time_series(stim_weights,figsize=figsize, subplot=(1, 1, 1),
            legend_label="Stimulation", savefig=False, **kwargs)
            plotting.plot_time_series(control_weights,figsize=figsize, subplot=(1, 1, 1),
            legend_label="Control", savefig=True, fname=f'{file}_model_weights',**kwargs)

        fig_num += 1

df_sample = pd.concat(df_sample_ls)
df_sample.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_{state_var}_all_treatments_random_output_nodes.csv')
print("Dataframe saved.")

# %% Import dataframe
if import_dataframe:
    df_sample = pd.read_csv(os.path.join(
        PROJ_DIR, 'dataframes', dataframe_name))

# %% Plotting performance

if plot_perf_curves:
        
    # Plot classification accuracy against control for each sample
    for idx,sample in sample_ids.items():

        sample_data = df_sample[df_sample['sample'] == sample].reset_index()

        plotting.boxplot('group', 'score', sample_data, width=0.5, figsize=(4, 9), savefig=True, ylim=[0,1], xticklabs=["Stimulation", "Control"], ylabel="Classification accuracy", chance_perf=0.5,
                            title=f'{names[idx]}_{state_var}_classification_accuracy.png')
        
    # Plot classification accuracy against control for each sample
    df_regrouped = df_sample.groupby('name').mean()
    df_regrouped['group'] = np.tile(["stimulation", "control"],len(sample_ids))
    plotting.boxplot(x='group', y='score', df=df_regrouped, width=0.5, figsize=(4, 9), savefig=True, ylim=[0,1], xticklabs=["Stimulation", "Control"],ylabel="Classification accuracy", chance_perf=0.5,
                            title=f'all_samples_{state_var}_classification_accuracy.png')
# %%