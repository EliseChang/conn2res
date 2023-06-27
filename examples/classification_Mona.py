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
dataframe_name = ''
plot_diagnostics = 0
plot_perf_curves = 1
threshold = 0

state_var = 'spike_rates'
window = '20-100'

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_Mona_stim_dataset_conn2res"
MEA_DATA = os.path.join(PROJ_DIR, 'data', 'MEA', dataset, state_var, window)
METADATA = f"{dataset}.xlsx"

# Today's date for saving objects
today = date.today()

# Import metadata
metadata = pd.read_excel(os.path.join(PROJ_DIR, 'data', METADATA),
                         sheet_name='OWT', engine="openpyxl")
names = metadata["Pair name"]
sample_ids = metadata["Sample ID"]
baseline_recs = metadata["Baseline"]
pattern_0_recs = metadata["Pattern 0"]
pattern_1_recs = metadata["Pattern 1"]
ages = metadata["Age"]
genotypes = metadata["Genotype/Treat"]
protocol = metadata["Stimulation amplitude"]

# Get trial-based dataset for task
frac_train = 0.8

task_name = 'spatial_classification'

# Get pattern data
n_patterns = 2
n_pres = 60  # 60 number of presentations of each pattern
# pattern_0_idx = [20,21,23, 19, 25, 24, 22]
# pattern_1_idx = [9,10,8,6,7,5,4]
pattern_0_idx = metadata["Hub input node index"]
pattern_1_idx = metadata["Peripheral input node index"]
pattern_0_coord = metadata["Hub input node coord"]
pattern_1_coord = metadata["Peripheral input node coord"]
kwargs = {}

# output_nodes = np.array([39, 41, 44, 45, 48, 50]) # [0,1,5,6]
# n_output_nodes = 6 #len(output_nodes)

n_trials = n_patterns*n_pres
n_training_trials = round(n_pres*frac_train)  # for each pattern
n_testing_trials = round(n_pres*(1-frac_train))  # for each pattern
n_channels = 60

# Split into early and late

n_run = 100 # random training-testing split

fig_num = 1
figsize = (19.2, 9.43)

# %% Main

df_sample_ls = []  # initialise list to contain dictionaries for each alpha trial

# define model for RC ouput
model = RidgeClassifier(alpha=1e-8, fit_intercept=True)

# file = "OWT220207_1H_DIV57_3UA_STIM"
# idx = 0

for idx,file in names.items():

    print(
    f'\n*************** file = {file} ***************')

    # fetch data
    net_response_0 = pd.read_csv(os.path.join(MEA_DATA, f'{pattern_0_recs[idx]}.csv'),header=None).to_numpy()
    net_response_1 = pd.read_csv(os.path.join(MEA_DATA, f'{pattern_1_recs[idx]}.csv'),header=None).to_numpy()
    rs_control = pd.read_csv(os.path.join(MEA_DATA, f'{baseline_recs[idx]}.csv'),header=None).to_numpy()

    # Exclude stimulation and reference nodes
    if state_var == 'spike_rates':
        exclude_nodes = [pattern_0_idx[idx], pattern_1_idx[idx], 14]
        rs_nodes = [n for n in range(n_channels) if n not in exclude_nodes]
        net_response_0 = np.delete(net_response_0, exclude_nodes, axis=1)
        net_response_1 = np.delete(net_response_1, exclude_nodes, axis=1)
        rs_control = np.delete(rs_control, exclude_nodes, axis=1)

    # # Get input node coords
    # input_coords = np.array([[int(u) for u in str(pattern_0_coord[idx])],
    #                         [int(u) for u in str(pattern_1_coord[idx])]])
    # output_nodes = random.sample(rs_nodes, n_output_nodes)
    # io_dist = iodata.io_dist(input_coords, output_nodes)

    # net_response_0 = net_response_0[:,output_nodes]
    # net_response_1 = net_response_1[:,output_nodes]
    # rs_control = rs_control[:,output_nodes]

        # Initialise variable for storing model weights
        stim_weights = np.zeros((n_run,len(rs_nodes))) # n_output_nodes
        control_weights = np.zeros((n_run,len(rs_nodes))) #  len(rs_nodes)

    for run in range(0,n_run):

        # # randomly select output nodes and calculate input-output distance
        # output_nodes = random.sample(rs_nodes, n_output_nodes)
        # io_dist = iodata.io_dist(input_coords, output_nodes)

        # randomly select testing trials
        testing_trial_idx_0 = random.sample(range(n_pres), n_testing_trials)
        testing_trials_0 = net_response_0[testing_trial_idx_0,:]

        testing_trial_idx_1 = random.sample(range(n_pres), n_testing_trials)
        testing_trials_1 = net_response_1[testing_trial_idx_1,:]

        rs_test = np.concatenate((testing_trials_0, testing_trials_1), axis=0)
        y_test = np.concatenate((np.zeros(n_testing_trials,dtype='int64'),
                                np.ones(n_testing_trials,dtype='int64')), axis=0)

        training_trials_0 = np.delete(net_response_0, testing_trial_idx_0, axis=0)
        training_trials_1 = np.delete(net_response_1, testing_trial_idx_1, axis=0)
        rs_train = np.concatenate((training_trials_0, training_trials_1), axis=0)
        y_train = np.concatenate((np.zeros(n_training_trials,dtype='int64'), np.ones(n_training_trials,dtype='int64')), axis=0)

        print("Stimulation")
        df_stim, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                        target=(y_train, y_test),
                                        model=model,
                                        metric=['score'])
        if threshold:
            if df_stim.at[0, 'score'] < 0.5: df_stim['score'] = 0.5 # threshold

        df_stim['name'] = file
        df_stim['sample'] = sample_ids[idx]
        df_stim['age'] = ages[idx]
        df_stim['genotype'] = genotypes[idx]
        # df_stim['Treatment'] = treatment[idx]
        df_stim['Protocol'] = protocol[idx]
        df_stim['run'] = run
        df_stim['group'] = "stimulation"
        # df_stim['io_dist'] = io_dist
        df_sample_ls.append(df_stim)

        # stim_weights[run,:] = modelout.coef_

        # Split control data
        np.random.shuffle(rs_control)
        rs_train,rs_test = iodata.split_dataset((rs_control), frac_train=frac_train)
        # rs_test_run = rs_test[:,output_nodes]
        # rs_train_run = rs_train[:,output_nodes]

        print("Control")
        df_control, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                        target=(y_train, y_test),
                                        model=model,
                                        metric=['score'])
        if threshold:
            if df_control.at[0, 'score'] < 0.5: df_control['score'] = 0.5 # threshold
        df_control['name'] = f'{file}_CONTROL'
        df_control['sample'] = sample_ids[idx]
        df_control['age'] = ages[idx]
        df_control['genotype'] = genotypes[idx]
        # df_control['Treatment'] = treatment[idx]
        df_control['Protocol'] = protocol[idx]
        df_control['run'] = run
        df_control['group'] = "control"
        # df_stim['io_dist'] = io_dist
        df_sample_ls.append(df_control)

        # control_weights[run,:] = modelout.coef_

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
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_{state_var}_{window}.csv')
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

        plotting.boxplot('group', 'score', sample_data, width=0.5, figsize=(4, 9), savefig=True, ylim=[0,1], xticklabs=["Stimulation", "Control"], ylab="Classification accuracy", chance_perf=0.5,
                            title=f'{names[idx]}_{state_var}_classification_accuracy.png')
        
    #     # stim_data = sample_data[sample_data['group'] == 'stimulation'].reset_index()
        
        # plotting.plot_perf_reg(df=stim_data,x='io_dist',ylabel='Classification accuracy',ylim=[0,1],
        #                        xlabel='Mean input-output distance',savefig=True,
        #                        title=f'{names[idx]}_{state_var}_input-output_dist.png')
        
    # Plot classification accuracy against control for all samples
    df_temp = df_sample.groupby('name').mean().reset_index()
    df_temp['group'] = np.tile(["stimulation", "control"], len(sample_ids))
    genotypes_col = genotypes.repeat(2).reset_index() # label = 'Genotype'
    # treatments_col = treatment.repeat(2).reset_index()
    protocol_col = protocol.repeat(2).reset_index() # label = 'Stimulation amplitude'
    df_regrouped = pd.concat([df_temp,genotypes_col,protocol_col],axis=1) # treatments_col

    # df_regrouped.to_csv(
    # f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_{state_var}_regrouped.csv')
    # print("Dataframe saved.")

    # Temporary: set a different default palette for non-genotype comparisons to avoid confusion
    import seaborn as sns
    cp = sns.color_palette()
    # # WT untreated samples only
    # df_untreated = df_regrouped[df_regrouped['Treatment'] == 'Untreated']
    plotting.boxplot(x='group', y='score', df=df_regrouped, figsize=(8, 8), savefig=True, ylim=[0,1], hue='group', hue_order=["stimulation","control"],
                     xticklabs=["Stimulation", "Control"],ylab="Classification accuracy", chance_perf=0.5, palette=cp,
                            title=f'all_samples_{state_var}_{window}_classification_accuracy')
    

    # Group by genotype or treatment
    # # Age
    # # plotting.boxplot(x='group', y='score', df=df_regrouped, figsize=(6, 8), savefig=True, ylim=[0,1], hue='age',
    # #                     xticklabs=["Stimulation", "Control"],ylab="Classification accuracy", chance_perf=0.5,
    # #                             title=f'all_samples_{state_var}_classification_accuracy_by_age.png')

    # # Genotype
    # plotting.boxplot(x='group', y='score', df=df_regrouped, figsize=(8, 8), savefig=True, ylim=[0,1.05], hue='Genotype',hue_order=["WT","HET","KO"],
    #                  xticklabs=["Stimulation", "Control"],ylab="Classification accuracy", chance_perf=0.5,
    #                         title=f'all_samples_{state_var}_classification_accuracy_by_genotype.png')
    
    # # # Treatment
    # # plotting.boxplot(x='group', y='score', df=df_regrouped, figsize=(8, 8), savefig=True, ylim=[0,1.1], hue='Treatment',hue_order=["Untreated", "Gabazine"],
    # #                     xticklabs=["Stimulation", "Control"],ylab="Classification accuracy", chance_perf=0.5,
    # #                             title=f'all_samples_{state_var}_classification_accuracy_by_treatment.png')
    
    # # Stimulation protocol
    # plotting.boxplot(x='group', y='score', df=df_regrouped, figsize=(8, 8), savefig=True, ylim=[0,1.1], hue='Stimulation amplitude',hue_order=['1UA','3UA','4UA','5UA','6UA'],
    #                     xticklabs=["Stimulation", "Control"],ylab="Classification accuracy", chance_perf=0.5,
    #                             title=f'all_samples_{state_var}_classification_accuracy_by_protocol.png')
    
    # %% Calculate sample mean scores

    # Plot coefficient of variation
    df_mean = df_sample.groupby('name').mean()
    mean_score = df_mean['score']
    std_score = df_sample.groupby('name').std()['score']
    df_mean['cv'] = std_score / mean_score
    df_mean['group'] = np.tile(["stimulation", "control"], len(sample_ids))

    df_mean.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_{state_var}_mean_scores.csv')
    print("Dataframe saved.")

    # plotting.plot_perf_reg(df=df_mean,x='cv',y='score',hue='sample',style='group',legend=False,figsize=(8,8),
    #                        ylabel='Mean classification score',xlabel='CV across training-testing runs',
    #                        title=f'all_samples_{state_var}_classification_accuracy_CV',savefig=True)
    
#     plotting.boxplot(df=df_sample,x='window',y='score',hue='group',savefig=True, xlabel='Classification window timing (ms)',
# xticklabs=['3-20','20-100','100-180','180-260'],ylab='Classification accuracy',title="Classification accuracy ~ window timing")

## %%
# %%
