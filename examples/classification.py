# -*- coding: utf-8 -*-
"""
MEA functional connectome-informed reservoir (Echo-State Network)
=================================================
"""

# %% Imports

# from reservoirpy.nodes import Reservoir, Ridge
import os
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, Ridge
from conn2res import reservoir, coding, plotting, iodata, task
import pandas as pd
import numpy as np
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
dataframe_name = '' # name of previously generated .csv dataframe
plot_diagnostics = 0
plot_perf_curves = 1

# Optional methods
rewire = 0
if rewire:
    itr = 50
    mode = 'random' # degrees
    lattice = False

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_2022_dataset_conn2res"
CONN_DATA = os.path.join(PROJ_DIR, 'data', 'connectivity')
NET_DATA = os.path.join(PROJ_DIR, 'data', 'network_metrics', 'All2022NetworkMetricsReduced.csv')
# EPHYS_DATA = os.path.join(PROJ_DIR, 'data', 'ephys_metrics','MEA-Mecp2_2022_23Dec2022.csv')
METADATA = f"{dataset}.xlsx"

# Today's date for saving objects
today = date.today()

# Import metadata
metadata = pd.read_excel(os.path.join(PROJ_DIR, 'data', METADATA),
                         sheet_name="Sheet1", engine="openpyxl") # fully_connected
names = metadata["File name"]
ages = metadata["Age"]
genotypes = metadata["Genotype"]
samples = metadata["Sample ID"]

# # For dataset with corresponding in vitro data
# pattern_0_idx = metadata["Hub input node index"]
# pattern_1_idx = metadata["Peripheral input node index"]
# real_scores = metadata["Real score"]

# Get trial-based dataset for task
frac_train = 0.8

task_name = 'spatial_classification_v0'

if task_name == 'spatial_classification_v0':
    n_patterns = 2
    n_pres = 60  # number of presentations of each pattern
    trial_duration = 305
    stim_duration = 5
    pre_stim_washout = 49
    post_stim = trial_duration - stim_duration - pre_stim_washout
    stim_onset = pre_stim_washout + 1
    stim_offset = pre_stim_washout + stim_duration
    input_sf = 3
    kwargs = {'trial_duration': trial_duration,
              'stim_duration': stim_duration,
              'post_stim': post_stim,
            'washout': pre_stim_washout,
            'input_shape': 'monophasic',
            'input_sf': input_sf,
            'noise': True}

elif task_name == 'temporal_classification_v0':
    kwargs = {}

# Input node selection
input_patterns = np.array([[19], [9]]) # [[17, 19, 20, 22, 24], [4, 6, 8, 9, 11]]
input_set = np.concatenate(input_patterns) # total number of nodes used for input across all patterns -- do not select any of these for output
n_input_nodes = len(input_set)
n_channels = 59
# output_nodes = [n for n in range(n_channels) if n not in input_set]
# n_output_nodes = len(output_nodes)

output_nodes = np.array([38, 40, 43, 44, 47, 49])
output_col_n = 8
n_output_nodes = len(output_nodes) # number of nodes in each set defined below

n_trials = n_patterns*n_pres
n_training_trials = round(n_pres*frac_train)  # for each pattern
n_testing_trials = round(n_pres*(1-frac_train))  # for each pattern

alphas = np.linspace(0.2, 2.0, num=10)
    #np.concatenate(np.arange(
    # 2.0, 10.0, 1.0))) # np.linspace(0.2, 2.0, num=10) np.array([1])

fig_num = 1

# %% Main

df_sample_ls = []  # initialise list to contain dictionaries for each alpha trial
# ephys_data = pd.read_csv(EPHYS_DATA)
network_data = pd.read_csv(NET_DATA)

# generate and fetch data
x = iodata.fetch_dataset(task_name, **kwargs)
y_train = iodata.random_pattern_sequence(n_patterns, n_training_trials)
y_test = iodata.random_pattern_sequence(n_patterns, n_testing_trials)
y = np.concatenate((y_train, y_test))

# number of features in task data
n_features = x.shape[1]

# define model for RC ouput
# LogisticRegression()
model = RidgeClassifier(alpha=1e-8, fit_intercept=True)

for idx, file in names.items():

    sample_id = samples[idx]

    # Import connectivity matrix
    print(
        f'\n*************** file = {file} ***************')
    conn = reservoir.Conn(
        w=None, conn_data=f'{file}.csv', conn_data_dir=CONN_DATA)

    # conn.reduce()
    # size = conn.n_nodes

    # if conn.n_nodes != 59:
    #     input(f"Cannot implement network. Not fully connected: only {conn.n_nodes} nodes. Press Enter to continue.")
    #     continue
        
    # scale connectivity weights between [0,1] and normalize by spectral radius
    try:
        conn.scale_and_normalize()
    except ValueError:
        input(
            "Cannot compute largest eigenvalue. Check connectivity matrix. Press Enter to continue.")
        continue

    if rewire:
        conn.rewire(itr, mode, lattice)

    # # For dataset with corresponding in vitro data
    # # Input node selection
    # input_patterns = np.array([[pattern_0_idx[idx]], [pattern_1_idx[idx]]]) # [[17, 19, 20, 22, 24], [4, 6, 8, 9, 11]]
    # input_set = np.concatenate(input_patterns)
    # n_input_nodes = len(input_set)

    # exclude_nodes = [pattern_0_idx[idx], pattern_1_idx[idx]] # input_set
    # output_nodes = [n for n in range(n_channels) if n not in exclude_nodes]
    # n_output_nodes = len(output_nodes)

    for alpha in alphas:
        if 0.2 <= alpha < 0.8:
            regime = 'stable'
        elif 0.8 <= alpha < 1.4:
            regime = 'critical'
        else:
            regime = 'chaotic'

        print(
            f'\n----------------------- alpha = {alpha} -----------------------')
        

        alpha_dict = {
            'name': file,
            'age': ages[idx],
            'Genotype': genotypes[idx],
            'sample id': samples[idx],
            'alpha': np.round(alpha, 3),
            'regime': regime}
        
        
        rs = np.zeros((n_trials, n_channels))

        for trial in range(n_trials):
            
            pattern = y[trial]
            pattern_0 = True
            pattern_1 = True

            # TRAINING

            # select stimulation pattern input nodes according to sequence
            w_in = np.random.normal(scale=1e-2, size=(n_features, conn.n_nodes, trial_duration))
            stim_input_nodes = input_patterns[pattern]
            w_in[:,stim_input_nodes,stim_onset:stim_offset] = np.ones(
                n_features, dtype=int)

            # instantiate an Echo State Network object
            ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                            w_hh=conn.w * alpha,
                                            activation_function='tanh')

            ic = ESN.set_initial_condition(**kwargs)

            # simulate reservoir states; returns states of all reservoir nodes
            # returns activity of all nodes
            rs_trial = ESN.simulate(
                ext_input=x, ic=ic, leak=0.75, **kwargs)
            
            # if plot_diagnostics:
            #     if (pattern_1 and pattern) or (pattern_0 and not pattern):
            #         plotting.plot_time_series_raster(rs_trial, num=fig_num, title=f'{file}_alpha_{np.round(alpha, 3)}_trial_{trial}_pattern_{pattern}.png',
            #                         savefig=True, **kwargs)
            #         if pattern: pattern_1 = False
            #         if not pattern: pattern_0 = False
            #         fig_num += 1

            rs_trial = ESN.add_washout_time(
                rs_trial, idx_washout=stim_offset)[0]

            rs[trial, :] = np.mean(rs_trial, axis=0)

        # ROWS
        # output_sets = np.array([
        #     [20, 23, 28, 29, 34, 37],
        #     [18, 21, 26, 31, 36, 39],
        #     [15, 16, 25, 32, 41, 42],
        #     [13, 12, 3, 55, 46, 45],
        #     [10, 7, 2, 56, 51, 48],
        #     [8, 5, 0, 58, 53, 50],
        #     [6, 4, 1, 57, 54, 52]])

        # COLUMNS
        # output_sets = np.array([
        # [20, 18, 15, 13, 10, 8],
        # [23, 21, 16, 12, 7, 5],
        # [28, 26, 25, 3, 2, 0],
        # [29, 31, 32, 55, 56, 58],
        # [34, 36, 41, 46, 51, 53],
        # [37, 39, 42, 45, 48, 50],
        # [38, 40, 43, 44, 47, 49]])

        # TESTING
        # rs_output = np.delete(rs, input_set, axis=1)
        rs_output = rs[:, output_nodes]  # activity of just output nodes
        rs_train, rs_test = iodata.split_dataset(rs_output, frac_train=frac_train)
        df_alpha, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                            target=(y_train, y_test),
                                            model=model,
                                            metric=['score'])

        # join alpha dictionary and df_alpha
        alpha_dict['score'] = df_alpha.at[0, 'score']

        df_sample_ls.append(alpha_dict | network_data.iloc[idx].to_dict()) # | ephys_data.iloc[sample_id].to_dict())

df_sample = pd.DataFrame(df_sample_ls)
df_sample.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{today}_all_ages_all_samples.csv')
print("Dataframe saved.")

# %% Import dataframe
if import_dataframe:
    df_sample = pd.read_csv(os.path.join(
        PROJ_DIR, 'dataframes', dataframe_name))

# %% Plotting performance

if plot_perf_curves:
    regimes = ['stable','critical','chaotic']
    if not isinstance(genotypes, list):
        genotypes = genotypes.to_list()
    figsize = (19.2, 9.43)
    
    m = 'score'

    plotting.plot_performance_curve(df_sample, y=m, xlim=[min(alphas), max(alphas)], ylabel="Classification accuracy",chance_perf=0.5,
                                    hue="Genotype", hue_order=["WT", "HE", "KO"],figsize=(12, 8),format=['png','svg'],
                                    savefig=True, title=f'{task_name}_{dataset}_all_ages', **kwargs)
    
    genotypes = ["WT", "HE", "KO"]
    network_vars = ["density_full", "n_mod","mod_score","small_world_sigma","small_world_omega",
                    "mean_node_degrees"]
    ephys_vars = ['meanspikes','FRmean','NBurstRate', 'meanNumChansInvolvedInNbursts','meanNBstLengthS']

    for regime in regimes:
        df_regime = df_sample[df_sample.regime == regime]
        df_regime['Genotype'] = pd.Categorical(df_regime['Genotype'], genotypes)
        df_regime.sort_values(by='Genotype',inplace=True)

        plotting.plot_perf_reg(df_regime, x='score', hue_order=["WT", "HE", "KO"],style='age',
                               figsize=(8,8),chance_perf=None,ylabel="Density", 
            y='density',hue='Genotype', legend=False, savefig=True,format=['png','svg']
            ,title=f'perf vs density {regime} regime')

    # Find alpha value at which performance is max
    # new_sample_ls = []
    # for idx,file in names.iteritems():
    #     sample_data = df_sample[df_sample['name'] == file].reset_index()
    #     max_idx = np.argmax(sample_data['score'])
    #     alpha_max = sample_data.at[max_idx,'alpha']
    #     max_score = np.max(sample_data['score'])
    #     age = ages[idx]
    #     genotype = genotypes[idx]
    #     sample_id = samples[idx]
    #     rho = sample_data.at[0, 'rho']
    #     sample = {'name':file,
    #                 'alpha_max': alpha_max,
    #                 'score': max_score,
    #                 'rho': rho,
    #                 'genotype': genotype,
    #                 'age': age,
    #                 'sample id': sample_id}
    #     new_sample_ls.append(sample)
    # alpha_max_df = pd.DataFrame(new_sample_ls)

    # alpha_max_df.to_csv(
    # f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_scaled_max_score.csv')
    # print("Dataframe saved.")

    # plotting.plot_perf_reg(alpha_max_df, x='rho', size='age',
    #         y='alpha_max',hue='genotype', legend=False, savefig=True, title=f'{task_name}_{dataset}_alphamax_vs_rho')

    # Plot performance against network variables
    for v in network_vars + ephys_vars:
        plotting.plot_perf_reg(df_regime, x=v, hue_order=["WT", "HE", "KO"],figsize=(8,8),
            y='score',hue='Genotype', legend=True, savefig=True, title=f'{task_name}_{dataset}_perf_vs_{v}')

    
    # Plotting performance for unscaled networks
    # plotting.plot_perf_reg(df_sample, x='rho',ylim=[0,1.05],ylabel="Classification accuracy",
    #                     hue="genotype", hue_order=["WT", "HE", "KO"], size='age',
    # figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_{m}', **kwargs)

    # plotting.boxplot(df=df_sample, x='genotype', y=m, ylim=[0.42,1.05],order=["WT", "HE", "KO"], hue='genotype', hue_order=["WT", "HE", "KO"], legend=False,
    #                     figsize=(4.5, 8), ylab="classification accuracy", linewidth=3.0, chance_perf=0.5,savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_genotype_comparison', **kwargs)
    
    # plotting.parallel_plot(df_sample, 'score', 'original_score', c='rho', cb_norm='log',cmap='coolwarm',
    #     cb_label="log(rho)", xlabels=["Original", "Rescaled to alpha = 1"], ylabel="R squared", savefig=True,
    #     title='rescaled_fully_connected_performance_change_parallel_plot')
    
    # # Percentage change in performance boxplots for rewired networks
    # df_sample['percentage_change'] = (df_sample[m] - df_sample['original_score']) / df_sample['original_score']
    # plotting.boxplot(x='alpha', y='percentage_change', ylabel='Percentage score change upon randomisation', df=df_sample, hue='genotype', hue_order=["WT", "HE", "KO"],
    #      width=1.0, ylim=[-100,200], figsize=(4.5, 8), savefig=True, title='rewired_fully_connected_performance_change_boxplots_scaled')

    # # plot rho distribution
    # plotting.boxplot(x='age', y='rho', ylabel='rho', df=df_sample, hue='genotype', hue_order=["WT", "HE", "KO"],
    #     figsize=figsize, savefig=True, title=f'{dataset}_fully_connected_rho_age_genotype_distribution_boxplots')

    # # plot performance curve at each developmental age
    # plotting.plot_performance_curve(df_sample, by_age=True, y=m, ylim=[0,1], xlim=[min(alphas),max(alphas)], hue='genotype', hue_order=["WT", "HE", "KO"],
    # figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_performance_{m}_by_age',
    # **kwargs)

# %%
