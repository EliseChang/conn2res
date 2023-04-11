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
dataframe_name = '' # name of previously generated .csv dataframe
plot_diagnostics = 0
plot_perf_curves = 1

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_2022_dataset_conn2res"
CONN_DATA = os.path.join(PROJ_DIR, 'data', 'connectivity')
# NET_DATA = os.path.join(PROJ_DIR, 'data', 'network_metrics', 'MEA-Mecp2_2022_28Dec2022.csv')
# EPHYS_DATA = os.path.join(PROJ_DIR, 'data', 'ephys_metrics','MEA-Mecp2_2022_23Dec2022.csv')
METADATA = f"{dataset}.xlsx"

# Today's date for saving objects
today = date.today()

# Import metadata
metadata = pd.read_excel(os.path.join(PROJ_DIR, 'data', METADATA),
                         sheet_name="fully_connected", engine="openpyxl")
names = metadata["File name"]
ages = metadata["Age"]
genotypes = metadata["Genotype"]
samples = metadata["Sample ID"] 
# orig_scores = metadata["Original score"]

# # Import distribution of extracellular potential, V, and spontaneous changes in this variable
# # for modelling initial reservoir condition
# voltage_distribution = pd.read_csv(os.path.join(PROJ_DIR, 'data', "culture_voltage_distribution.csv"),
#                                    header=None).to_numpy()
# noise_distribution = pd.read_csv(os.path.join(PROJ_DIR, 'data', "culture_noise_distribution.csv"),
#                                  header=None).to_numpy()

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
            'input_shape': 'biphasic',
            'input_sf': input_sf,
            'noise': True}
            # 'voltage_distribution': (voltage_distribution[0, :], voltage_distribution[1, :]),
            # 'noise_distribution': (noise_distribution[0, :], noise_distribution[1, :])}

elif task_name == 'temporal_classification_v0':
    kwargs = {}

# Input node selection
input_patterns = np.array([[17, 19, 20, 22, 24], [4, 6, 8, 9, 11]])
input_set = np.concatenate(input_patterns)
# total number of nodes used for input across all patterns -- do not select any of these for output
n_input_nodes = len(input_set)

# output_nodes = np.array([38, 40, 43, 44, 47, 49])
# output_col_n = 8
# n_output_nodes = len(output_nodes) # number of nodes in each set defined below

n_trials = n_patterns*n_pres
n_training_trials = round(n_pres*frac_train)  # for each pattern
n_testing_trials = round(n_pres*(1-frac_train))  # for each pattern

# Metrics to evaluate the regression model for RC output
# alphas = np.array([0.8]) # np.linspace(0.2, 2.0, num=10)
# np.concatenate((np.linspace(0.2, 1.2, num=6), np.arange(
#     2.0, 7.0, 1.0)))  # np.linspace(0.2, 3.0, num=15)

fig_num = 1

# %% Main

df_sample_ls = []  # initialise list to contain dictionaries for each alpha trial
# ephys_data = pd.read_csv(EPHYS_DATA)
# network_data = pd.read_csv(NET_DATA)

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

    # Import connectivity matrix
    print(
        f'\n*************** file = {file} ***************')
    conn = reservoir.Conn(
        w=None, conn_data=f'{file}.csv', conn_data_dir=CONN_DATA)

    conn.reduce()

    if conn.n_nodes != 59:
        input("Cannot implement network. Not fully connected. Press Enter to continue.")
        continue

    # # scale conenctivity weights between [0,1] and normalize by spectral radius
    # try:
    #     conn.scale_and_normalize()
    # except ValueError:
    #     input(
    #         "Cannot compute largest eigenvalue. Check connectivity matrix. Press Enter to continue.")
    #     continue

    # for alpha in alphas:
    #     if 0.2 <= alpha < 0.8:
    #         regime = 'stable'
    #     elif 0.8 <= alpha < 1.4:
    #         regime = 'critical'
    #     else:
    #         regime = 'chaotic'

    #     print(
    #         f'\n----------------------- alpha = {alpha} -----------------------')

    alpha_dict = {
        'name': file,
        'age': ages[idx],
        'genotype': genotypes[idx],
        'sample id': samples[idx],
        # 'alpha': np.round(alpha, 3),
        # 'regime': regime,
        'rho': conn.spectral_radius}
        
    rs = np.zeros((n_trials, conn.n_nodes))

    pattern_0 = True
    pattern_1 = True

    # TRAINING
    for trial in range(n_trials):

        pattern = y[trial]

        # select stimulation pattern input nodes according to sequence
        w_in = np.random.normal(scale=1.7416e-2, size=(n_features, conn.n_nodes, trial_duration))
        stim_input_nodes = input_patterns[pattern]
        w_in[:,stim_input_nodes,stim_onset:stim_offset] = np.ones(
            n_features, dtype=int)

        # instantiate an Echo State Network object
        ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                        w_hh=conn.w,# * alpha,
                                        activation_function='tanh')

        ic = ESN.set_initial_condition(**kwargs)

        # simulate reservoir states; returns states of all reservoir nodes
        # returns activity of all nodes
        rs_trial = ESN.simulate(
            ext_input=x, ic=ic, leak=1.0, **kwargs)
        
        if plot_diagnostics:
            if (pattern_1 and pattern) or (pattern_0 and not pattern):
                plotting.plot_time_series_raster(rs_trial, num=fig_num, title=f'{file}_alpha_{np.round(alpha, 3)}_trial_{trial}_pattern_{pattern}.png',
                                savefig=True, **kwargs)
                if pattern: pattern_1 = False
                if not pattern: pattern_0 = False
                fig_num += 1

        rs_trial = ESN.add_washout_time(
            rs_trial, idx_washout=pre_stim_washout)[0] # +stim_duration # return first item in tuple

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
    rs_output = rs #[:, output_nodes]  # activity of just output nodes
    rs_train, rs_test = iodata.split_dataset(rs_output, frac_train=frac_train)

    # if plot_diagnostics:
    #     plotting.plot_time_series_raster(rs_output, num=fig_num, title=f'{file}_activity_readout.png', # _alpha_{np.round(alpha, 3)}
    #                     savefig=True, **kwargs)
    #     fig_num += 1

    df_alpha, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                        target=(y_train, y_test),
                                        model=model,
                                        metric=['score'])
    
    # join alpha dictionary and df_alpha
    if df_alpha.at[0, 'score'] >= 0.5:
        alpha_dict['score'] = df_alpha.at[0, 'score']
    else:
        alpha_dict['score'] = 0.5

    df_sample_ls.append(alpha_dict) # | network_data.iloc[sample_id].to_dict() | ephys_data.iloc[sample_id].to_dict())

    # if plot_diagnostics:

    #     figsize = (19.2, 9.43)

    #     plotting.plot_time_series(y_test, feature_set='data', xlim=[
    #                             0, n_testing_trials*n_patterns], figsize=figsize, subplot=(1, 1, 1), legend_label='Target output', num=fig_num)
    #     plotting.plot_time_series(rs_test, feature_set='pred', xlim=[0, n_testing_trials*n_patterns], xticks=range(0, n_testing_trials*n_patterns), model=modelout,
    #                             figsize=figsize, subplot=(1, 1, 1), legend_label='Predicted output', block=False, num=fig_num,
    #                             savefig=True, title=f'{file}_{alpha}_performance')
    #     fig_num += 1

df_sample = pd.DataFrame(df_sample_ls)
df_sample.to_csv(
    f'{PROJ_DIR}/dataframes/{today}_all_networks_unscaled_5_inputs_6_outputs_5_stim_post_stim.csv')
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
    kwargs = {'ages': np.sort(ages.unique()),
            'regimes': regimes,
            'genotypes': genotypes}
    
    m = 'score'

    # plotting.plot_performance_curve(df_sample, y=m, ylim=[0, 1], xlim=[min(alphas), max(alphas)],
    #                                 hue="genotype", hue_order=["WT", "HE", "KO"],figsize=figsize,
    #                                 savefig=True, title=f'{task_name}_{dataset}_all_networks_scaled_5_inputs_6_outputs_5_stim_post_stim', **kwargs)

    # plotting.plot_line_plot(df_sample, x='output_col_n', title="Performance vs input-output distance", y='score', ylabel='R squared', xlim=None, ticks=None, ylim=[0,1], xlabel="Output column number",
    #               hue='genotype', hue_order=["WT", "HE", "KO"], chance_perf=0.5,
    #               figsize=(19.2, 9.43),savefig=True, show=False, block=True)
    
    # plotting performance for unscaled networks
    plotting.plot_perf_reg(df_sample, x='rho',ylim=[0,1],
                        hue="genotype", hue_order=["WT", "HE", "KO"], size='age',
    figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_{m}', **kwargs)

    plotting.boxplot(df=df_sample, x='genotype', y=m, ylim=[0,1],chance_perf=0.5, order=["WT", "HE", "KO"], hue='genotype', hue_order=["WT", "HE", "KO"], legend=False,
                        figsize=(4.5, 8), width=1.0, savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_genotype_comparison', **kwargs)
    
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

    # plot genotype comparisons for dynamical regimes at each developmental age
    # re-group
    regime_data_ls = []

    for regime in regimes:
        regime_data = df_sample[df_sample['regime'] == regime].reset_index().groupby('name').mean()
        regime_data['genotype'] = genotypes
        regime_data['regime'] = regime

        # plotting.plot_perf_reg(regime_data, x='density_full', y=m, xlabel="Density",hue='genotype', figsize=(9.43, 9.43),
        # size='age',savefig=True, title=regime,**kwargs)

        regime_data_ls.append(regime_data)

    df_regrouped = pd.concat(regime_data_ls)
    plotting.boxplot(x='regime', y=m, df=df_regrouped, by_age=True, hue='genotype', hue_order=["WT", "HE", "KO"],
        ylim=[0,1], figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_connected_scaled_5_inputs_all_outputs_120_trials_boxplots',**kwargs)

    # # # examine trends at individual alpha values/regimes
    # alpha_data = df_sample[df_sample['alpha'] == 1.0].reset_index()

    # # plot change in performance over development
    # plotting.plot_performance_curve(alpha_data, x='age', y=m, ylim=[0,1], xlim=[min(ages),max(ages)], ticks=ages.unique(),
    # hue='genotype', hue_order=["WT", "HE", "KO"], figsize=figsize, savefig=True, block=False,
    # title=f'{task_name}_{dataset}_performance_across_development_{m}',
    # **kwargs)

    # # linear regression model for performance as a function of network and electrophysiological variables
    # network_vars = ["density_full", "n_mod"] #, "net_size", "n_mod","small_world_sigma","small_world_omega"] # ["density_full", "density_full", "net_size", "n_mod","small_world_sigma","small_world_omega"]
    # ephys_vars = ['FRmean', 'NBurstRate'] #, 'NBurstRate', 'meanspikes', 'FRmean','meanNumChansInvolvedInNbursts','meanNBstLengthS', 'meanISIWithinNbursts_ms', 'CVofINBI']
    # var_names = ["Density", "Number of modules", "Mean firing rate","Network burst rate"]

    # for (v, var_name) in zip(network_vars + ephys_vars, var_names):
    #     plotting.plot_perf_reg(df_sample, x=v,y=m,xlabel=var_name,hue='genotype',
    #     size='age',savefig=True, title=f'{task_name}_{dataset}_{m}_perf_vs_{v}',**kwargs)

# %%
