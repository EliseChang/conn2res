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
import_dataframe = 1
dataframe_name = 'spatial_classification_v0_MEA-Mecp2_2022_dataset_conn2res_2023-02-14_1_input_sf.csv'  # name of previously generated .csv dataframe
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

# Input node selection
input_patterns = np.array([[45], [72]])
input_set = np.concatenate(input_patterns)
# total number of nodes used for input across all patterns -- do not select any of these for output
n_input_nodes = len(input_set)

output_nodes = np.array([24, 25, 28, 45, 48, 49])
n_output_nodes = len(output_nodes)

n_trials = n_patterns*n_pres
n_training_trials = round(n_pres*frac_train)  # for each pattern
n_testing_trials = round(n_pres*(1-frac_train))  # for each pattern

n_run = 10 # random trial order shuffling

# Metrics to evaluate the regression model for RC output
metrics = ['score']

fig_num = 1
figsize = (19.2, 9.43)

file = "OWT220207_1H_DIV57"

# %% Stimulation

df_sample_ls = []  # initialise list to contain dictionaries for each alpha trial

# define model for RC ouput
model = RidgeClassifier(alpha=1e-8, fit_intercept=True)

# fetch data

for run in range(0,n_run):

        rs_stim = pd.read_csv(os.path.join(MEA_DATA, f'{file}_STIMULATION_3UA.csv'),header=None).to_numpy()
        np.random.shuffle(rs_stim)
        y = list(rs_stim[:,-1])
        rs_stim = rs_stim[:,output_nodes]
        y_train,y_test,rs_train,rs_test = iodata.split_dataset(y, rs_stim, frac_train, axis=0)

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

# # join alpha dictionary and df_alpha
# for m in metrics:
#     alpha_dict[m] = df_alpha.at[0, m]

        # plotting.plot_time_series(y_test, feature_set='data', xlim=[
        #                         0, n_testing_trials*n_patterns], figsize=figsize, subplot=(1, 1, 1), legend_label='Target output', num=fig_num)
        # plotting.plot_time_series(rs_test, feature_set='pred', xlim=[0, n_testing_trials*n_patterns], xticks=range(0, n_testing_trials*n_patterns), model=modelout,
        #                         figsize=figsize, subplot=(1, 1, 1), legend_label='Predicted output', block=False, num=fig_num,
        #                         savefig=True, title=f'{file}_STIMULATION_3UA_diagnostic_plot')
        # fig_num += 1

# df_sample = pd.DataFrame(df_sample_ls)
# df_sample.to_csv(
#     f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_randomised.csv')
# print("Dataframe saved.")

# %% Baseline

        # fetch control data
        rs_control = pd.read_csv(os.path.join(MEA_DATA, f'{file}_BASELINE.csv'),header=None).to_numpy()
        np.random.shuffle(rs_control)

        rs_control = rs_control[:, output_nodes]
        rs_train,rs_test = iodata.split_dataset(rs_control, frac_train)

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

# # join alpha dictionary and df_alpha
# for m in metrics:
#     alpha_dict[m] = df_alpha.at[0, m]

# df_sample_ls.append(alpha_dict)

        # plotting.plot_time_series(y_test, feature_set='data', xlim=[
        #                         0, n_testing_trials*n_patterns], figsize=figsize, subplot=(1, 1, 1), legend_label='Target output', num=fig_num)
        # plotting.plot_time_series(rs_test, feature_set='pred', xlim=[0, n_testing_trials*n_patterns], xticks=range(0, n_testing_trials*n_patterns), model=modelout,
        #                         figsize=figsize, subplot=(1, 1, 1), legend_label='Predicted output', block=False, num=fig_num,
        #                         savefig=True, title=f'{file}_BASELINE_diagnostic_plot')
        # fig_num += 1

df_sample = pd.concat(df_sample_ls)
# df_sample = pd.DataFrame(df_sample_ls)
df_sample.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_{file}.csv')
print("Dataframe saved.")

# %% Import dataframe
if import_dataframe:
    df_sample = pd.read_csv(os.path.join(
        PROJ_DIR, 'dataframes', dataframe_name))

# # %% Plotting performance

# if plot_perf_curves:
#     regimes = ['stable','critical','chaotic']
#     genotypes = genotypes.to_list()
#     figsize = (19.2, 9.43)
#     kwargs = {'ages': np.sort(ages.unique()),
#               'regimes': regimes,
#               'genotypes': genotypes}
#     # plot performance over trials for each sample at each alpha value
#     for m in metrics:

        # plotting.plot_performance_curve(df_sample, y=m, ylim=[0, 1], xlim=[min(alphas), max(alphas)],
        #                                 hue="genotype", hue_order=["WT", "HE", "KO"],
        #                                 figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_input_sf_{input_sf}_performance_{m}', **kwargs)

        # # plotting performance for unscaled networks
        # plotting.plot_perf_reg(df_sample, x='rho',ylim=[0,1], ticks=[0,1,10,20,30,40],
        #                     hue="genotype", hue_order=["WT", "HE", "KO"],
        # figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_{m}', **kwargs)

        # Percentage change in performance boxplots for rewired networks
        # df_sample['percentage_change'] = (df_sample[m] - df_sample['original_score']) / df_sample['original_score']
       
        # plotting.boxplot(x='alpha', y='percentage_change', ylabel='Percentage score change upon randomisation', df=df_sample, hue='genotype', hue_order=["WT", "HE", "KO"],
        #      ylim=[-100,200], figsize=figsize, savefig=True, title='rewired_fully_connected_performance_change_boxplots')

        # # plot rho distribution
        # plotting.boxplot(x='age', y='rho', df=df_sample, hue='genotype', hue_order=["WT", "HE", "KO"],
        #     figsize=figsize, savefig=True, title=f'{dataset}_fully_connected_rho_age_genotype_distribution_boxplots')

        # # plot performance curve at each developmental age
        # plotting.plot_performance_curve(df_sample, by_age=True, y=m, ylim=[0,1], xlim=[min(alphas),max(alphas)], hue='genotype', hue_order=["WT", "HE", "KO"],
        # figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_performance_{m}_across_development_{m}',
        # **kwargs)

        # # plot genotype comparisons for dynamical regimes at each developmental age
        # # re-group
        # regime_data_ls = []
        # for regime in regimes:
        #     regime_data = df_sample[df_sample['regime'] == regime].reset_index().groupby('name').mean()
        #     regime_data['genotype'] = genotypes
        #     regime_data['regime'] = regime
        #     regime_data_ls.append(regime_data)
        # df_regrouped = pd.concat(regime_data_ls)
        # plotting.boxplot(x='regime', y=m, df=df_regrouped, by_age=True, hue='genotype', hue_order=["WT", "HE", "KO"],
        #     ylim=[0,1], figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_performance_{m}_boxplots',**kwargs)

        # # examine trends at individual alpha values/regimes
        # alpha_data = df_sample[df_sample['alpha'] == 1.0].reset_index()

        # # plot change in performance over development
        # plotting.plot_performance_curve(alpha_data, x='age', y=m, ylim=[0,1], xlim=[min(ages),max(ages)], ticks=ages.unique(),
        # hue='genotype', hue_order=["WT", "HE", "KO"], figsize=figsize, savefig=True, block=False,
        # title=f'{task_name}_{dataset}_performance_across_development_{m}',
        # **kwargs)

        # # linear regression model for performance as a function of network and electrophysiological variables
        # network_vars = ["density_full", "net_size", "n_mod","small_world_sigma","small_world_omega"] # ["density_full", "density_full", "net_size", "n_mod","small_world_sigma","small_world_omega"]
        # ephys_vars = ['meanspikes', 'FRmean','NBurstRate', 'meanNumChansInvolvedInNbursts','meanNBstLengthS', 'meanISIWithinNbursts_ms', 'CVofINBI']

        # for v in network_vars + ephys_vars:
        #     plotting.plot_perf_reg(df_sample, x=v, by_regime=True,
        #         y=m,hue='genotype',savefig=True, title=f'{task_name}_{dataset}_{m}_perf_vs_{v}',
        #         **kwargs)

# %%
