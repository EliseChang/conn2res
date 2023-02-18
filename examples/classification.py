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
dataframe_name = 'spatial_classification_v0_MEA-Mecp2_2022_dataset_conn2res_2023-02-15_unscaled.csv'  # name of previously generated .csv dataframe
plot_diagnostics = 0
plot_perf_curves = 1

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_2022_dataset_conn2res"
CONN_DATA = os.path.join(PROJ_DIR, 'data', 'connectivity', 'nulls_comparisons','2022_connected_randomised_randmio_1000')
NET_DATA = os.path.join(PROJ_DIR, 'data', 'network_metrics', 'MEA-Mecp2_2022_28Dec2022.csv')
EPHYS_DATA = os.path.join(PROJ_DIR, 'data', 'ephys_metrics','MEA-Mecp2_2022_23Dec2022.csv')
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

# Import distribution of extracellular potential, V, and spontaneous changes in this variable
# for modelling initial reservoir condition
voltage_distribution = pd.read_csv(os.path.join(PROJ_DIR, 'data', "culture_voltage_distribution.csv"),
                                   header=None).to_numpy()
noise_distribution = pd.read_csv(os.path.join(PROJ_DIR, 'data', "culture_noise_distribution.csv"),
                                 header=None).to_numpy()

# Get trial-based dataset for task
frac_train = 0.8

task_name = 'spatial_classification_v0'

if task_name == 'spatial_classification_v0':
    n_patterns = 2
    n_pres = 50  # number of presentations of each pattern
    n_timesteps = 200
    idx_washout = 50
    input_sf = 1
    kwargs = {'n_timesteps': n_timesteps,
            'washout': idx_washout,
            'ITI': 0,
            'input_shape': 'biphasic',
            'input_sf': input_sf,
            'voltage_distribution': (voltage_distribution[0, :], voltage_distribution[1, :]),
            'noise_distribution': (noise_distribution[0, :], noise_distribution[1, :])}

elif task_name == 'temporal_classification_v0':
    kwargs = {}

# Input node selection
input_patterns = np.array([[21], [10]])
input_set = np.concatenate(input_patterns)
# total number of nodes used for input across all patterns -- do not select any of these for output
n_input_nodes = len(input_set)

output_nodes = np.array([40, 42, 45, 49, 51])  # last column
n_output_nodes = len(output_nodes)

n_trials = n_patterns*n_pres
n_training_trials = round(n_pres*frac_train)  # for each pattern
n_testing_trials = round(n_pres*(1-frac_train))  # for each pattern

# Metrics to evaluate the regression model for RC output
metrics = ['score']
alphas = np.concatenate((np.linspace(0.2, 1.2, num=6), np.arange(
    2.0, 7.0, 1.0)))  # np.linspace(0.2, 3.0, num=15)

fig_num = 1

# %% Main

df_sample_ls = []  # initialise list to contain dictionaries for each alpha trial
ephys_data = pd.read_csv(EPHYS_DATA)
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

    conn.reduce()

    if conn.n_nodes != 59:
        input("Cannot implement network. Not fully connected. Press Enter to continue.")
        continue

    # scale conenctivity weights between [0,1] and normalize by spectral radius
    try:
        conn.scale_and_normalize()
    except ValueError:
        input(
            "Cannot compute largest eigenvalue. Check connectivity matrix. Press Enter to continue.")
        continue

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
            'genotype': genotypes[idx],
            'sample id': sample_id,
            # 'original_score': orig_scores[idx],
            'alpha': np.round(alpha, 3),
            'regime': regime,
            'rho': conn.spectral_radius}
        # 'edge weights': np.triu(conn.w * alpha)}

        # TRAINING
        rs = np.zeros((n_trials, conn.n_nodes))  # n_output_nodes))

        for trial in range(n_trials):

            # select stimulation pattern input nodes according to sequence
            stim_input_nodes = input_patterns[y[trial]]

            # create stimulation input connectivity matrix
            stim_in = np.zeros(
                (n_features, conn.n_nodes, n_timesteps), dtype=int)
            stim_in[:, stim_input_nodes, :] = np.ones(
                n_features, dtype=int)

            # create input matrix with random background input followed by input node stimulation
            w_in = np.concatenate((np.random.randint(2, size=(n_features, conn.n_nodes, idx_washout)),
                                stim_in), axis=2)
            input_nodes = conn.get_nodes(node_set='all')[w_in]

            # instantiate an Echo State Network object
            ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                            w_hh=conn.w * alpha,
                                            activation_function='tanh',
                                            #  input_gain=10.0,
                                            input_nodes=input_nodes,
                                            output_nodes=output_nodes
                                            )

            ic = ESN.set_initial_condition(**kwargs)

            # simulate reservoir states; returns states of all reservoir nodes
            # returns activity of all nodes
            rs_trial = ESN.simulate(
                ext_input=x, ic=ic, noise=True, **kwargs)

            rs_trial = ESN.add_washout_time(
                rs_trial, idx_washout=idx_washout)[0]  # return first item in tuple

            rs[trial, :] = np.mean(rs_trial, axis=0)

            # if trial == 0 or trial == 1:
            #     # plot reservoir activity TODO: only plot one raster for each pattern
            #     plotting.plot_time_series_raster(rs_trial, xlim=[0,len(rs_trial)], figsize=(19.2, 9.43),
            #     cmap_lim=[-1,1],title=f"{file}_reservoir_activity_trial_{trial}", savefig=True)

        # TESTING
        rs = rs[:, output_nodes]  # activity of just output nodes
        rs_train, rs_test = iodata.split_dataset(rs, frac_train=frac_train)

        scores = {m: [] for m in metrics}
        df_alpha, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                            target=(y_train, y_test),
                                            model=model,
                                            metric=metrics)

        # join alpha dictionary and df_alpha
        for m in metrics:
            alpha_dict[m] = df_alpha.at[0, m]
        
        df_sample_ls.append(alpha_dict | ephys_data.iloc[sample_id].to_dict() | network_data.iloc[sample_id].to_dict())

        if plot_diagnostics:

            figsize = (19.2, 9.43)

            plotting.plot_time_series(y_test, feature_set='data', xlim=[
                                    0, n_testing_trials*n_patterns], figsize=figsize, subplot=(1, 1, 1), legend_label='Target output', num=fig_num)
            plotting.plot_time_series(rs_test, feature_set='pred', xlim=[0, n_testing_trials*n_patterns], xticks=range(0, n_testing_trials*n_patterns), model=modelout,
                                    figsize=figsize, subplot=(1, 1, 1), legend_label='Predicted output', block=False, num=fig_num,
                                    savefig=True, title=f'{file}_{alpha}_performance')
            fig_num += 1

df_sample = pd.DataFrame(df_sample_ls)
df_sample.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_randomised.csv')
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
    # plot performance over trials for each sample at each alpha value
    for m in metrics:

        # plotting.plot_performance_curve(df_sample, y=m, ylim=[0, 1], xlim=[min(alphas), max(alphas)],
        #                                 hue="genotype", hue_order=["WT", "HE", "KO"],
        #                                 figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_input_sf_{input_sf}_performance_{m}', **kwargs)

        # plotting performance for unscaled networks
        plotting.plot_perf_reg(df_sample, x='rho',xlim=[0,45],ticks=[0,1,10,20,30,40],
                            hue="genotype", hue_order=["WT", "HE", "KO"], size='age',
        figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_{m}', **kwargs)

        # plotting.boxplot(df=df_sample, x='genotype', y=m, ylim=[0,1],chance_perf=0.5, order=["WT", "HE", "KO"],
        #                  figsize=(19.2/2, 9.43), width=0.2, savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_genotype_comparison', **kwargs)

        # # Percentage change in performance boxplots for rewired networks
        # # df_sample['percentage_change'] = (df_sample[m] - df_sample['original_score']) / df_sample['original_score']
        # plotting.boxplot(x='alpha', y='percentage_change', ylabel='Percentage score change upon randomisation', df=df_sample, hue='genotype', hue_order=["WT", "HE", "KO"],
        #      ylim=[-100,200], figsize=figsize, savefig=True, title='rewired_fully_connected_performance_change_boxplots')

        # # plot rho distribution
        # plotting.boxplot(x='age', y='rho', ylabel='rho', df=df_sample, hue='genotype', hue_order=["WT", "HE", "KO"],
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

        # # # linear regression model for performance as a function of network and electrophysiological variables
        # network_vars = ["density_full"] #, "net_size", "n_mod","small_world_sigma","small_world_omega"] # ["density_full", "density_full", "net_size", "n_mod","small_world_sigma","small_world_omega"]
        # ephys_vars = ['NBurstRate'] #, 'meanspikes', 'FRmean','meanNumChansInvolvedInNbursts','meanNBstLengthS', 'meanISIWithinNbursts_ms', 'CVofINBI']
        # var_names = ["Density", "Network burst rate (Hz)"]

        # for (v, var_name) in zip(network_vars + ephys_vars, var_names):
        #     plotting.plot_perf_reg(df_sample, x=v,y=m,xlabel=var_name,hue='genotype',
        #     size='age',savefig=True, title=f'{task_name}_{dataset}_{m}_perf_vs_{v}',**kwargs)

# %%
