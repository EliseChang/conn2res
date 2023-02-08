# -*- coding: utf-8 -*-
"""
MEA functional connectome-informed reservoir (Echo-State Network)
=================================================
"""

# %% Imports

# from reservoirpy.nodes import Reservoir, Ridge
import os
from bct.algorithms.degree import strengths_und
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from conn2res import reservoir, coding, plotting, iodata, task
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import date
plt.ioff() # turn off interactive mode
# plt.ion() # turn on interactive mode
import networkx as nx
import scipy.io as sio
import mat73
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %% Set global variables

# Metaparameters
import_dataframe = 1
dataframe_name = 'mackey_glass_MEA-Mecp2_2022_full_2022-12-16_50_runs.csv' # name of previously generated .csv dataframe to import for plotting
reduce = 0
node_level = 0
plot_diagnostics = 0
plot_perf_curves = 1

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_2022"
CONN_DATA = os.path.join(PROJ_DIR, 'data', 'connectivity', dataset) # MEA-Mecp2_2020_24Dec2022
# NET_DATA = os.path.join(PROJ_DIR, 'data', 'network_metrics') TODO: save network metrics in same format as ephys data
EPHYS_DATA = os.path.join(PROJ_DIR, 'data', 'ephys_metrics','MEA-Mecp2_2022_23Dec2022.csv')
METADATA = "Mecp2_2022_dataset_conn2res.xlsx"

# Today's date for saving objects
today = date.today()

# Import metadata
metadata = pd.read_excel(os.path.join(PROJ_DIR, 'data', METADATA),
            sheet_name="Sheet1", engine="openpyxl") # sheet_name="0.9_thr" sheet_name="0.95_thr"
names = metadata["File name"]
ages = metadata["Age"]
genotypes = metadata["Genotype"]

# Get trial-based dataset for task
task_name = 'mackey_glass'
if task_name == 'PerceptualDecisionMaking':
    kwargs = {'timing': {'fixation': 0, 'stimulus': 2000,
                         'decision': 100}, 'cohs': np.array([51.2])}  # 0, 6.4, 12.8, 51.2
elif task_name == 'PerceptualDecisionMakingDelayResponse':
    kwargs = {'timing': {'fixation': 0, 'stimulus': 2000,
                         'decision': 100, 'delay': (200)}, 'cohs': np.array([51.2])}
elif task_name == 'ContextDecisionMaking':
    kwargs = {'timing': {'fixation': 0, 'stimulus': 2000,
                         'decision': 100, 'delay': 200}}
elif task_name == 'GoNogo':
    kwargs = {'timing': {'fixation': 0, 'stimulus': 1000,
                         'decision': 100, 'delay': 200}}
elif task_name == 'mackey_glass':
    tau = 17
    horizon = 10
    kwargs = {'n_timesteps': 4000, 'tau': tau, 'horizon': horizon}
elif task_name == 'MemoryCapacity':
    horizon = -25
    kwargs = {'n_timesteps': 4000, 'horizon': horizon}

idx_washout = 200

# Metrics to evaluate the regression model for RC output
metrics = ['score'] # , 'mse', 'nrmse'
alphas = np.array([0.2, 0.8, 1.0, 1.2, 2.0]) # 0.2, 0.8, 1.0, 1.2, 2.0

# Number of random input node selections
n_input_nodes = 5
nruns = 50

# %% Main

subj_ls = [] # initialise list to contain dictionaries for each alpha run
ephys_data = pd.read_csv(EPHYS_DATA)

for idx,file in names.iteritems():

    # Import connectivity matrix
    print(
            f'\n*************** file = {file} ***************')
    conn = reservoir.Conn(w=None, conn_data=f'{file}.csv', conn_data_dir=CONN_DATA)
    if reduce: conn.reduce()

    if conn.n_nodes <= n_input_nodes:
        input("Cannot implement network. Total nodes must be greater than specified input nodes.  Press Enter to continue.")
        continue
    # scale conenctivity weights between [0,1] and normalize by spectral radius
    try:
        conn.scale_and_normalize()
    except ValueError:
        # input("Cannot compute largest eigenvalue. Check connectivity matrix. Press Enter to continue.")
        continue

    # generate and fetch data
    x, y = iodata.fetch_dataset(task_name, **kwargs)

    # split trials into training and test sets
    x_train, x_test, y_train, y_test = iodata.split_dataset(
        x, y, frac_train=0.75)

    # number of features in task data
    n_features = x_train.shape[1]

    # define model for RC ouput
    model = Ridge(alpha=1e-8, fit_intercept=True)

    for alpha in alphas:
        print(
            f'\n----------------------- alpha = {alpha} -----------------------')

        scores = {m:[] for m in metrics}
        alpha_dict = {
            'index': idx,
            'name': file,
            'age': ages[idx],
            'genotype': genotypes[idx],
            'n_output_nodes': conn.n_nodes-n_input_nodes,
            'alpha': np.round(alpha, 3)}
        
        for run in range(nruns):
            # we select a random set of input nodes
            if reduce:
                input_nodes = conn.get_nodes('random', n_nodes=n_input_nodes)
            elif node_level: #TODO: define
                input_nodes = np.array([run])
            else:
                input_subset = conn.reduce()
                input_nodes = conn.get_nodes('random', nodes_from=input_subset) # pass whole network but only select from connected nodes

            output_nodes = conn.get_nodes('all', nodes_without=input_nodes)

            # create input connectivity matrix
            w_in = np.zeros((n_features, conn.n_nodes))
            w_in[:, input_nodes] = np.eye(n_features)
        
            # instantiate an Echo State Network object
            ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                                w_hh=alpha * conn.w,
                                                activation_function='tanh',
                                                #  input_gain=10.0,
                                                input_nodes=input_nodes,
                                                output_nodes=output_nodes
                                                )

            # simulate reservoir states; select only output nodes.
            rs_train = ESN.simulate(ext_input=x_train)
            rs_test = ESN.simulate(ext_input=x_test)

            rs_train, rs_test, y_train2, y_test2 = ESN.add_washout_time(
                rs_train, rs_test, y_train, y_test, idx_washout=idx_washout)

            # perform task
            df_run, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                            target=(y_train2, y_test2),
                                            model=model,
                                            metric=metrics)
            df_run['run'] = run
            #TODO: save df_run
            for m in metrics: scores[m].append(df_run.at[0, m])

            if plot_diagnostics:
                N = 200
                tau = 100
                figsize=(19.2, 9.43)
                sample = [0+horizon, N] if horizon > 0 else [0, N+horizon]
                plotting.plot_time_series(x_test[idx_washout+tau:], feature_set='data', xlim=[0, N], sample=sample,
                                            figsize=figsize, subplot=(3, 3, (1, 2)), legend_label='Input', block=False)
                plotting.plot_mackey_glass_phase_space(x_test[idx_washout:], x_test[idx_washout+tau:], xlim=[0.2, 1.4], ylim=[0.2, 1.4], color='magma', sample=sample,
                                                        figsize=figsize, subplot=(3, 3, 3), block=False)

                sample = [0, N-horizon] if horizon > 0 else [-horizon, N]
                plotting.plot_time_series(y_test2[tau:N+tau-horizon], feature_set='data', xlim=[0, N], sample=sample,
                                            figsize=figsize, subplot=(3, 3, (4, 5)), legend_label='Label', block=False)
                plotting.plot_time_series(rs_test[tau:], feature_set='pred', xlim=[0, N], sample=sample,
                                            figsize=figsize, subplot=(3, 3, (4, 5)), legend_label='Predicted label', block=False, model=modelout)
                plotting.plot_mackey_glass_phase_space(modelout.predict(rs_test[0:]), modelout.predict(rs_test[0+tau:]), xlim=[0.2, 1.4], color='magma', sample=sample,
                                                        ylim=[0.2, 1.4], figsize=figsize, subplot=(3, 3, 6), block=False)

                plotting.plot_time_series(rs_test[tau:N+tau-horizon], feature_set='pc', xlim=[0, N], normalize=True, idx_features=[1, 2, 3],
                                            figsize=figsize, subplot=(3, 3, (7, 8)), legend_label='Readout PC', block=False,
                                            savefig=True, fname=f'{task_name}_diagnostics_{file}_a{alpha}')
        
        for m in metrics: alpha_dict[m] = np.mean(scores[m])
        all_data = alpha_dict | ephys_data.iloc[idx].to_dict()
        subj_ls.append(all_data)

df_subj = pd.DataFrame(subj_ls)
df_subj.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_{nruns}_runs.csv')
print("Dataframe saved.")

#%% Import dataframe
if import_dataframe: 
    df_subj = pd.read_csv(os.path.join(PROJ_DIR, 'dataframes', dataframe_name))

# %% Plotting performance

if plot_perf_curves:
    figsize=(19.2, 9.43)
    kwargs = {'ages': np.sort(ages.unique()),
    'genotypes': genotypes.unique()}

    # plot performance over runs for each sample at each alpha value
    for m in metrics:

        # plot genotype comparisons for all ages grouped
        plotting.plot_performance_curve(
            df_subj, title=f'{task_name}_{dataset}_performance_{nruns}_runs', y=m, hue='genotype',
            hue_order=["WT", "HE", "KO"], figsize=figsize, savefig=True, show=True, block=False)

        # # plot genotype comparisons at each developmental age
        # plotting.plot_performance_curve(df_subj, by_age=True, y=m, hue='genotype', hue_order=["WT", "HE", "KO"],
        # figsize=figsize, savefig=True, block=False,
        # title=f'{task_name}_{dataset}_performance_across_development_{m}_{nruns}_runs',
        # **kwargs)

        # plot performance normalised by average spikes across active channels
        plotting.plot_performance_curve(df_subj, y=m, hue='genotype', hue_order=["WT", "HE", "KO"],
        norm=True, norm_method="channel_avg_spikes", figsize=figsize, savefig=True, block=False,
        title=f'{task_name}_{dataset}_performance_{m}_normalised_by_channel_average_spikes_{nruns}_runs',
        **kwargs)

        # # examine trends at peak performance (alpha = 1.0)
        # alpha_data = df_subj[df_subj['alpha'] == 1.0].reset_index()

        # # plot change in performance over development
        # plotting.plot_performance_curve(alpha_data, x='age', y=m, hue='genotype', hue_order=["WT", "HE", "KO"],
        # figsize=figsize, savefig=True, block=False,
        # title=f'{task_name}_{dataset}_performance_across_development_{m}_{nruns}_runs',
        # **kwargs)

        # plot linear regression model for performance as a function of network variables
        # network_vars = ["density","density_reduced", "aN"] # "aN", "Dens", "nMod", "PL", "SW", "SWw"
        # node_vars = ["NS", "aveControl", "modalControl", "BC", "PC"]
        ephys_vars = ['channelavgnumspikes']
        for v in ephys_vars:
            plotting.plot_perf_reg(df_subj, x=v, title=f'{task_name}_{dataset}_{m}_perf_vs_{v}_{nruns}_runs',
                y=m,hue='genotype',savefig=True)
        
        # # plot cumulative mean performance over runs at given alpha value(s)
        # for idx,file in names.iteritems():
        #     alpha_data['cum_avg_perf'] = alpha_data[m].expanding().mean()
        #     plotting.plot_performance_curve(alpha_data, title=f'{task_name}_{dataset}_{file}_{alpha}_performance_over_{nruns}_runs',
        #     x='run', y='cum_avg_perf',savefig = True)
# %%
