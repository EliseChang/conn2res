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
plt.ioff() # turn off interactive mode
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %% Set global variables

# Metaparameters
import_dataframe = 0
dataframe_name = '' # 'mackey_glass_MEA-Mecp2_2020_0.95_thr_50_runs_average.csv' # name of previously generated .csv dataframe to import for plotting
reduce = 1
plot_diagnostics = 0
plot_perf_curves = 0

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_2020_dataset_conn2res"
CONN_DATA = os.path.join(PROJ_DIR, 'data', 'connectivity', dataset)
# NET_DATA = os.path.join(PROJ_DIR, 'data', 'network_metrics', 'MEA-Mecp2_2022_28Dec2022.csv')
# EPHYS_DATA = os.path.join(PROJ_DIR, 'data', 'ephys_metrics','MEA-Mecp2_2022_23Dec2022.csv')
METADATA = f"{dataset}.xlsx"

# Today's date for saving objects
today = date.today()

# Import metadata
metadata = pd.read_excel(os.path.join(PROJ_DIR, 'data', METADATA),
            sheet_name="0.95_thr", engine="openpyxl") # "0.95_thr"
names = metadata["File name"]
ages = metadata["Age"]
genotypes = metadata["Genotype"]

# Input node selection
n_input_nodes = 5
n_output_nodes = 5
nruns = 1

# Get trial-based dataset for task
task_name = 'mackey_glass'

if task_name == 'mackey_glass':
    tau = 30
    horizon = 30
    kwargs = {'n_timesteps': 4000, 'tau': tau, 'horizon': horizon}
elif task_name == 'MemoryCapacity':
    horizon = -25
    kwargs = {'n_timesteps': 4000, 'horizon': horizon}

idx_washout = 200

# Metrics to evaluate the regression model for RC output
metrics = ['score'] # , 'mse', 'nrmse'
alphas = np.array([1.0]) # np.linspace(0.2, 3.0, num=15) # np.array([0.2, 0.8, 1.0, 1.2, 2.0])

# %% Main

subj_ls = [] # initialise list to contain dictionaries for each alpha run
# ephys_data = pd.read_csv(EPHYS_DATA)
# network_data = pd.read_csv(NET_DATA)

for idx,file in names.iteritems():

    # Import connectivity matrix
    print(
            f'\n*************** file = {file} ***************')
    conn = reservoir.Conn(w=None, conn_data=f'{file}.csv', conn_data_dir=CONN_DATA)
    if reduce: conn.reduce()

    # if conn.n_nodes < n_input_nodes + n_output_nodes:
    #     input("Cannot implement network. Total nodes must be greater than specified input nodes.  Press Enter to continue.")
    #     continue
    # # scale conenctivity weights between [0,1] and normalize by spectral radius
    try:
        conn.scale_and_normalize()
    except ValueError:
        # input("Cannot compute largest eigenvalue. Check connectivity matrix. Press Enter to continue.")
        continue

    # # generate and fetch data
    # x, y = iodata.fetch_dataset(task_name, **kwargs)

    # # split trials into training and test sets
    # x_train, x_test, y_train, y_test = iodata.split_dataset(
    #     x, y, frac_train=0.75)

    # # number of features in task data
    # n_features = x_train.shape[1]

    # # define model for RC ouput
    # model = Ridge(alpha=1e-8, fit_intercept=True)

    for alpha in alphas:
        if 0.2 <= alpha < 0.8:
            regime = 'stable'
        elif 0.8 <= alpha < 1.4:
            regime = 'critical'
        else:
            regime = 'chaotic'

        print(
            f'\n----------------------- alpha = {alpha} -----------------------')

        # scores = {m:[] for m in metrics}

        # find rho
        from scipy.linalg import eigh
        ew, _ = eigh(conn.w)
        rho = np.abs(ew.max())

        alpha_dict = {
            'name': file,
            'age': ages[idx],
            'genotype': genotypes[idx],
            'size': conn.n_nodes,
            'rho': rho,
            'alpha': np.round(alpha, 3),
            'regime': regime}
        
        # for run in range(nruns):

        #     # we select a random set of input nodes
        #     if reduce:
        #         conn.reduce()
        #         alpha_dict['size'] = conn.n_nodes
            
        #     # else: # pass whole network but only select from connected nodes
        #     #     input_subset = conn.reduce()
        #     #     input_nodes = conn.get_nodes('random', n_nodes=n_input_nodes,nodes_from=input_subset)

        #     input_nodes = conn.get_nodes('random', n_nodes=n_input_nodes)
        #     output_nodes = conn.get_nodes('random', n_nodes=n_output_nodes, nodes_without=input_nodes)

        #     # create input connectivity matrix
        #     w_in = np.zeros((n_features, conn.n_nodes))
        #     w_in[:, input_nodes] = np.eye(n_features)
        
        #     # instantiate an Echo State Network object
        #     ESN = reservoir.EchoStateNetwork(w_ih=w_in,
        #                                         w_hh=conn.w * alpha,
        #                                         activation_function='tanh',
        #                                         #  input_gain=10.0,
        #                                         input_nodes=input_nodes,
        #                                         output_nodes=output_nodes
        #                                         )

        #     # simulate reservoir states; select only output nodes.
        #     rs_train = ESN.simulate(ext_input=x_train)
        #     rs_test = ESN.simulate(ext_input=x_test)

        #     # plot reservoir activity before washout
        #     # plotting.plot_time_series_raster(rs_train, xlim=[0,len(rs_train)], figsize=(19.2, 9.43), title=f"{file}_reservoir_activity_training",
        #     #                 savefig=True, block=True)
        #     # plotting.plot_time_series_raster(rs_test, xlim=[0,len(rs_test)], figsize=(19.2, 9.43), title=f"{file}_reservoir_activity_testing",
        #     #                 savefig=True, block=True)

        #     rs_train, rs_test, y_train2, y_test2 = ESN.add_washout_time(
        #         rs_train, rs_test, y_train, y_test, idx_washout=idx_washout)

        #     # perform task
        #     df_run, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
        #                                     target=(y_train2, y_test2),
        #                                     model=model,
        #                                     metric=metrics)

        #     for m in metrics: scores[m].append(df_run.at[0, m])

            # if plot_diagnostics:
            #     N = 200
            #     tau = 100
            #     figsize=(19.2, 9.43)
            #     sample = [0+horizon, N] if horizon > 0 else [0, N+horizon]
            #     plotting.plot_time_series(x_test[idx_washout+tau:], feature_set='data', xlim=[0, N], sample=sample,
            #                                 figsize=figsize, subplot=(3, 3, (1, 2)), legend_label='Input', block=False)
            #     plotting.plot_mackey_glass_phase_space(x_test[idx_washout:], x_test[idx_washout+tau:], xlim=[0.2, 1.4], ylim=[0.2, 1.4], color='magma', sample=sample,
            #                                             figsize=figsize, subplot=(3, 3, 3), block=False)

            #     sample = [0, N-horizon] if horizon > 0 else [-horizon, N]
            #     plotting.plot_time_series(y_test2[tau:N+tau-horizon], feature_set='data', xlim=[0, N], sample=sample,
            #                                 figsize=figsize, subplot=(3, 3, (4, 5)), legend_label='Label', block=False)
            #     plotting.plot_time_series(rs_test[tau:], feature_set='pred', xlim=[0, N], sample=sample,
            #                                 figsize=figsize, subplot=(3, 3, (4, 5)), legend_label='Predicted label', block=False, model=modelout)
            #     plotting.plot_mackey_glass_phase_space(modelout.predict(rs_test[0:]), modelout.predict(rs_test[0+tau:]), xlim=[0.2, 1.4], color='magma', sample=sample,
            #                                             ylim=[0.2, 1.4], figsize=figsize, subplot=(3, 3, 6), block=False)

            #     plotting.plot_time_series(rs_test[tau:N+tau-horizon], feature_set='pc', xlim=[0, N], normalize=True, idx_features=[1, 2, 3],
            #                                 figsize=figsize, subplot=(3, 3, (7, 8)), legend_label='Readout PC', block=False,
            #                                 savefig=True, fname=f'{task_name}_diagnostics_{file}_a{alpha}')
        
        # for m in metrics: alpha_dict[m] = np.mean(scores[m])
        subj_ls.append(alpha_dict) # | ephys_data.iloc[idx].to_dict() | network_data.iloc[idx].to_dict())

df_subj = pd.DataFrame(subj_ls)
df_subj.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_with_network_size.csv')
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

        plotting.plot_performance_curve(df_subj, y=m, ylim=[0,1], xlim=[min(alphas),max(alphas)], hue="genotype", hue_order=["WT", "HE", "KO"],
        figsize=figsize, savefig=True, legend=False, title=f'{task_name}_{dataset}_performance_{m}',**kwargs)

        # plot genotype comparisons at each developmental age
        plotting.plot_performance_curve(df_subj, by_age=True, y=m, ylim=[0,1], xlim=[min(alphas),max(alphas)], hue='genotype', hue_order=["WT", "HE", "KO"],
        figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_performance_{m}_across_development_{m}',
        **kwargs)

        # examine trends at peak performance (alpha = 1.0)
        alpha_data = df_subj[df_subj['alpha'] == 1.0].reset_index()

        # plot change in performance over development
        plotting.plot_performance_curve(alpha_data, x='age', y=m, ylim=[0,1], xlim=[min(ages),max(ages)], ticks=ages.unique(),
        hue='genotype', hue_order=["WT", "HE", "KO"], figsize=figsize, savefig=True, block=False, legend=False,
        title=f'{task_name}_{dataset}_performance_across_development_{m}',**kwargs)

        # linear regression model for performance as a function of network and electrophysiological variables
        network_vars = ['rho'] # ["density_full", "net_size", "n_mod","small_world_sigma","small_world_omega"]
        # ephys_vars = ['meanspikes', 'FRmean','NBurstRate', 'meanNumChansInvolvedInNbursts','meanNBstLengthS', 'meanISIWithinNbursts_ms', 'CVofINBI']
         
        for v in network_vars: #+ ephys_vars:
            plotting.plot_perf_reg(df_subj, x=v, xlim=[0,15], ylim=[0,1],ticks=[0,1,5,10,15],
                y=m,hue='genotype', legend=False, savefig=True, title=f'{task_name}_{dataset}_{m}_perf_vs_{v}')
# %%