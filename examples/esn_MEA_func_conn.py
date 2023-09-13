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
dataframe_name = '' # name of previously generated .csv dataframe to import for plotting
reduce = 0
plot_diagnostics = 0
plot_perf_curves = 1

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_2022_dataset_conn2res"
CONN_DATA = os.path.join(PROJ_DIR, 'data', 'connectivity')
# NET_DATA = os.path.join(PROJ_DIR, 'data', 'network_metrics', 'MEA-Mecp2_2022_16Jul2023.csv')
# EPHYS_DATA = os.path.join(PROJ_DIR, 'data', 'ephys_metrics','MEA-Mecp2_2022_23Dec2022.csv')
METADATA = f"{dataset}.xlsx"

# Today's date for saving objects
today = date.today()

# Import metadata
metadata = pd.read_excel(os.path.join(PROJ_DIR, 'data', METADATA),
            sheet_name="fully_connected_mature", engine="openpyxl")
names = metadata["File name"]
ages = metadata["Age"]
genotypes = metadata["Genotype"]
sample_ids = metadata["Sample ID"]
# size = metadata["Size"]

# Input node selection
n_channels = 59
input_nodes = np.array([19, 17, 14, 11, 9]) # NOTE: connectivity data saved without reference electrode
n_input_nodes = len(input_nodes)
output_nodes = np.array([38, 40, 43, 44, 47, 49]) # np.array([n for n in range(n_channels) if n not in input_nodes])
n_output_nodes = len(output_nodes)

input_sf = 10
n_timesteps = 2500
nruns = 1 # 60

# Get trial-based dataset for task
task_name = 'mackey_glass'

if task_name == 'mackey_glass':
    tau = 30
    horizon = 30
    kwargs = {'n_timesteps': n_timesteps, 'tau': tau, 'horizon': horizon, 'input_sf': input_sf}
elif task_name == 'MemoryCapacity':
    horizon = -25
    kwargs = {'n_timesteps': n_timesteps, 'horizon': horizon}

idx_washout = 200

# Metrics to evaluate the regression model for RC output
metrics = ['score'] # , 'mse', 'nrmse'
alphas = np.linspace(0.2, 2.0, num=10)

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

    if conn.n_nodes < n_input_nodes + n_output_nodes:
        input("Cannot implement network. Total nodes must be greater than specified input nodes.  Press Enter to continue.")
        continue
    # scale conenctivity weights between [0,1] and normalize by spectral radius
    try:
        conn.scale_and_normalize()
    except ValueError:
        input("Cannot compute largest eigenvalue. Check connectivity matrix. Press Enter to continue.")
        continue

    # generate and fetch data
    x, y = iodata.fetch_dataset(task_name, **kwargs)

    # split trials into training and test sets
    x_train, x_test, y_train, y_test = iodata.split_dataset(
        x, y, frac_train=0.8)

    # number of features in task data
    n_features = x_train.shape[1]

    # define model for RC ouput
    model = Ridge(alpha=1e-8, fit_intercept=True)

    for alpha in alphas:
        if 0.2 <= alpha < 0.8:
            regime = 'stable'
        elif 0.8 <= alpha < 1.4:
            regime = 'critical'
        else:
            regime = 'chaotic'

        print(
            f'\n----------------------- alpha = {alpha} -----------------------')

        scores = {m:[] for m in metrics}

        alpha_dict = {
            'name': file,
            'age': ages[idx],
            'Genotype': genotypes[idx],
            'rho': conn.spectral_radius,
            'alpha': np.round(alpha, 3),
            'regime': regime}
        
        for run in range(nruns):

        #     # we select a random set of input nodes
        #     if reduce:
        #         conn.reduce()
        #         alpha_dict['size'] = conn.n_nodes
            
        #     # else: # pass whole network but only select from connected nodes
        #     #     input_subset = conn.reduce()
        #     #     input_nodes = conn.get_nodes('random', n_nodes=n_input_nodes,nodes_from=input_subset)

        #     input_nodes = conn.get_nodes('random', n_nodes=n_input_nodes)
        #     output_nodes = conn.get_nodes('random', n_nodes=n_output_nodes, nodes_without=input_nodes)

            # create input connectivity matrix
            w_in = np.random.normal(scale=1e-3, size=(n_features, conn.n_nodes, n_timesteps))
            # w_in = np.zeros(shape=(n_features, conn.n_nodes, n_timesteps))
            w_in[:,input_nodes,:] = np.ones(
                n_features, dtype=int)
        
            # instantiate an Echo State Network object
            ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                                w_hh=conn.w * alpha,
                                                activation_function='tanh',
                                                input_nodes=input_nodes,
                                                output_nodes=output_nodes
                                                )

            # Set random initial condition
            ic = ESN.set_initial_condition()

            # simulate reservoir states; select only output nodes.
            rs_train = ESN.simulate(ext_input=x_train, ic=ic, leak=0.8)[:,output_nodes]
            rs_test = ESN.simulate(ext_input=x_test, ic=ic, leak=0.8)[:,output_nodes]

            # # plot reservoir activity before washout
            # plotting.plot_time_series_raster(rs_train, xlim=[0,len(rs_train)], figsize=(19.2, 9.43), title=f"{file}_reservoir_activity_training",
            #                 savefig=True, block=True)
            # plotting.plot_time_series_raster(rs_test, xlim=[0,len(rs_test)], figsize=(19.2, 9.43), title=f"{file}_reservoir_activity_testing",
            #                 savefig=True, block=True)

            rs_train, rs_test, y_train2, y_test2 = ESN.add_washout_time(
                rs_train, rs_test, y_train, y_test, idx_washout=idx_washout)

            # perform task
            df_run, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                            target=(y_train2, y_test2),
                                            model=model,
                                            metric=metrics)

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
        subj_ls.append(alpha_dict) # | network_data.iloc[idx].to_dict()) # | ephys_data.iloc[idx].to_dict() 

df_subj = pd.DataFrame(subj_ls)
df_subj.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_horizon_{horizon}_tau_{tau}_noise.csv')
print("Dataframe saved.")

#%% Import dataframe
if import_dataframe: 
    df_subj = pd.read_csv(os.path.join(PROJ_DIR, 'dataframes', dataframe_name))

# %% Plotting performance

if plot_perf_curves:
    figsize=(19.2, 9.43)

    # plot performance over runs for each sample at each alpha value
    for m in metrics:

        # Plot unscaled performance
        plotting.plot_perf_reg(df_subj, x='rho',ylim=[0,1],
                        hue="genotype", hue_order=["WT", "HE", "KO"], size='age',
        figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_{m}', **kwargs)

        plotting.boxplot(df=df_subj, x='genotype', y=m, ylim=[0,1],order=["WT", "HE", "KO"], hue='genotype', legend=True,ylab="R squared",
                        figsize=(4.5, 8), width=1.0, savefig=True, title=f'{task_name}_{dataset}_unscaled_performance_genotype_comparison', **kwargs)
        
        network_vars = ["density_full", "net_size", "n_mod","small_world_sigma","small_world_omega"] # rho
        ephys_vars = ['NBurstRate', 'meanNumChansInvolvedInNbursts','meanNBstLengthS', 'meanISIWithinNbursts_ms']
         
        for v in network_vars: # ephys_vars:
            plotting.plot_perf_reg(df_subj, x=v, ylim=[0,1],
                y=m,hue='genotype', legend=False, savefig=True, title=f'{task_name}_{dataset}_{m}_perf_vs_{v}')
        df_connected = df_subj[df_subj.connectedness == 1].reset_index()

        plotting.plot_performance_curve(df_subj, y=m, ylim=[0,1], xlim=[min(alphas),max(alphas)], hue="Genotype", hue_order=["WT", "HE", "KO"],legend=True,
         figsize=(12,8), savefig=True, title=f'{task_name}_{dataset}_performance_curve_horizon_{horizon}_tau_{tau}',format=['png','svg'],**kwargs)
        df_mature = df_connected[df_connected.age > 7].reset_index()
        

        df_regrouped = df_subj.groupby(['name','regime','Genotype']).mean().reset_index()
        genotypes = ["WT", "HE", "KO"]
        df_regrouped['Genotype'] = pd.Categorical(df_regrouped['Genotype'], genotypes)
        df_regrouped['Age'] = df_regrouped['age'].astype('category')
        regimes = ['stable','critical','chaotic']
        df_regrouped['Regime'] = pd.Categorical(df_regrouped['regime'], regimes)
        df_regrouped.sort_values(by='Genotype',inplace=True)
        for regime in regimes:
            df_regime = df_regrouped[df_regrouped.Regime == regime]

            plotting.boxplot(df=df_regime, x='Genotype', order=genotypes, y=m, ylim=[0,1]
                            legend=False, xticklabs=["WT","HET","KO"],
            hue="Genotype", hue_order=["WT", "HE", "KO"],marker="Age", figsize=(8, 8),format=['png','svg'],
            savefig=True, title=f'{task_name}_{dataset}_{horizon}_noise_{regime}', **kwargs)

                                                              
        # Find alpha value at which performance is max
        # new_sample_ls = []
        # for idx,file in names.iteritems():
        #     sample_data = df_subj[df_subj['name'] == file].reset_index()
        #     max_idx = np.argmax(sample_data['score'])
        #     alpha_max = sample_data.at[max_idx,'alpha']
        #     max_score = np.max(sample_data['score'])
        #     genotype = genotypes[idx]
        #     age = ages[idx]
        #     rho = sample_data.at[0, 'rho']
        #     sample_id = sample_ids[idx]
        #     sample = {'name':file,
        #               'id': sample_id,
        #               'alpha_max': alpha_max,
        #               'score': max_score,
        #               'rho': rho,
        #               'genotype': genotype,
        #               'age': age}
        #     new_sample_ls.append(sample)
        # alpha_max_df = pd.DataFrame(new_sample_ls)

        # alpha_max_df.to_csv(
        # f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_scaled_max_score.csv')
        # print("Dataframe saved.")

        # plotting.plot_perf_reg(alpha_max_df, x='rho', size='age',xlabel="rho",
        #     y='alpha_max',hue='genotype', legend=False, savefig=True, title=f'{task_name}_{dataset}_alphamax_vs_rho')
