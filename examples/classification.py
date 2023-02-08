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
import_dataframe = 1
dataframe_name = 'spatial_classification_v0_MEA-Mecp2_2022_dataset_conn2res_2023-02-02_100_trials_5_output_nodes.csv' # name of previously generated .csv dataframe
plot_diagnostics = 0
plot_perf_curves = 1

# Paths and spreadsheets
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = "MEA-Mecp2_2022_dataset_conn2res"
CONN_DATA = os.path.join(PROJ_DIR, 'data', 'connectivity', dataset)
# NET_DATA = os.path.join(PROJ_DIR, 'data', 'network_metrics', 'MEA-Mecp2_2022_26Dec2022.csv')
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

# Get trial-based dataset for task
frac_train=0.8

task_name = 'spatial_classification_v0'

if task_name == 'spatial_classification_v0':
    n_patterns = 2
    n_pres = 50 # number of presentations of each pattern
    kwargs = {'n_timesteps': 250, 'ITI': 0}
elif task_name == 'temporal_classification_v0':
    kwargs = {}

# Input node selection

random_input = False
if random_input:
    n_input_nodes = 5
    n_trials = 10

fixed_input = True
if fixed_input:
    n_input_nodes = 1 # number of nodes in each pattern
    input_patterns = np.array([[0],[1]])
    input_set = np.concatenate(input_patterns)
    
n_output_nodes = 5
output_nodes = np.array([58,57,56,55,54])

n_trials = n_patterns*n_pres
n_training_trials = round(n_pres*frac_train) # for each pattern
n_testing_trials = round(n_pres*(1-frac_train)) # for each pattern

idx_washout = 50

# Metrics to evaluate the regression model for RC output
metrics = ['score'] # , 'mse', 'nrmse'
alphas = np.array([0.2, 0.8, 1.0, 1.2, 2.0]) # np.linspace(0.2, 3.0, num=15)

fig_num = 1

# %% Main

df_sample_ls = [] # initialise list to contain dictionaries for each alpha trial
# ephys_data = pd.read_csv(EPHYS_DATA)
# network_data = pd.read_csv(NET_DATA)

# generate and fetch data
x = iodata.fetch_dataset(task_name, **kwargs)

y_train = iodata.random_pattern_sequence(n_patterns, n_training_trials)
y_test = iodata.random_pattern_sequence(n_patterns, n_testing_trials)
y = np.concatenate([y_train, y_test])

# number of features in task data
n_features = x.shape[1]

# define model for RC ouput
model = RidgeClassifier(alpha=1e-8, fit_intercept=True)

for idx,file in names.iteritems():

    # Import connectivity matrix
    print(
            f'\n*************** file = {file} ***************')
    conn = reservoir.Conn(w=None, conn_data=f'{file}.csv', conn_data_dir=CONN_DATA)
    
    conn.reduce()

    if random_input and conn.n_nodes < n_input_nodes + n_output_nodes:
        input("Cannot implement network. Total nodes must be greater than specified input nodes. Press Enter to continue.")
        continue
    if fixed_input and conn.n_nodes != 59:
        input("Cannot implement network. Not fully connected. Press Enter to continue.")
        continue
    
    # scale conenctivity weights between [0,1] and normalize by spectral radius
    try:
        conn.scale_and_normalize()
    except ValueError:
        # input("Cannot compute largest eigenvalue. Check connectivity matrix. Press Enter to continue.")
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
            'alpha': np.round(alpha, 3),
            'regime': regime,
            'rho': conn.spectral_radius}
        
        # TRAINING

        rs = np.zeros((n_trials, n_output_nodes))
        # output_nodes = conn.get_nodes('random', n_nodes=n_output_nodes, nodes_without=input_set)

        for trial in range(n_trials):

            # we select a random set of input nodes
            if random_input: input_nodes = conn.get_nodes('random', n_nodes=n_input_nodes)
            
            # for spatial classification, select input nodes according to sequence
            if task_name == "spatial_classification_v0" and fixed_input:
                input_nodes = input_patterns[y[trial]]
                
            # output_nodes = conn.get_nodes('all', nodes_without=input_nodes)

            # create input connectivity matrix
            w_in = np.zeros((n_features, conn.n_nodes))
            w_in[:, input_nodes] = np.eye(n_features)
        
            # instantiate an Echo State Network object
            ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                                w_hh=conn.w * alpha,
                                                activation_function='tanh',
                                                #  input_gain=10.0,
                                                input_nodes=input_nodes,
                                                output_nodes=output_nodes
                                                )

            # simulate reservoir states; select only output nodes.
            rs_trial = ESN.simulate(ext_input=x)

            rs_trial = ESN.add_washout_time(
                rs_trial, idx_washout=idx_washout)[0] # return first item in tuple
            
            rs[trial,:] = np.mean(rs_trial, axis=0)

            # # plot reservoir activity TODO: only plot one raster for each pattern
            # plotting.plot_time_series_raster(rs_trial, xlim=[0,len(rs_trial)], figsize=(19.2, 9.43),
            # title=f"{file}_reservoir_activity_trial_{trial}", savefig=True)

        # TESTING
        rs_train, rs_test = iodata.split_dataset(rs, frac_train=frac_train)

        scores = {m:[] for m in metrics}
        df_alpha, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                        target=(y_train, y_test),
                                        model=model,
                                        metric=metrics)
        # join alpha dictionary and df_alpha
        for m in metrics: alpha_dict[m] = df_alpha.at[0, m]
        df_sample_ls.append(alpha_dict) # | ephys_data.iloc[idx].to_dict() # | network_data.iloc[idx].to_dict()

        if plot_diagnostics:
            
            figsize=(19.2, 9.43)
            
            plotting.plot_time_series(y_test, feature_set='data', xlim=[0,n_testing_trials*n_patterns], figsize=figsize, subplot=(1,1,1), legend_label='Target output', num=fig_num)
            plotting.plot_time_series(rs_test, feature_set='pred', xlim=[0,n_testing_trials*n_patterns], xticks=range(0,n_testing_trials*n_patterns), model=modelout,
            figsize=figsize, subplot=(1,1,1), legend_label='Predicted output', block=False, num=fig_num,
                                        savefig=True, title=f'{file}_{alpha}_performance')
            fig_num += 1

df_sample = pd.DataFrame(df_sample_ls)
df_sample.to_csv(
    f'{PROJ_DIR}/dataframes/{task_name}_{dataset}_{today}_{n_trials}_trials_{n_output_nodes}_output_nodes.csv')
print("Dataframe saved.")

#%% Import dataframe
if import_dataframe: 
    df_sample = pd.read_csv(os.path.join(PROJ_DIR, 'dataframes', dataframe_name))

# %% Plotting performance

if plot_perf_curves:
    figsize=(19.2, 9.43)
    kwargs = {'ages': np.sort(ages.unique()),
    'genotypes': genotypes.unique()}

    # plot performance over trials for each sample at each alpha value
    for m in metrics:

        plotting.plot_perf_reg(df_sample, x='alpha', y=m, ylim=[0,1], hue="genotype", hue_order=["WT", "HE", "KO"], # xlim=[min(alphas),max(alphas)]
        figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_performance_{m}', **kwargs)

        # plot boxplots deparated by age and genotype for performance of unscaled networks
        
           
        # # plot genotype comparisons for dynamical regimes
        # plotting.boxplot(x='regime', y=m, df=df_subj, hue='genotype', hue_order=["WT", "HE", "KO"],
        #     ylim=[0,1], figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_performance_{m}_boxplots')

        # # plot genotype comparisons at each developmental age
        # plotting.plot_performance_curve(df_subj, by_age=True, y=m, ylim=[0,1], xlim=[min(alphas),max(alphas)], hue='genotype', hue_order=["WT", "HE", "KO"],
        # figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_performance_{m}_across_development_{m}',
        # **kwargs)

        # # # plot performance normalised by energetic cost metrics
        # # plotting.plot_performance_curve(df_subj, y=m, xlim=[min(alphas), max(alphas)], hue='genotype', hue_order=["WT", "HE", "KO"],
        # # norm=True, norm_var= "meanspikes", figsize=figsize, savefig=True, title=f'{task_name}_{dataset}_performance_{m}_normalised_by_channel_average_spikes',
        # # **kwargs)

        # # examine trends at peak performance (alpha = 1.0)
        # alpha_data = df_subj[df_subj['alpha'] == 1.0].reset_index()

        # # plot change in performance over development
        # plotting.plot_performance_curve(alpha_data, x='age', y=m, ylim=[0,1], xlim=[min(ages),max(ages)], ticks=ages.unique(),
        # hue='genotype', hue_order=["WT", "HE", "KO"], figsize=figsize, savefig=True, block=False,
        # title=f'{task_name}_{dataset}_performance_across_development_{m}',
        # **kwargs)

        # # linear regression model for performance as a function of network and electrophysiological variables
        # network_vars = ["size"] # ["density_full", "density_full", "net_size", "n_mod","small_world_sigma","small_world_omega"]
        # ephys_vars = ['meanspikes', 'FRmean'] # ['meanspikes','NBurstRate', 'meanNumChansInvolvedInNbursts','meanNBstLengthS', 'meanISIWithinNbursts_ms', 'CVofINBI']
         
        # for v in [network_vars, ephys_vars]:
        #     plotting.plot_perf_reg(alpha_data, x=v,
        #         y=m,hue='genotype',savefig=True, title=f'{task_name}_{dataset}_{m}_perf_vs_{v}')

# %%
