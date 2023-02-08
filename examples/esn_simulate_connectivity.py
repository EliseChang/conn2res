# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Echo-State Network
=================================================
This example demonstrates how to use the conn2res toolbox
to perform a task using a human connectomed-informed
Echo-State network (Jaeger, 2000).
"""

# from reservoirpy.nodes import Reservoir, Ridge
from bct.algorithms.degree import strengths_und
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from conn2res import reservoir, coding, plotting, iodata, task
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
# plt.ioff() # turn off interactive mode
import networkx as nx
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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
    horizon = 25
    kwargs = {'n_timesteps': 4000, 'tau': tau, 'horizon': horizon}
elif task_name == 'MemoryCapacity':
    horizon = -25
    kwargs = {'n_timesteps': 4000, 'horizon': horizon}

idx_washout = 0

# Metrics to evaluate the regression model for RC output
metric = ['rsquare', 'mse', 'nrmse']

nruns = 1
nparams = 5
# Watts-Strogatz rewiring parameter
p = np.logspace(-4, 0, num=nparams, base=10)

fig_num = 1
df_subj = []
for t in range(nparams):  # iterate over rewiring parameter of Watts-Strogatz networks
    for run in range(nruns):  # iterate due to random input nodes
        # Generate Watts-Strogatz networks
        ws = nx.connected_watts_strogatz_graph(1000, 2, p[t])
        conn = reservoir.Conn(w=nx.to_numpy_matrix(ws))

        # scale conenctivity weights between [0,1] and normalize by spectral radius
        conn.scale_and_normalize()

        # generate and fetch data
        x, y = iodata.fetch_dataset(task_name, **kwargs)

        # split trials into training and test sets
        x_train, x_test, y_train, y_test = iodata.split_dataset(
            x, y, frac_train=0.75)

        # number of features in task data
        n_features = x_train.shape[1]

        # we select a random set of input nodes
        input_nodes = conn.get_nodes('random', n_nodes=5)

        # we use cortical regions as output nodes
        output_nodes = conn.get_nodes('all', nodes_without=input_nodes)

        # create input connectivity matrix, which defines the connec-
        # tions between the input layer (source nodes where the input signal is
        # coming from) and the input nodes of the reservoir.
        w_in = np.zeros((n_features, conn.n_nodes))
        w_in[:, input_nodes] = np.eye(n_features)

        # define model for RC ouput
        model = Ridge(alpha=1e-8, fit_intercept=True)

        # evaluate network performance across various dynamical regimes
        # we do so by varying the value of alpha

        # alphas = np.array([0.2, 0.8, 1.0, 1.2, 2.0])
        alphas = np.array([1.0])
        for alpha in alphas:
            print(
                f'\n----------------------- alpha = {alpha} -----------------------')

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
            df, modelout = coding.encoder(reservoir_states=(rs_train, rs_test),
                                          target=(y_train2, y_test2),
                                          model=model,
                                          metric=metric
                                          )

            df['alpha'] = np.round(alpha, 3)
            df['watts_strogatz'] = np.round(p[t], 4)
            df['run'] = run
            df['nodes'] = conn.n_nodes
            N = 200
            tau = 100
            figsize = (12, 6)

            # Plot testing data
            # input
            # x_test = x_test[100:,:]
            # plotting.plot_time_series(x_test[:25,:], feature_set='data', xlim=[0, N], sample=None,
            #                       linestyle='--', num=fig_num, figsize=figsize, subplot = (1, 1, 1), block=False)
            # plotting.plot_time_series(x_test, feature_set='data', xlim=[0, N], sample=[25,200],
            #                       num=fig_num, figsize=figsize, subplot = (1, 1, 1), block=False,                
            #                       savefig=True, fname='Mackey_Glass_testing_input')
            # target output
            # plotting.plot_time_series(x_test[25:200,:], feature_set='data', xlim=[0, N], sample=None,
            #                       num=fig_num, figsize=figsize, subplot = (1, 1, 1), block=False)
            # plotting.plot_time_series(x_test[200:225,:], feature_set='data', xlim=[0, N], sample=None,
            #                       num=fig_num, figsize=figsize, subplot = (1, 1, 1), block=False,                
            #                       temp=True, linestyle='--', savefig=True, fname='Mackey_Glass_testing_output')
            
            # sample = [0+horizon, N] if horizon > 0 else [0, N+horizon]
            # plotting.plot_time_series(x_test[idx_washout+tau:], feature_set='data', xlim=[0, N], sample=sample,
            #                           num=fig_num, figsize=figsize, subplot=(3, 3, (1, 2)), block=False, savefig=True, fname=f'{task_name}_testing_input') # legend_label='Input'
            # plotting.plot_mackey_glass_phase_space(x_test[idx_washout:], x_test[idx_washout+tau:], xlim=[0.2, 1.4], color='magma', sample=sample,
            #                                        num=fig_num, figsize=figsize, subplot=(3, 3, 3), block=False)

            # prediction
            # sample = [0, N-horizon] if horizon > 0 else [-horizon, N]
            # # plotting.plot_time_series(y_test2[tau:N+tau-horizon], feature_set='data', xlim=[0, N], sample=sample,
            # #                           num=fig_num, figsize=figsize, subplot=(3, 3, (4, 5)), legend_label='Target', block=False)
            rs_test = rs_test[100:]
            plotting.plot_time_series(rs_test[:175], feature_set='pred', xlim=[0, N], sample=None,
                                      num=fig_num, figsize=figsize, subplot=(1, 1, 1), block=False, model=modelout)
            plotting.plot_time_series(rs_test, feature_set='pred', xlim=[0, N], sample=[175,200],
                                      num=fig_num, figsize=figsize, subplot=(1, 1, 1), block=False, model=modelout,
                                      linestyle = '--', savefig=True, fname=f'{task_name}_prediction_p{np.round(p[t], 4)}_a{alpha}_run{run}')
            # plotting.plot_mackey_glass_phase_space(modelout.predict(rs_test[0:]), modelout.predict(rs_test[0+tau:]), xlim=[0.2, 1.4], color='magma', sample=sample,
                                                    # num=fig_num, figsize=figsize, subplot=(3, 3, 6), block=False)

            # plotting.plot_time_series(rs_test[tau:N+tau-horizon], feature_set='pc', xlim=[0, N], normalize=True, idx_features=[1, 2, 3],
            #                           num=fig_num, figsize=figsize, subplot=(3, 3, (7, 8)), legend_label='Readout PC', block=False,
            #                           savefig=True, fname=f'{task_name}_diagnostics_p{np.round(p[t], 4)}_a{alpha}_run{run}')

            fig_num += 1

            df_subj.append(df)

df_subj = pd.concat(df_subj, ignore_index=True)

############################################################################
# Now we plot the performance curve

# df_subj.to_csv(
#     f'/Users/amihalik/Documents/projects/reservoir/conn2res/figs/{task_name}_performance.csv')

# performance

# for m in metric:
#     if use_data:
#         plotting.plot_performance_curve(
#         df_subj, f'{task_name}_performance', y=m, norm=False, num=fig_num, figsize=(12, 6), savefig=True, block=False)
#     elif sim_conn:
#         plotting.plot_performance_curve(
#             df_subj, f'{task_name}_watts_strogatz_performance', y=m, hue='watts_strogatz', norm=False, num=fig_num, figsize=(12, 6), savefig=True, block=False)
#     fig_num += 1