# -*- coding: utf-8 -*-
"""
MEA functional connectome-informed reservoir (Echo-State Network)
=================================================
"""

# from reservoirpy.nodes import Reservoir, Ridge
import os
from bct.algorithms.degree import strengths_und
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from conn2res import reservoir, coding, plotting, iodata, task
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
plt.ioff() # turn off interactive mode
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
    horizon = 10
    kwargs = {'n_timesteps': 4000, 'tau': tau, 'horizon': horizon}
elif task_name == 'MemoryCapacity':
    horizon = -25
    kwargs = {'n_timesteps': 4000, 'horizon': horizon}

idx_washout = 200

# Metrics to evaluate the regression model for RC output
metric = ['rsquare'] # , 'mse', 'nrmse'

nruns = 3

# Import metadata
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'examples', 'data', 'MEA-Mecp2_all')
dataset = 'MEA-Mecp2_all'

metadata = pd.read_excel(os.path.join(DATA_DIR, "Mecp2_dataset_all.xlsx"),
            sheet_name="Sheet1", engine="openpyxl")
names = metadata["File name"]
ages = metadata["Age"]
genotypes = metadata["Genotype"]

fig_num = 1
df_subj = []

for idx,file in names.iteritems():
    # Import connectivity matrix
    print(
            f'\n*************** file = {file} ***************')
    conn = reservoir.Conn(w=None, conn_data=f'{file}.csv', conn_data_dir=DATA_DIR) # conn_data=file, file_type='.csv'
    # TODO: remove reference electrode nodes from adjacency matrix using subset_nodes

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

    for run in range(nruns):
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
        # alphas = np.array([1.0])
        alphas = np.array([0.2, 0.8, 1.0, 1.2, 2.0])
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
            df['age'] = ages[idx]
            df['genotype'] = genotypes[idx]
            df['run'] = run
            df['nodes'] = conn.n_nodes
            N = 200
            tau = 100
            figsize = (12, 6)
            # sample = [0+horizon, N] if horizon > 0 else [0, N+horizon]
            # plotting.plot_time_series(x_test[idx_washout+tau:], feature_set='data', xlim=[0, N], sample=sample,
            #                             num=fig_num, figsize=figsize, subplot=(3, 3, (1, 2)), legend_label='Input', block=False)
            # plotting.plot_mackey_glass_phase_space(x_test[idx_washout:], x_test[idx_washout+tau:], xlim=[0.2, 1.4], ylim=[0.2, 1.4], color='magma', sample=sample,
            #                                         num=fig_num, figsize=figsize, subplot=(3, 3, 3), block=False)

            # sample = [0, N-horizon] if horizon > 0 else [-horizon, N]
            # plotting.plot_time_series(y_test2[tau:N+tau-horizon], feature_set='data', xlim=[0, N], sample=sample,
            #                             num=fig_num, figsize=figsize, subplot=(3, 3, (4, 5)), legend_label='Label', block=False)
            # plotting.plot_time_series(rs_test[tau:], feature_set='pred', xlim=[0, N], sample=sample,
            #                             num=fig_num, figsize=figsize, subplot=(3, 3, (4, 5)), legend_label='Predicted label', block=False, model=modelout)
            # plotting.plot_mackey_glass_phase_space(modelout.predict(rs_test[0:]), modelout.predict(rs_test[0+tau:]), xlim=[0.2, 1.4], color='magma', sample=sample,
            #                                         ylim=[0.2, 1.4], num=fig_num, figsize=figsize, subplot=(3, 3, 6), block=False)

            # plotting.plot_time_series(rs_test[tau:N+tau-horizon], feature_set='pc', xlim=[0, N], normalize=True, idx_features=[1, 2, 3],
            #                             num=fig_num, figsize=figsize, subplot=(3, 3, (7, 8)), legend_label='Readout PC', block=False,
            #                             savefig=True, fname=f'{task_name}_diagnostics_{file}_a{alpha}')
            fig_num += 1

            df_subj.append(df)

df_subj = pd.concat(df_subj, ignore_index=True)

############################################################################
# Now we plot the performance curve

# df_subj.to_csv(
#     f'/Users/amihalik/Documents/projects/reservoir/conn2res/figs/{task_name}_performance.csv')

# performance

for m in metric:
    plotting.plot_performance_curve(
        df_subj, f'{task_name}_{dataset}_performance_{nruns}_runs', y=m, hue='genotype' # name of column in dataframe to use for grouping
        , norm=False, num=fig_num, figsize=(12, 6), savefig=True, show=True, block=False)
    fig_num += 1