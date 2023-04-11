# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Memristive Network
=================================================
This example demonstrates how to use the conn2res toolbox 
to perform a task using a human connectomed-informed
Memristive network
"""

from conn2res import reservoir, coding, iodata, plotting
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

###############################################################################
# First let's import the connectivity matrix we are going to use to define the
# connections of the reservoir.  For this we will be using the human connectome
# parcellated into 1015 brain regions following the Desikan  Killiany atlas
# (Desikan, et al., 2006).

# load connectivity data of one subject
conn = reservoir.Conn(subj_id=10)

# binarize connectivity matrix
conn.binarize()

# # normalize connectivity matrix by the spectral radius
# conn.normalize()

###############################################################################
# Second let's get the data to perform the task. We first generate the data and
# then we split it into training and test sets. 'x' corresponds to the input
# signals and 'y' corresponds to the output labels.

# get trial-based dataset for task
task = 'PerceptualDecisionMaking'
x, y = iodata.fetch_dataset(task, n_trials=1000, dt=100)

# visualize task data
iodata.visualize_data(task, x, y, plot=False)

# split data into training and test sets
x_train, x_test, y_train, y_test = iodata.split_dataset(x, y)

###############################################################################
# Third we will simulate the dynamics of the reservoir using the previously
# generated input signal x (x_train and x_test).

# define sets of internal, external and ground nodes
ctx = iodata.load_file('cortical.npy')

# number of features in task data
n_features = x_train.shape[1]

# we select a single random ground node from cortical regions
gr_nodes = conn.get_nodes(
    'random', nodes_from=conn.get_nodes('ctx'), n_nodes=1)

# we use the remaining cortical regions as output nodes
output_nodes = conn.get_nodes('ctx', nodes_without=gr_nodes)

# we select external nodes as random set of input nodes from subcortical regions
ext_nodes = conn.get_nodes(
    'random', nodes_from=conn.get_nodes('subctx'), n_nodes=n_features)

# we select internal nodes as all nodes except of ground nodes external nodes
int_nodes = conn.get_nodes(
    'all', nodes_without=np.union1d(gr_nodes, ext_nodes))

# We will use resting-state networks as readout modules. These intrinsic networks
# define different sets of output nodes
rsn_mapping = iodata.load_file('rsn_mapping.npy')
# we select the mapping only for output nodes
rsn_mapping = rsn_mapping[conn.idx_node][output_nodes]

# evaluate network performance across various dynamical regimes
# we do so by varying the value of alpha

alphas = np.linspace(0, 2, 11)[1:]
df_subj = []
for alpha in alphas:

    print(f'\n----------------------- alpha = {alpha} -----------------------')

    # instantiate an Memristive Network object
    MMN = reservoir.MSSNetwork(w=alpha * conn.w,
                               int_nodes=int_nodes,
                               ext_nodes=ext_nodes,
                               gr_nodes=gr_nodes
                               )

    # simulate reservoir states; select only readout nodes.
    rs_train = MMN.simulate(Vext=x_train[:], mode='forward')[:, output_nodes]
    rs_test = MMN.simulate(Vext=x_test[:],  mode='forward')[:, output_nodes]

    # perform task
    df = coding.encoder(reservoir_states=(rs_train, rs_test),
                        target=(y_train, y_test),
                        readout_modules=rsn_mapping,
                        )

    df['alpha'] = np.round(alpha, 3)

    # reorganize the columns
    if 'module' in df.columns:
        df_subj.append(df[['module', 'n_nodes', 'alpha', 'score']])
    else:
        df_subj.append(df[['alpha', 'score']])

df_subj = pd.concat(df_subj, ignore_index=True)
df_subj['score'] = df_subj['score'].astype(float)

#############################################################################
# Now we plot the performance curve

plotting.plot_performance_curve(df_subj, task)