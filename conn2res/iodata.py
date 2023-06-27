# -*- coding: utf-8 -*-
"""
Functions for generating input/output data for tasks

@author: Estefany Suarez
"""

import os
from re import I
import pandas as pd
import numpy as np
import random
import neurogym as ngym
from reservoirpy import datasets

from .task import select_model
from . import plotting

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_DIR = os.path.join(PROJ_DIR, 'examples', 'data', 'tasks')

NEUROGYM_TASKS = [
    'AntiReach',
    # 'Bandit',
    'ContextDecisionMaking',
    # 'DawTwoStep',
    'DelayComparison',
    'DelayMatchCategory',
    'DelayMatchSample',
    'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation',
    # 'Detection',  # TODO: Temporary removing until bug fixed
    'DualDelayMatchSample',
    # 'EconomicDecisionMaking',
    'GoNogo',
    'HierarchicalReasoning',
    'IntervalDiscrimination',
    'MotorTiming',
    'MultiSensoryIntegration',
    # 'Null',
    'OneTwoThreeGo',
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'PostDecisionWager',
    'ProbabilisticReasoning',
    'PulseDecisionMaking',
    'Reaching1D',
    'Reaching1DWithSelfDistraction',
    'ReachingDelayResponse',
    'ReadySetGo',
    'SingleContextDecisionMaking',
    'SpatialSuppressMotion',
    # 'ToneDetection'  # TODO: Temporary removing until bug fixed
]

NATIVE_TASKS = [
    'MemoryCapacity',
    # 'TemporalPatternRecognition'
]

RESERVOIRPY_TASKS = [
    'henon_map',
    'logistic_map',
    'lorenz',
    'mackey_glass',
    'multiscroll',
    'doublescroll',
    'rabinovich_fabrikant',
    'narma',
    'lorenz96',
    'rossler'
]

STIMULATION_CLASSIFICATION_TASKS = [
    'spatial_classification_v0',
    'temporal_classification_v0'
]

def load_file(file_name, data_dir, file_type=None):

    """
    Loads data files and optionally converts .csv files to numpy arrays
    """
    if file_type is None:
        file_type = os.path.splitext(file_name)[-1]
    if file_type == '.npy':
        return np.load(os.path.join(data_dir, file_name))
    elif file_type == '.csv':
        df = pd.read_csv(os.path.join(data_dir, file_name), header=None)
        return df.to_numpy()
    else:
        raise TypeError("Unsupported file type. Import .npy or .csv file.")

def get_available_tasks():
    return NEUROGYM_TASKS + NATIVE_TASKS + RESERVOIRPY_TASKS + STIMULATION_CLASSIFICATION_TASKS


def unbatch(x):
    """
    Removes batch_size dimension from array

    Parameters
    ----------
    x : numpy.ndarray
        array with dimensions (seq_len, batch_size, features)

    Returns
    -------
    new_x : numpy.ndarray
        new array with dimensions (batch_size*seq_len, features)

    """
    # TODO right now it only works when x is (batch_first = False)
    return np.concatenate(x, axis=0)

def encode_labels(labels):
    """
        Binary encoding of categorical labels for classification
        problems
        # TODO
    """

    enc_labels = -1 * \
        np.ones((labels.shape[0], len(np.unique(labels))), dtype=np.int16)

    for i, label in enumerate(labels):
        enc_labels[i][label] = 1

    return enc_labels

def fetch_dataset(task, horizon=None, **kwargs):
    """
    Fetches inputs and labels for 'task' from the NeuroGym
    repository

    Parameters
    ----------
    task : {'AntiReach', 'Bandit', 'ContextDecisionMaking',
    'DawTwoStep', 'DelayComparison', 'DelayMatchCategory',
    'DelayMatchSample', 'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation', 'Detection', 'DualDelayMatchSample',
    'EconomicDecisionMaking', 'GoNogo', 'HierarchicalReasoning',
    'IntervalDiscrimination', 'MotorTiming', 'MultiSensoryIntegration',
    'OneTwoThreeGo', 'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse', 'PostDecisionWager',
    'ProbabilisticReasoning', 'PulseDecisionMaking',
    'Reaching1D', 'Reaching1DWithSelfDistraction',
    'ReachingDelayResponse', 'ReadySetGo',
    'SingleContextDecisionMaking', 'SpatialSuppressMotion',
    'ToneDetection'}
    Task to be performed

    unbatch_data : bool, optional
        If True, it adds an extra dimension to inputs and labels
        that corresponds to the batch_size. Otherwise, it returns an
        observations by features array for the inputs, and a one
        dimensional array for the labels.

    Returns
    -------
    inputs : numpy.ndarray
        array of observations by features
    labels : numpy.ndarray
        unidimensional array of labels
    """

    if task in NEUROGYM_TASKS:
        # create a conn2res Dataset
        return create_neurogymn_dataset(task, **kwargs)

    elif task in NATIVE_TASKS + RESERVOIRPY_TASKS:
        # create a conn2res Dataset
        return create_regression_dataset(task, horizon, **kwargs)
    elif task in STIMULATION_CLASSIFICATION_TASKS:
        return create_classification_dataset(task, **kwargs)


def create_neurogymn_dataset(task, n_trials=100, add_constant=False, **kwargs):
    # create a Dataset object from NeuroGym
    dataset = ngym.Dataset(task+'-v0', env_kwargs=kwargs)

    # get environment object
    env = dataset.env

    # generate per trial dataset
    _ = env.reset()
    inputs = []
    labels = []
    if task == 'ContextDecisionMaking':
        sample_class = []
    for trial in range(n_trials):
        env.new_trial()
        ob, gt = env.ob, env.gt

        # add constant term
        if add_constant:
            ob = np.concatenate((np.ones((ob.shape[0], 1)), ob), axis=1)

        # store input
        inputs.append(ob)

        # store labels
        if gt.ndim == 1:
            labels.append(gt[:, np.newaxis])
        else:
            labels.append(gt)

        # extra label for context dependent decision making task
        if task == 'ContextDecisionMaking':
            _class = ob[:, -1, np.newaxis]
            if np.mean(ob[:, 2] - ob[:, 1]) > 0:
                _class = np.hstack((_class, np.ones((gt.size, 1))))
            else:
                _class = np.hstack((_class, np.zeros((gt.size, 1))))
            if np.mean(ob[:, 4] - ob[:, 3]) > 0:
                _class = np.hstack((_class, np.ones((gt.size, 1))))
            else:
                _class = np.hstack((_class, np.zeros((gt.size, 1))))
            sample_class.append(_class)

    if task == 'ContextDecisionMaking':
        return inputs, labels, sample_class
    else:
        return inputs, labels


def create_regression_dataset(task, horizon, n_timesteps, input_sf, **kwargs):

    # make sure horizon is a list
    if isinstance(horizon, int):
        horizon = [horizon]

    # check that horizon has elements with same sign
    horizon_sign = np.unique(np.sign(horizon))
    if horizon_sign.size == 1:
        horizon_sign = horizon_sign[0]
    else:
        raise ValueError('horizon should have elements with same sign')

    # transform horizon elements to positive values (and nd.array)
    horizon = np.abs(horizon)

    # calculate maximum horizon
    horizon_max = np.max(horizon)

    if task == 'MemoryCapacity':
        x = np.random.uniform(-1, 1, (n_timesteps+horizon_max))[:, np.newaxis]

    elif task in RESERVOIRPY_TASKS:
        func = getattr(datasets, task)
        n_timesteps = n_timesteps + horizon_max
        x = func(n_timesteps=n_timesteps, **kwargs)

    # elif task in STIMULATION_REGRESSION_TASKS:
    #     n_timesteps = n_timesteps + horizon_max
    #     x = load_file(file_name=f'{task}.csv', data_dir=TASK_DIR)[:n_timesteps]

    y = np.hstack([x[horizon_max-h:-h] for h in horizon])
    x = x[horizon_max:]

    mean_centre_func = lambda x: x - x.mean()

    if horizon_sign == -1:
        y = mean_centre_func(y)
        y *= input_sf # mean and centre
        return x, y
    elif horizon_sign == 1:
        x = mean_centre_func(x)
        x *= input_sf # mean and centre
        if y.shape[1] == 1:
            return y, x
        else:
            raise ValueError('positive horizon should be integer not list')

def random_pattern_sequence(n_patterns, n_trials):
    
    order = np.tile(np.arange(n_patterns), n_trials)
    rand_order = random.sample(range(n_patterns*n_trials),n_patterns*n_trials)
    sequence = np.array([order[i] for i in rand_order])
    return sequence

def inv_trans_sample(n,dist,quantiles):

    samples = np.random.uniform(min(quantiles),max(quantiles),size=(n,1))
    bin_idx = np.argmin(np.abs(quantiles[np.newaxis,:]-samples),axis=1)
    vals = dist[bin_idx]
    return vals

def create_classification_dataset(task, input_shape, noise, washout, stim_duration, post_stim, input_sf, **kwargs):
    if task == "spatial_classification_v0":
        # TODO: ensure even number of timesteps for biphasic signal

        if input_shape == "biphasic":
            
            stim = np.concatenate((np.ones(int(stim_duration/2)),-np.ones(int(stim_duration/2)))) * input_sf
        
        elif input_shape == "monophasic":
            stim = np.ones(stim_duration,dtype=int) * input_sf
            
        pre_stim_background = np.ones(washout,dtype=int)
        post_stim_bacgkround =  np.ones(post_stim,dtype=int)
        x = np.concatenate((pre_stim_background,stim,post_stim_bacgkround))[:,np.newaxis]
            
        # y = np.array([np.repeat(trial,n_timesteps)[:,np.newaxis] for trial in sequence])
    # elif task == "temporal_classification_v0":

    return x


def split_dataset(*args, frac_train, axis=0, n_train=None):
    """
    Splits data into training and test sets according to
    'frac_train'

    Parameters
    ----------
    data : numpy.ndarray
        data array to be split
    frac_train : float, from 0 to 1
        fraction of samples in training set
    axis: int
        axis along which the data should be split or concatenated
    n_train: int
        number of samples in training set

    Returns
    -------
    training_set : numpy.ndarray
        array with training observations
    test_set     : numpy.ndarray
        array with test observations
    """

    argout = []
    for arg in args:
        if isinstance(arg, list):
            if n_train is None:
                n_train = int(frac_train * len(arg))

            if axis == 0:
                argout.extend([np.vstack(arg[:n_train]),
                              np.vstack(arg[n_train:])])
            elif axis == 1:
                argout.extend([np.hstack(arg[:n_train]),
                              np.hstack(arg[n_train:])])

        elif isinstance(arg, np.ndarray):
            if n_train is None:
                n_train = int(frac_train * arg.shape[axis])

            if axis == 0:
                argout.extend([arg[:n_train, :], arg[n_train:, :]])
            elif axis == 1:
                argout.extend([arg[:, :n_train], arg[:, n_train:]])

    return tuple(argout)


def visualize_data(task, x, y, plot=True):
    """
    # TODO
    Visualizes dataset for task

    Parameters
    ----------
    task : str
    x,y  : list or numpy.ndarray

    """

    print(f'\n----- {task.upper()} -----')
    print(f'\tNumber of trials = {len(x)}')

    # convert x and y to arrays for visualization
    if isinstance(x, list):
        x = np.vstack(x[:5])
    if isinstance(y, list):
        y = np.vstack(y[:5]).squeeze()

    print(f'\tinputs shape = {x.shape}')
    print(f'\tlabels shape = {y.shape}')

    # number of features, labels and classes
    try:
        n_features = x.shape[1]
    except:
        n_features = 1

    try:
        n_labels = y.shape[1]
    except:
        n_labels = 1
    n_classes = len(np.unique(y))

    model = select_model(y)

    print(f'\t  n_features = {n_features}')
    print(f'\tn_labels   = {n_labels}')
    print(f'\tn_classes  = {n_classes}')
    print(f'\tlabel type : {y.dtype}')
    print(f'\tmodel = {model.__name__}')

    if plot:
        plotting.plot_task(x, y, task)


def get_sample_weight(inputs, labels, sample_block=None):
    """
    Time averages dataset based on sample class and sample weight

    Parameters
    ----------
    inputs : numpy.ndarray or list of numpy.ndarrays
        input data
    labels: numpy.ndarray or list of numpy.ndarrays
        label data
    sample_block : numpy.ndarray
        block structure which is used as a basis for weighting
        (i.e., same weights are applied within each block)

    Returns
    -------
    sample_weight: numpy.ndarray or list of numpy.ndarrays
        weights of samples which can be used either for averaging time
        series or training models whilst weighting samples in the cost
        function
    idx_sample: numpy.ndarray or list of numpy.ndarrays
        indexes of samples with one index per block (see sample_block)
    """

    if isinstance(inputs, np.ndarray):
        inputs = [inputs]

    if isinstance(labels, np.ndarray):
        labels = [labels]

    sample_weight = []
    if sample_block is None:
        for data in inputs:
            # sample block based on unique combinations of classes in data
            icol = [col for col in range(data.shape[1]) if np.unique(
                data[:, col]).size <= 3]  # class is based on <=3 real values
            _, sample_block = np.unique(
                data[:, icol], return_inverse=True, axis=0)

        # get unique sample blocks
        _, ia, nc = np.unique(
            sample_block, return_index=True, return_counts=True)

        # sample weight
        sample_weight.append(
            np.hstack([np.tile(1/e, e) for e in nc[np.argsort(ia)]]))
    else:
        # get unique sample blocks
        _, ia, nc = np.unique(
            sample_block, return_index=True, return_counts=True)

        for data in inputs:
            # sample weight
            sample_weight.append(
                np.hstack([np.tile(1/e, e) for e in nc[np.argsort(ia)]]))


    return sample_weight

def io_dist(inputs,outputs):
    idx = np.array([24,26,29,32,35,37,21,22,25,30,31,36,39,40,19,20, 23,28,33,38,41,42,
            16,17,18,27,34,43,44,45,15,14,13,4,57,48,47,46,12,11,8,3,58,53,50,49,
            10,9,6,1,60,55,52,51,7,5,2,59,56,54]) - 1
    coords = np.array([21,31, 41, 51, 61, 71, 12, 22, 32, 42, 52, 62, 72, 82, 13, 23, 33, 43, 53, 63,
            73, 83, 14, 24, 34, 44, 54, 64, 74, 84, 15, 25, 35, 45, 55, 65, 75, 85, 16, 26,
            36, 46, 56, 66, 76, 86, 17, 27, 37, 47, 57, 67, 77, 87, 28, 38, 48, 58, 68, 78])
    output_coords = np.array(coords[[np.where(idx == c)[0][0] for c in outputs]])
    all_dist = np.zeros((len(output_coords),1))
    for idx,val in np.ndenumerate(output_coords):
        output_vec = np.array([int(u) for u in str(val)])
        dist = np.mean(np.linalg.norm((inputs-output_vec),axis=1))
        all_dist[idx] = dist
    mean_dist = np.mean(all_dist)
    return mean_dist


