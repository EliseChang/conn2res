# -*- coding: utf-8 -*-
"""
Functions to train the readout module to perform
tasks

@author: Estefany Suarez
"""

import sys
import numpy as np
import pandas as pd
import scipy as sp
# import mdp

from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor


from reservoirpy import observables


def check_xy_dims(x, y):
    """
    Check that X,Y have the right dimensions
    # TODO
    """
    x_train, x_test = x
    y_train, y_test = y

    if ((x_train.ndim == 1) and (x_test.ndim == 1)):
        x_train = x_train[:, np.newaxis]
        x_test = x_test[:, np.newaxis]
    elif ((x_train.ndim > 2) and (x_test.ndim > 2)):
        x_train = x_train.squeeze()
        x_test = x_test.squeeze()

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    return x_train, x_test, y_train, y_test


def corrcoef(y_true, y_pred):
    """Absolute Pearson's correlation between true and predicted label

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.

    Returns
    -------
    float
        Absolute Pearson's correlation.
    """

    return np.abs(np.corrcoef(y_true, y_pred)[0][1])


def nrmse(y_true, y_pred):
    """Normalized root mean squared error

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.

    Returns
    -------
    float
        Root mean squared error normalized by variance.
    """

    error = metrics.mean_squared_error(y_true, y_pred, squared=False)

    return error / y_true.var()

def regression(x, y, model=None, metric=['score'], **kwargs):

    """
    Regression tasks
    # TODO
    """

    # pop variables from kwargs
    sample_weight_train, sample_weight_test = kwargs.pop(
        'sample_weight', (None, None))

    x_train, x_test = x
    y_train, y_test = y

    # specify default model
    if model is None:
        model = Ridge(fit_intercept=False, alpha=0.5,
                      **kwargs)

    # fit model on training data
    model.fit(x_train, y_train, sample_weight_train)

    # calculate model scores on test data
    score = []
    for m in metric:
        if m == 'score':
            score.append(model.score(x_test, y_test, sample_weight_test))
        else:
            func = getattr(observables, m)
            y_pred = model.predict(x_test)
            score.append(func(y_test, y_pred))

    return score, model


def multiOutputRegression(x, y, model=None, metric=['corrcoef'], **kwargs):
    """
    Multiple output regression tasks
    # TODO
    """

    # pop variables from kwargs
    sample_weight_train, sample_weight_test = kwargs.pop(
        'sample_weight', (None, None))

    x_train, x_test = x
    y_train, y_test = y

    # specify default model
    if model is None:
        model = MultiOutputRegressor(
            Ridge(fit_intercept=False, alpha=0.5, **kwargs))

    if isinstance(model, MultiOutputRegressor):
        # fit model on training data
        model.fit(x_train, y_train)

        # calculate model scores on test data
        if metric[0] == 'corrcoef':
            y_pred = model.predict(x_test)
            n_outputs = y_pred.shape[1]

            corrcoef = []
            for output in range(n_outputs):
                corrcoef.append(
                    np.abs((np.corrcoef(y_test[:, output], y_pred[:, output])[0][1])))

            score = [np.sum(corrcoef)]

    else:
        y_pred = model.predict(x_test)
        try:
            # fall back option is to use sklearn.metrics function
            func = getattr(metrics, metric)
        except AttributeError:
            # finally, try functions from current module
            func = getattr(sys.modules[__name__], metric)
        metric_value = func(y_test, y_pred)
    return metric_value, model

def classification(x, y, model=None, metric=['score'], **kwargs):

    """
    Binary classification tasks
    # TODO
    """

    if metric != 'score':
        raise NotImplementedError(
            'This metric is not yet implemented to evaluate the current model.')

    # pop variables from kwargs
    sample_weight_train, sample_weight_test = kwargs.pop(
        'sample_weight', (None, None))

    x_train, x_test = x
    y_train, y_test = y

    # specify default model
    if model is None:
        model = RidgeClassifier(alpha=0.0, fit_intercept=True, **kwargs)

    # fit model on training data
    model.fit(x_train, y_train, sample_weight_train)

    # calculate model scores on test data
    if metric[0] == 'score':
        score = [model.score(x_test, y_test, sample_weight_test)]

    # # confusion matrix
    # ConfusionMatrixDisplay.from_predictions(y_test, model.predict(x_test))
    # plt.show()
    # plt.close()

    return score, model



def multiClassClassification(x, y, model=None, metric=['score'], **kwargs):

    """
    Multi-class Classification tasks
    # TODO
    """

    if metric != 'score':
        raise NotImplementedError(
            'This metric is not yet implemented to evaluate the current model.')

    # pop variables from kwargs
    sample_weight_train, sample_weight_test = kwargs.pop(
        'sample_weight', (None, None))

    x_train, x_test = x
    y_train, y_test = y

    # specify default model
    if model is None:
        model = OneVsRestClassifier(RidgeClassifier(
            alpha=0.0, fit_intercept=False, **kwargs))

    if isinstance(model, OneVsRestClassifier):
        # select decision time points
        idx_train = np.nonzero(y_train)
        idx_test = np.nonzero(y_test)

        # fit model on training data (OneVsRestClassifier does not support sample weights)
        model.fit(x_train[idx_train], y_train[idx_train])

        # calculate model scores on test data
        if metric[0] == 'score':
            score = [model.score(x_test[idx_test], y_test[idx_test])]
    else:
        # fit model on training data
        model.fit(x_train, y_train, sample_weight_train)

        # calculate model scores on test data
        if metric[0] == 'score':
            score = [model.score(x_test, y_test, sample_weight_test)]

    # # confusion matrix
    # ConfusionMatrixDisplay.from_predictions(y_test[idx_test], model.predict(x_test[idx_test]))
    # plt.show()
    # plt.close()

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     cm = metrics.confusion_matrix(y_test[idx_test], model.predict(x_test[idx_test]))
    #     score = np.sum(np.diagonal(cm))/np.sum(cm)  # turned out to be equivalent to the native sklearn score

    return score, model

def multiOutputRegression(x, y, model=None, metric='corrcoef', **kwargs):
    """
    Multiple output regression tasks
    # TODO
    """

    # pop variables from kwargs
    sample_weight_train, sample_weight_test = kwargs.pop(
        'sample_weight', (None, None))

    x_train, x_test = x
    y_train, y_test = y

    # specify default model
    if model is None:
        model = MultiOutputRegressor(
            Ridge(fit_intercept=False, alpha=0.5, **kwargs))

    if isinstance(model, MultiOutputRegressor):
        # fit model on training data
        model.fit(x_train, y_train)

        # calculate model metric on test data
        if metric == 'corrcoef':
            y_pred = model.predict(x_test)
            n_outputs = y_pred.shape[1]

            metric_value = []
            for output in range(n_outputs):
                metric_value.append(corrcoef(y_test[:, output], y_pred[:, output]))

            metric_value = np.sum(metric_value)
        else:
            raise NotImplementedError(
                'This metric is not yet implemented to evaluate the current model.')
    else:
        # fit model on training data
        model.fit(x_train, y_train, sample_weight_train)

        # calculate model scores on test data
        if metric == 'score':
            metric_value = model.score(x_test, y_test, sample_weight_test)
        else:
            raise NotImplementedError(
                'This metric is not yet implemented to evaluate the current model.')

    return metric_value, model

def multiOutputClassification(x, y, model=None, metric=['score'], **kwargs):

    """
    Multiple output (binary and multi-class) classification tasks
    # TODO
    """

    if metric != 'score':
        raise NotImplementedError(
            'This metric is not yet implemented to evaluate the current model.')

    # pop variables from kwargs
    sample_weight_train, sample_weight_test = kwargs.pop(
        'sample_weight', (None, None))

    x_train, x_test = x
    y_train, y_test = y

    # specify default model
    if model is None:
        model = MultiOutputClassifier(RidgeClassifier(
            alpha=0.5, fit_intercept=True, **kwargs))

    if isinstance(model, MultiOutputClassifier):
        # fit model on training data (MultiOutputClassifier does not support sample weights)
        model.fit(x_train, y_train)

        # calculate model scores on test data

        if metric[0] == 'score':
            score = [model.score(x_test, y_test)]
    else:
        # fit model on training data
        model.fit(x_train, y_train, sample_weight_train)

        # calculate model scores on test data
        if metric[0] == 'score':
            score = [model.score(x_test, y_test, sample_weight_test)]

    return score, model


def select_model(y):
    """
    Select the right model depending on the nature of the target
    variable
    # TODO
    """
    if y.dtype in [np.float32, np.float64]:
        if y.squeeze().ndim == 1:
            return regression  # regression
        else:
            return multiOutputRegression  # multilabel regression

    elif y.dtype in [np.int32, np.int64]:
        if y.squeeze().ndim == 1:
            if len(np.unique(y)) == 2:  # binary classification
                return classification
            else:
                return multiClassClassification  # multiclass classification
        else:
            return multiOutputClassification  # multilabel and/or multiclass classification


def run_task(reservoir_states, target, metric, **kwargs):
    """
    # TODO
    Function that calls the method to run the task specified by 'task'

    Parameters
    ----------
    task : {'regression', 'classification'}
    reservoir_states : tuple of numpy.ndarrays
        simulated reservoir states for training and test; the shape of each
        numpy.ndarray is n_samples, n_reservoir_nodes
    target : tuple of numpy.ndarrays
        training and test targets or output labels; the shape of each
        numpy.ndarray is n_samples, n_labels
    kwargs : other keyword arguments are passed to one of the following
        functions:
            memory_capacity_task(); delays=None, t_on=0
            pattern_recognition_task(); pttn_lens

    Returns
    -------
    df_res : pandas.DataFrame
        data frame with task scores
    """

    # print('\n PERFORMING TASK ...')

    # verify dimensions of x and y
    x_train, x_test, y_train, y_test = check_xy_dims(
        x=reservoir_states, y=target)

    # select training model
    func = select_model(y=y_train)

    
    # make metric a list to enable different metrics on the same model
    if not isinstance(metric, list):
        metric = [metric]

    # fit model
    score, model = func(x=(x_train, x_test), y=(
        y_train, y_test), metric=metric, **kwargs)
    print(f'\t\t {metric[0]} = {score[0]}')

    df_res = pd.DataFrame(
        data={val: score[i] for i, val in enumerate(metric)}, index=[0])

    return df_res, model

if __name__ == '__main__':

    # X = np.random.rand(100,50)

    func = getattr(metrics,'accuracy_score')
    print(func)
    print(type(func))

    y_pred = np.random.randint(0, 2, (100)) # binary, single output
    y_test  = np.random.randint(0, 2, (100)) # binary, single output
    print(func(y_pred, y_test))

    y_pred = np.random.randint(0, 10, (100)) # multi-class, single output
    y_test  = np.random.randint(0, 10, (100)) # multi-class, single output
    print(func(y_pred, y_test))

    y_pred = np.random.randint(0, 2, (100,5)) # binary, multi output = multilabel
    y_test = np.random.randint(0, 2, (100,5)) # binary, multi output = multilabel
    print(func(y_pred, y_test))

    # y_pred = np.random.randint(0, 10, (100,5)) # multi-class, multi output
    # y_test = np.random.randint(0, 10, (100,5)) # multi-class, multi output
    # print(func(y_pred, y_test))

    func = getattr(metrics, 'mean_squared_error')
    # func = getattr(metrics,'r2')
    print(func)
    print(type(func))

    y_pred = np.random.rand(100) # regression, single output
    y_test = np.random.rand(100) # regression, single output
    print(func(y_pred, y_test))

    y_pred = np.random.rand(100,5) # regression, multi output
    y_test = np.random.rand(100,5) # regression, multi output
    print(func(y_pred, y_test))
