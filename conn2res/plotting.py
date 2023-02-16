# -*- coding: utf-8 -*-
"""
Plotting functions

@author: Estefany Suarez
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 20})
import numpy as np
from numpy.linalg import svd, norm
import math
from datetime import date

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
today = date.today()
FIG_DIR = os.path.join(PROJ_DIR, 'figs', today.strftime("%d%b%Y"))
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)

class PCA:
    """
    Class that represents a simplified PCA object

    key features:
    - specifying the index of principal components we want to keep
    - principal components either unnormalized/normalized by singular values

    TODO
    """

    def __init__(self, idx_pcs=None, n_pcs=None, **kwargs):
        """
        Constructor class for time series
        """
        # indexes of principal components to keep
        if idx_pcs is not None:
            if isinstance(idx_pcs, list):
                idx_pcs = np.array(idx_pcs)
            if isinstance(idx_pcs, int):
                idx_pcs = np.array([idx_pcs])
            self.idx_pcs = idx_pcs
            self.n_pcs = len(self.idx_pcs)

        # number of principal components to keep
        if n_pcs is not None:
            self.setdefault('n_pcs', n_pcs)

    def setdefault(self, attribute, value):
        # add attribute (with given value) if not existing
        if not hasattr(self, attribute):
            setattr(self, attribute, value)

    def fit(self, data, full_matrices=False, **kwargs):
        # fit PCA
        self.u, self.s, self.vh = svd(data, full_matrices=full_matrices)

        # set number of principal components if not existing
        self.setdefault('n_pcs', self.s.size)

        # set indexes of principal components if not existing
        self.setdefault('idx_pcs', np.arange(self.n_pcs))

        return self

    def transform(self, data, normalize=False, **kwargs):
        # transform data into principal components
        pc = (data @ self.vh.T[:, self.idx_pcs]).reshape(-1, self.n_pcs)

        # normalize principal components by singular values (loop for efficiency)
        if normalize == True:
            for i in range(self.n_pcs):
                pc[:, i] /= self.s[self.idx_pcs[i]]

        return pc

    def fit_transform(self, data, **kwargs):
        # fit PCA
        self.fit(data, **kwargs)

        # transform data into principal components
        return self.transform(data, **kwargs)


def plot_task(x, y, title, num=1, figsize=(12, 10), savefig=False, show=False, block=True):

    fig = plt.figure(num=num, figsize=figsize)
    ax = plt.subplot(111)

    # xlabels, ylabels
    try:
        x_labels = [f'I{n+1}' for n in range(x.shape[1])]
    except:
        x_labels = 'I1'
    try:
        y_labels = [f'O{n+1}' for n in range(y.shape[1])]
    except:
        y_labels = 'O1'

    plt.plot(x[:], label=x_labels)
    plt.plot(y[:], label=y_labels)
    plt.legend()
    plt.suptitle(title)

    sns.despine()
    if savefig:
        fig.savefig(fname=os.path.join(FIG_DIR, f'{title}_io.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    if show: plt.show(block=block)
    # plt.close()


def plot_performance_curve(df, by_age=False, title=None, x='alpha', y='score', ylabel="R squared", hue=None, hue_order=None, palette=None, ylim=None, chance_perf=0.5,
                           xlim=None, legend=True,ticks=None, markers=False,norm=False, norm_var=None, figsize=(19.2, 9.43), savefig=False, show=False, block=True, **kwargs):

    sns.set(style="ticks", font_scale=1.0)
    fig = plt.figure(figsize=figsize)

    if palette is None and hue is not None:
        n_modules = len(np.unique(df[hue]))
        palette = sns.color_palette('husl', n_modules+1)[:n_modules]

        if hue_order is None and "WT" in list(np.unique(df[hue])):
            hue_order = ["WT", "HE", "KO"]

    if norm:
        df[y] = df[y] / df[norm_var]

        max_score = max(df[y])
        min_score = min(df[y])
        df[y] = (df[y] - min_score) / (max_score - min_score)
    
    if by_age:     
        ages = kwargs.get('ages')
        age_point = 1
        plot_legend = True
        for DIV in ages:
            ax = plt.subplot(2, 3, age_point)
            # ax = plt.subplot(2, 2, age_point)
            age_data = df[df['age'] == DIV]
            plot = sns.lineplot(data=age_data, x=x, y=y,
                        hue=hue,
                        hue_order=hue_order,
                        palette=palette,
                        markers=True,
                        legend=plot_legend,
                        ax=ax)
            plot.set_title(f'DIV{DIV}', loc='left')
            if plot_legend:
                sns.move_legend(ax, loc='center', bbox_to_anchor=(2.5, -0.5))
                plot_legend = False # plot legend only once
            age_point += 1

    else:
        ax = fig.add_subplot()
        plot = sns.lineplot(data=df, x=x, y=y,
                        hue=hue,
                        hue_order=hue_order,
                        palette=palette,
                        markers=True,
                        legend=legend,
                        ax=ax)
    if markers:
        plot = sns.scatterplot(data=df, x=x, y=y,
                        hue=hue,
                        hue_order=hue_order,
                        palette=palette,
                        ax=ax)
    
    if chance_perf is not None:
        plot.axhline(chance_perf,color='r',linestyle='--',linewidth='2.0')

    plot.axvline(1.0,color='g',linestyle='--',linewidth='2.0')
    if ylim is not None: plt.ylim(ylim)
    plt.legend(title=hue, loc='upper right')

    if xlim is not None: plt.xlim(xlim)
    plt.xlabel(x, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    if ticks is not None: plt.xticks(ticks, fontsize=20)
    else: plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    sns.despine()

    fig.suptitle(title, fontsize=24)
    
    if savefig:
        fig.savefig(fname=os.path.join(FIG_DIR, f'{title}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    if show: plt.show(block=block)
    # plt.close()

def plot_perf_reg(df, x, by_regime=False, title=None, y='score', xlim=None, ticks=None, ylim=None, hue='genotype', hue_order=["WT", "HE", "KO"],
    chance_perf=0.5,figsize=(19.2, 9.43),savefig=False, show=False, block=True, **kwargs):

    fig = plt.figure(figsize=figsize)

    if hue is not None:
        n_modules = len(np.unique(df[hue]))
        palette = sns.color_palette('husl', n_modules+1)[:n_modules]

        if hue_order and "WT" in list(np.unique(df[hue])):
                hue_order = ["WT", "HE", "KO"]
    if by_regime:
        regimes = kwargs.get('regimes')
        genotypes = kwargs.get('genotypes')
        regime_n = 1
        plot_legend = False
        for regime in regimes:
            ax = plt.subplot(1, 3, regime_n)
            regime_data = df[df['regime'] == regime].groupby('name').mean()
            regime_data['genotype'] = genotypes
            g = sns.scatterplot(data=regime_data,
                x=x,
                y=y,
                markers=True,
                legend=plot_legend,
                hue=hue,
                hue_order=hue_order,
                palette=palette,
                s=100, ax=ax)
            if plot_legend:
                sns.move_legend(ax, loc='upper right')#, bbox_to_anchor=(2.5, 0.5))
                plot_legend = False # plot legend only once
            g.set_title(regime, loc='center')
            regime_n += 1

    else:
        g = sns.scatterplot(data=df,
                x=x,
                y=y,
                markers=True,
                legend=True,
                hue=hue,
                hue_order=hue_order,
                palette=palette,
                s=100)
                #,ci=None)

        if xlim is not None: g.set_xlim(left=xlim[0], right=xlim[1])
        plt.ylim(ylim)
        plt.yticks(fontsize=20)
        if ticks is not None: plt.xticks(ticks, fontsize=20)
        else: plt.xticks(fontsize=20)
        plt.xlabel(x,fontsize=24)
        plt.ylabel(y,fontsize=24)

        if chance_perf is not None and y=='score':
            g.axhline(chance_perf,color='r',linestyle='--',linewidth='2.0')
        if x=='rho': g.axvline(1.0,color='g',linestyle='--',linewidth='2.0')

        fig = g.get_figure()
    sns.despine()

    if y=='percentage_change': g.axhline(0,color='r',linestyle='--',linewidth='2.0')

    # g.suptitle(title, fontsize=24)

    if savefig:
        fig.savefig(fname=os.path.join(FIG_DIR, f'{title}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    if show: plt.show(block=block)

def plot_time_series(x, feature_set='orig', idx_features=None, n_features=None, sample=None, xlim=[0, 150], ylim=None, xticks=None, yticks=None,
                     cmap=None, style='solid', scaler=1, num=1, figsize=(19.2, 9.43), subplot=None, title=None, fname=None,
                     legend_label=None, savefig=False, show=False, block=True, **kwargs):

    # transform data
    x = transform_data(x, feature_set, idx_features=idx_features,
                       n_features=n_features, **kwargs)

    # open figure and create subplot
    plt.figure(num=num, figsize=figsize)
    if subplot is None:
        subplot = (1, 1, 1)
    plt.subplot(*subplot)

    # plot data
    if x.size > 0:
        if sample is None:
            plt.plot(x,linestyle=style)
        else:
            t = np.arange(*sample)
            if cmap is None:
                plt.plot(t, x[t],linestyle=style)
            else:
                for i, _ in enumerate(t[:-1]):
                    plt.plot(t[i:i+2], x[t[i:i+2]],
                             color=getattr(plt.cm, cmap)(255*i//np.diff(sample)))
                             
        # add x and y limits
        plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        # plot legend
        if legend_label is not None:
            if x.ndim == 2 and x.shape[1] > 1:
                legend = [f'{legend_label} {n+1}' for n in range(x.shape[1])]
            else:
                legend = [f'{legend_label}']
            try:  # quick fix to get previously plotted legends
                lg = plt.gca().lines[-1].axes.get_legend()
                legend = [text.get_text() for text in lg.texts] + legend
            except:
                pass
            if len(legend) <= 5:
                plt.legend(legend, loc='upper right', fontsize=12)
            else:
                plt.legend(legend, loc='upper right', fontsize=12, ncol=2)
            # plt.legend(legend, loc='upper center', bbox_to_anchor=(
            #     0.5, 1.05), fontsize=10, ncol=len(legend))

    # set xtick/ythick fontsize
    if xticks is not None: plt.xticks(xticks,fontsize=22)
    if yticks is not None: plt.yticks(yticks, fontsize=22)
    # plt.yticks([y_min, y_max], fontsize=22)

    # add title
    if title is not None:
        plt.title(f'{title} time course', fontsize=22)

    # set tight layout in case there are different subplots
    plt.tight_layout()

    if savefig:
        plt.savefig(fname=os.path.join(FIG_DIR, f'{title}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    if show: plt.show(block=block)
    # plt.close()

def plot_time_series_raster(x, feature_set='orig', idx_features=None, n_features=None, xlim=None,
                            cmap='viridis', cbar_norm='norm', cmap_lim=None,cbar_pad=0.02,
                            num=1, figsize=(19.2, 9.43), subplot=None, title=None,
                            savefig=False, block=True, **kwargs):

    # transform data
    x = transform_data(x, feature_set, idx_features=idx_features,
                       n_features=n_features, **kwargs)

    # open figure and create subplot
    fig = plt.figure(num=num, figsize=figsize)
    if subplot is None:
        subplot = (1, 1, 1)
    ax = plt.subplot(*subplot)

    # plot data
    if x.size > 0:
        plt.imshow(x.T, aspect='auto')
        # set xtick/ytick fontsize
        ax.tick_params(axis='both', labelsize=22)

        # set tight layout in case there are different subplots
        plt.tight_layout()

        # add colorbar
        if cmap_lim is None:
            vmin = x[x != 0].min()
            vmax = x.max()
        else:
            vmin = cmap_lim[0]
            vmax = cmap_lim[1]
        if cbar_norm == 'lognorm' and vmax/vmin > 10:
            # use log scale if data spread more than 1 magnitude
            pcm = ax.pcolormesh(x.T, norm=colors.LogNorm(
                vmin=vmin, vmax=vmax), cmap=getattr(plt.cm, cmap))
        else:
            # use linear scale by default
            pcm = ax.pcolormesh(x.T, cmap=getattr(plt.cm, cmap),
                                vmin=vmin, vmax=vmax)

        # divider = make_axes_locatable(ax)
        # cbar_width = 1 - x.shape[0]/xlim[-1] - cbar_pad
        # cax = divider.append_axes(
        #     "right", f'{cbar_width*100:.2f}%', pad=f'{cbar_pad*100:.2f}%')

        plt.colorbar(pcm) # cax=cax

    # add title
    if title is not None:
        plt.title(f'{title} time course', fontsize=22)

    if savefig:
        plt.savefig(fname=os.path.join(FIG_DIR, f'{title}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    # plt.show(block=block)

def boxplot(x, y, df, ylabel="R squared", by_age=False, ages=None,regimes=None, genotypes=None,yticks=None, xticks=None, title=None, hue=None, hue_order=None, palette=None, orient='v', \
            width=0.5, linewidth=1, xlim=None, ylim=None,
            legend=True, figsize=(19.2, 9.43), show=False, savefig=False, block=True, **kwargs):

    fig = plt.figure(figsize=figsize)

    if palette is None and hue is not None:
        n_modules = len(np.unique(df[hue]))
        palette = sns.color_palette('husl', n_modules+1)[:n_modules]

    if by_age:     
        age_point = 1
        # plot_legend = True
        for DIV in ages:
            ax = plt.subplot(2, 3, age_point)
            # ax = plt.subplot(2, 2, age_point)
            age_data = df[df['age'] == float(DIV)]
            plot = sns.boxplot(x=x, y=y,
                        data=age_data,
                        palette=palette,
                        hue=hue,
                        hue_order=hue_order,
                        width=width,
                        linewidth=linewidth,
                        ax=ax,
                        **kwargs
                        )
            plot.set_title(f'DIV{DIV}', loc='left')
            # if plot_legend:
            #     sns.move_legend(ax, loc='center', bbox_to_anchor=(2.5, -0.5))
            #     plot_legend = False # plot legend only once
            plot.legend_.remove()
            age_point += 1
    else:
        ax = plt.subplot()
        plot = sns.boxplot(x=x, y=y,
                        data=df,
                        palette=palette,
                        hue=hue,
                        hue_order=hue_order,
                        orient=orient,
                        width=width,
                        linewidth=linewidth,
                        ax=ax,
                        **kwargs
                        )
        if legend: ax.legend(fontsize=24, frameon=True, ncol=1, loc='best')
        if title is not None: ax.set_title(title, fontsize=24)
    #  for patch in axis.artists:
    #      r, g, b, a = patch.get_facecolor()
    #      patch.set_facecolor((r, g, b, 0.9))
    
    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)

    plt.xlabel(x, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    if xticks is not None: plt.xticks(xticks, fontsize=20)
    else: plt.xticks(fontsize=20)
    if yticks is not None: plt.yticks(yticks, fontsize=20)
    else: plt.yticks(fontsize=20)

    if y=='percentage_change': plot.axhline(0,color='r',linestyle='--',linewidth='2.0')

    sns.despine()

    # set tight layout in case there are different subplots
    plt.tight_layout()

    if savefig:
        plt.savefig(fname=os.path.join(FIG_DIR, f'{title}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    if show: plt.show(block=block)

def transform_data(data, feature_set, idx_features=None, n_features=None, scaler=None, model=None, **kwargs):

    if feature_set == 'pc':
        # transform data into principal components
        data = PCA(n_pcs=n_features).fit_transform(data, **kwargs)

    elif feature_set == 'rnd':
        # update default number of features
        if n_features is None:
            n_features = 1

        # choose features randomly
        data = data[:, np.random.choice(
            np.arange(data.shape[1]), size=n_features)]

    elif feature_set == 'decfun':
        # calculate decision function using model fitted on time series
        data = model.decision_function(data)

    elif feature_set == 'pred':
        # calculate predicted labels
        data = model.predict(data)[:, np.newaxis]

    elif feature_set == 'coeff':
        # update default number of features
        if n_features is None:
            n_features = 5

        # get coefficient from model
        if model.coef_.ndim > 1:
            idx_class = kwargs.get('idx_class', 0)
            coef = model.coef_[idx_class, :]
        else:
            coef = model.coef_

        # choose features that correspond to largest absolute coefficients
        idx_coef = np.argsort(np.absolute(coef))
        if sum(coef != 0) > n_features:
            # use top 5 features
            idx_coef = idx_coef[-n_features:]
        else:
            # use <5 non-zero features
            idx_coef = np.intersect1d(idx_coef, np.where(coef != 0)[0])

        # scale time series with coefficients
        data = data[:, idx_coef]
        if data.size > 0:
            data = data @ np.diag(coef[idx_coef])
            # data = np.sum(
            #     data @ np.diag(coef[idx_coef]), axis=1).reshape(-1, 1)

    # select given features
    if idx_features is not None:
        data = data[:, idx_features]

    # scale features
    if scaler is not None:
        if scaler == 'l1-norm':
            scaler = norm(data, ord=1, axis=0)
        if scaler == 'l2-norm':
            scaler = norm(data, ord=2, axis=0)
        elif scaler == 'max':
            scaler = norm(data, ord=np.inf, axis=0)
        elif isinstance(scaler, int):
            scaler = np.array([int])
        data /= scaler

    return data


def plot_mackey_glass_phase_space(x, y, sample=None, xlim=None, ylim=None, subplot=None, cmap=None,
                                  num=1, figsize=(13, 5), title=None, fname='phase_space', savefig=False, show=False, block=False):
    # open figure and create subplot
    plt.figure(num=num, figsize=figsize)
    if subplot is None:
        subplot = (1, 1, 1)
    plt.subplot(*subplot)

    # plot data
    if sample is None:
        plt.plot(x)
    else:
        t = np.arange(*sample)
        if cmap is None:
            plt.plot(t, x[t])
        else:
            for i, _ in enumerate(t[:-1]):
                plt.plot(x[t[i:i+2]], y[t[i:i+2]],
                         color=getattr(plt.cm, cmap)(255*i//np.diff(sample)))

    # add x and y limits
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.xlim(ylim)

    # set xtick/ythick fontsize
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # add title
    if title is not None:
        plt.title(f'{title} phase space', fontsize=22)

    # set tight layout in case there are different subplots
    plt.tight_layout()

    if savefig:
        plt.savefig(fname=os.path.join(FIG_DIR, f'{fname}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    if show: plt.show(block=block)
    # plt.close()