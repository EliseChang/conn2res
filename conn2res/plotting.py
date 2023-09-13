# -*- coding: utf-8 -*-
"""
Plotting functions

@author: Estefany Suarez
"""
import os
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statannotations.Annotator import Annotator
import scipy.stats
import statistics
import numpy as np
import pandas as pd
from numpy.linalg import svd, norm
import math
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
today = date.today()
FIG_DIR = os.path.join(PROJ_DIR, 'figs', today.strftime("%d%b%Y"))
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)
dpi = 300
sns.set(font="Arial", style='ticks', font_scale=1.0)



# age_marker_dict = {"21": "o",
#                    "28": "s",
#                    "35": "v"}
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

def get_iqr(x):
    min = np.quantile(x, 0.25)
    max = np.quantile(x, 0.75)
    return (min, max)

def get_confints(df, indexvar, withinvars, x, y, hue, ci_lims):
    x_cats = pd.Categorical(np.unique(df[x]))
    x_n = len(x_cats)
    groups = pd.unique(df[hue])
    hue_n = len(groups)
    confint = np.zeros((x_n, hue_n, 2))

    means = df.groupby(withinvars + [hue]).mean().reset_index()
    
    for i, group in enumerate(groups):
        group_df = df[df[hue] == group]
        group_ci = ci_within(group_df, indexvar, withinvars, y)
        for j, cat in enumerate(x_cats):
            mean = means[(means[x] == cat) & (means[hue] == group)][y].values[0]
            ci = group_ci[group_ci[x] == cat]['sem'].values[0]
            min = mean - ci
            max = mean + ci
            if ci_lims is not None:
                if min < ci_lims[0]: min = ci_lims[0]
                if max > ci_lims[1]: max = ci_lims[1]
            confint[j, i, 0] = min
            confint[j, i, 1] = max

    return confint

def ci_within(df, indexvar, withinvars, measvar, confint=0.95,
                      copy=True):
    
    """ Compute CI / SEM correction factor
    Morey 2008, Cousinaueu 2005, Loftus & Masson, 1994
    Also see R-cookbook http://goo.gl/QdwJl
    Note. This functions helps to generate appropriate confidence
    intervals for repeated measure designs.
    Standard confidence intervals are are computed on normalized data
    and a correction factor is applied that prevents insanely small values.
    df : instance of pandas.DataFrame
        The data frame objetct.
    indexvar : str
        The column name of of the identifier variable that representing
        subjects or repeated measures
    withinvars : str | list of str
        The column names of the categorial data identifying random effects
    measvar : str
        The column name of the response measure
    confint : float
        The confidence interval
    copy : bool
        Whether to copy the data frame or not.
    """
    if copy:
        df = df.copy()

    # Apply Cousinaueu's method:
    # compute grand mean
    mean_ = df[measvar].mean()

    # compute subject means
    subj_means = df.groupby(indexvar)[measvar].mean().values
    for subj, smean_ in zip(df[indexvar].unique(), subj_means):
        # center
        df[measvar][df[indexvar] == subj] -= smean_
        # add grand average
        df[measvar][df[indexvar] == subj] += mean_

    def sem(x):
        return x.std() / np.sqrt(len(x))

    def ci(x):
        se = sem(x)
        return se * scipy.stats.t.interval(confint, len(x) - 1)[1]

    aggfuncs = [np.mean, np.std, sem, ci]
    out = df.groupby(withinvars)[measvar].agg(aggfuncs)
    out.reset_index(inplace=True)
    # # compute & apply correction factor
    # n_within = np.prod([len(df[k].unique()) for k in withinvars],
    #                    dtype= df[measvar].dtype)
    # cf = np.sqrt(n_within / (n_within - 1))
    # for k in ['sem', 'std', 'ci']:
    #     out[k] *= cf

    return out


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


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
    plt.suptitle('')

    sns.despine()
    if savefig:
        fig.savefig(fname=os.path.join(FIG_DIR, f'{title}_io.png'),
                    transparent=True, bbox_inches='tight', dpi=dpi)
    if show: plt.show(block=block)
    # plt.close()


def plot_performance_curve(df, title=None, x='alpha', y='score', ylabel=None, hue=None, hue_order=None, style=None, style_order=None, palette=None, ylim=None, chance_perf=None,
                           xlim=None, legend=True,xticks=None, yticks=None,norm=False, norm_var=None, figsize=(19.2, 9.43), savefig=False, format=['png'], show=False, block=True, **kwargs):

    fig = plt.figure(figsize=figsize)

    if palette is None and hue is not None:
        n_modules = len(np.unique(df[hue]))
        palette = sns.color_palette('husl', n_modules+1)[:n_modules]

        # if hue_order is None and "WT" in list(np.unique(df[hue])):
        #     hue_order = ["WT", "HET", "KO"]

    if norm:
        df[y] = df[y] / df[norm_var]

        max_score = max(df[y])
        min_score = min(df[y])
        df[y] = (df[y] - min_score) / (max_score - min_score)

    ax = fig.add_subplot()
    plot = sns.lineplot(data=df, x=x, y=y,
                    hue=hue,
                    hue_order=hue_order,
                    style=style,
                    style_order=style_order, 
                    palette=palette,
                    markers=True,
                    legend=legend,
                    linewidth=2,
                    ax=ax)

    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)

    if chance_perf is not None:
        plot.axhline(chance_perf,color='r',linestyle='--',linewidth='2.0')
    if x == 'alpha': plot.axvline(1.0,color='gray',linestyle='--',linewidth='2.0')
    if legend:
        l = plt.legend(title=hue, loc='upper right', fontsize=32, frameon=False)
        plt.setp(l.get_title(), fontsize='40')

    plt.xlabel(x, fontsize=48)
    plt.ylabel(ylabel, fontsize=48)
    if xticks is not None: plt.xticks(xticks, fontsize=40)
    else: plt.xticks(fontsize=40)
    if yticks is not None: plt.yticks(yticks, fontsize=40)
    else: plt.yticks(fontsize=40)
    sns.despine()

    fig.suptitle('')
    
    if savefig:
        for ext in format:
            plt.savefig(fname=os.path.join(FIG_DIR, f'{title}.{ext}'),
                        transparent=True, bbox_inches='tight', dpi=dpi)
    if show: plt.show(block=block)
    # plt.close()

def plot_perf_reg(df, x, title=None, y='score', ylabel=None, xlim=None, xticks=None, yticks=None, ylim=None, xlabel=None,
                  hue=None, hue_order=None, size=None,size_order=None,style=None,palette=None,legend=True,
    chance_perf=0.5, figsize=(19.2, 9.43),savefig=False, show=False, block=True,format=['png'], **kwargs):

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1,1,1)
    if hue is not None:
        n_modules = len(np.unique(df[hue]))
        palette = sns.color_palette('husl', n_modules+1)[:n_modules]

        if hue_order and "WT" in list(np.unique(df[hue])):
                hue_order = ["WT", "HE", "KO"]

    if size == 'age': size_order=kwargs.get('ages')
    if style == 'group': markers = {"stimulation": "s", "control": "X"}
    elif style == 'Age': markers = {14: "o", 21: "^", 28: "X", 35: "s"}
    else: markers=True

    g = sns.relplot(data=df,
                    kind='scatter',
                    x=x,
                    y=y,
                    legend=legend,
                    hue=hue,
                    hue_order=hue_order,
                    s=100,
                    linewidth=0.5,
                    edgecolor='black',
                    style=style,
                    markers=markers,
                    palette=palette)
    # g.map(sns.regplot, data=df,x=x, scatter=False,
    #                 y=y,color='black')

    if xlim is not None: ax.set_xlim(left=xlim[0], right=xlim[1])
    plt.ylim(ylim)
    if yticks is not None: plt.yticks(yticks, fontsize=26)
    else: plt.yticks(fontsize=32)
    if xticks is not None: plt.xticks(xticks, fontsize=26)
    else: plt.xticks(fontsize=32)
    if xlabel is not None: plt.xlabel(xlabel,fontsize=30)
    else: plt.xlabel(x,fontsize=40)
    if ylabel is not None: plt.ylabel(ylabel,fontsize=30)
    else: plt.ylabel(x,fontsize=40)

    if legend:
        if size is not None:
            l = plt.legend(fontsize='32', loc="upper left", bbox_to_anchor=(1, 1))
        else: l = plt.legend(title=hue, title_fontsize='32',fontsize='32', loc="upper left", bbox_to_anchor=(1, 1))
        plt.setp(l.get_title(), fontsize='40')

    # if hue=='genotype':
    #     l.legendHandles[1].set_sizes([50])
    #     l.legendHandles[2].set_sizes([50])
    #     l.legendHandles[3].set_sizes([50])

    if y=='percentage_change': g.refline(y=0,color='r',linestyle='--',linewidth='2.0')
    if chance_perf is not None and x=='score':
        g.refline(x=chance_perf,color='r',linestyle='--',linewidth='2.0')
    if x=='rho':
        g.refline(x=1.0, color='gray',linestyle='--',linewidth='2.0')
    if y=='rho':
        g.refline(y=1.0, color='gray',linestyle='--',linewidth='2.0')

    # fig = g.get_figure()
    # plt.title(title, fontsize=28)

    sns.despine()

    if savefig:
        for ext in format:
            plt.savefig(fname=os.path.join(FIG_DIR, f'{title}.{ext}'),
                        transparent=True, bbox_inches='tight', dpi=dpi)
    if show: plt.show(block=block)

def plot_line_plot(df, x, title=None, y='score', ylabel='R squared', xlim=None, ticks=None, ylim=None, xlabel=None,
                  hue='genotype', hue_order=["WT", "HE", "KO"], chance_perf=0.5,
                  figsize=(19.2, 9.43),savefig=False, show=False, block=True):

    fig = plt.figure(figsize=figsize)
    p = sns.lineplot(df, x=x, y=y, hue=hue, hue_order=hue_order)

    if xlim is not None: g.set_xlim(left=xlim[0], right=xlim[1])
    plt.ylim(ylim)
    plt.yticks(fontsize=26)
    if ticks is not None: plt.xticks(ticks, fontsize=26)
    else: plt.xticks(fontsize=26)
    if xlabel is not None: plt.xlabel(xlabel,fontsize=30)
    else: plt.xlabel(x,fontsize=30)
    plt.ylabel(ylabel,fontsize=30)

    if chance_perf is not None and y=='score':
        p.axhline(chance_perf,color='r',linestyle='--',linewidth='2.0')

    fig = p.get_figure()
    # plt.title(title, fontsize=28)

    sns.despine()

    if savefig:
        fig.savefig(fname=os.path.join(FIG_DIR, f'{title}.png'),
                    transparent=True, bbox_inches='tight', dpi=dpi)
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
                plt.plot(t, x[t],linestyle=style, color='black')
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
                plt.legend(legend, loc='upper right', fontsize=16)
            else:
                plt.legend(legend, loc='upper right', fontsize=16, ncol=2)
            # plt.legend(legend, loc='upper center', bbox_to_anchor=(
            #     0.5, 1.05), fontsize=10, ncol=len(legend))

    # set xtick/ytick fontsize
    if xticks is not None: plt.xticks(xticks,fontsize=40)
    else: plt.xticks(fontsize=40)
    if yticks is not None: plt.yticks(yticks, fontsize=40)
    else: plt.yticks(fontsize=40)
    # plt.yticks([y_min, y_max], fontsize=22)


    plt.title('')

    # set tight layout in case there are different subplots
    plt.tight_layout()

    if savefig:
        plt.savefig(fname=os.path.join(FIG_DIR, f'{fname}.png'),
                    transparent=True, bbox_inches='tight', dpi=dpi)
    if show: plt.show(block=block)
    # plt.close()

def plot_time_series_raster(x, feature_set='orig', idx_features=None, n_features=None, xlim=None, ylim=None,ticks=None,
                            xlabel="Timestep", ylabel="Electrode", cmap='viridis', cbar_norm='norm', cmap_lim=None,cbar_pad=0.02,
                            num=1, figsize=(19.2, 9.43), subplot=None, title=None, format=['png'],
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
        g = plt.imshow(x.T, aspect='auto')
        # set xtick/ytick fontsize
        ax.tick_params(axis='both', labelsize=26)

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

        cb = plt.colorbar(pcm, pad=0.01) # cax=cax

        cb.ax.tick_params(labelsize=40)
        cb.ax.set_ylabel('Activation state of reservoir unit', fontsize=40, labelpad=5.5)

    if xlim is not None: ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim is not None: ax.set_ylim(left=ylim[0], right=ylim[1])
    plt.yticks(fontsize=40)
    if ticks is not None: plt.xticks(ticks, fontsize=40)
    else: plt.xticks(fontsize=40)
    plt.xlabel(xlabel,fontsize=40)
    plt.ylabel(ylabel,fontsize=40)

    plt.title('')

    if savefig:
        for ext in format:
            plt.savefig(fname=os.path.join(FIG_DIR, f'{title}.{ext}'),
                        transparent=True, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    # plt.show(block=block)

def boxplot(x, y, df, ylab=None, order=None, marker=None, yticks=None, xticks=None, xticklabs=None,
            title=None, hue=None, hue_order=None, palette=None, xlabel=None, facet=None,
            width=0.5, linewidth=1.5, xlim=None, ylim=None, chance_perf=None,
            indexvar='sample', withinvars=['Age'], ci_lims=None,
            stats=False,pairs=None,p_vals=None, dodge=0.5,
            legend=True, figsize=(19.2, 9.43), show=False, savefig=False, format=['png'], block=True, **kwargs):
    
    sns.set(font="Arial", style='ticks', font_scale=1.0)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)

    if palette is None and hue is not None:
        n_modules = len(np.unique(df[hue]))
        palette = sns.color_palette('husl', n_modules+1)[:n_modules]

    plotting_parameters = {'data': df, 
                        'x': x,
                        'y': y,
                        'hue': hue,
                        'hue_order': hue_order,
                        'palette': palette,
                        'linewidth': 0.5,
                        'edgecolor': 'black',
                        'dodge': dodge,
                        'zorder':0,
                        's': 16,
                        'ax': ax}
    # # Calculate corrected CIs
    # if ci_lims is None: ci_lims = ylim
    # if hue is not None:
    #     confint = get_confints(df, indexvar=indexvar, withinvars=withinvars, x=x, y=y, hue=hue, ci_lims=ci_lims)
    # # else: #TODO

    if marker is not None:
        # genotype_color_dict = {"WT": (0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
        #                "HE": (0.5920891529639701, 0.6418467016378244, 0.1935069134991043),
        #                "KO":(0.21044753832183283, 0.6773105080456748, 0.6433941168468681)}

        # p = (
        #     so.Plot(df, x=x, y=y, color=hue)
        #     .add(
        #         so.Dot(pointsize=24, edgecolor="black"),
        #         so.Jitter(0.5),
        #         so.Dodge(),
        #         marker=marker
        #     )
        #     .add(so.Range(color='black', linewidth=3, artist_kws={'capsize': 0.1}), so.Est(errorbar=get_iqr), so.Dodge())
        #     .add(so.Dot(pointsize=32, marker="d",color='black'), so.Agg("median"), so.Dodge())
        #     .scale(
        #         color=so.Nominal(genotype_color_dict)
        #     #     ,x=so.Continuous().tick(at=xticks),
        #     #     y=so.Continuous().tick(at=yticks),
        #         marker=so.Nominal(age_marker_dict)
        #     )
        #     # .limit(x=tuple(xlim),y=tuple(ylim))
        #     # .label(
        #     #     x=xlabel, y=ylab
        #     # )
        # )
        # p.theme({**axes_style("ticks")})
        # p.on(ax).plot()
        # if facet is not None: p.facet(col=facet)
        if marker == 'group': markers = {"stimulation": "s", "control": "X"}
        elif marker == 'Age': markers = {14: "o", 21: "^", 28: "X", 35: "s"}
        elif marker == 'amplitude': markers = {3: "o", 4: "^", 5: "X", 6: "s"}

        g = sns.relplot(data=df,
                    kind='scatter',
                    x=x,
                    y=y,
                    legend=legend,
                    hue=hue,
                    hue_order=hue_order,
                    s=150,
                    linewidth=0.5,
                    edgecolor='black',
                    style=marker,
                    markers=markers,
                    palette=palette)
        for points in g.ax.collections:
            vertices = points.get_offsets().data
            if len(vertices) > 0:
                vertices[:, 0] += np.random.uniform(-0.3, 0.3, vertices.shape[0])
                points.set_offsets(vertices)
            xticks = g.ax.get_xticks()
            g.ax.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5) # the limits need to be moved to show all the jittered dots
        
        pp = g.map(sns.pointplot,
              data=df,x=x, y=y,
                hue=hue,
                hue_order=hue_order,
                join=False,
                estimator='median',
                errorbar=get_iqr,
                markers="d",
                order=order,
                color="black",
                orient='v',
                dodge=dodge,
                errwidth=linewidth,
                scale=2,
                capsize=0.2,
                ax=ax)
        
    else:
        # if hue == x: dodge=False
        # else: dodge=0.5

        pp = sns.pointplot(data=df,x=x, y=y,
                        hue=hue,
                        hue_order=hue_order,
                        join=False,
                        estimator='median',
                        errorbar=get_iqr,
                        markers="d",
                        order=order,
                        color="black",
                        orient='v',
                        dodge=dodge,
                        errwidth=linewidth,
                        scale=3,
                        capsize=0.1,
                        ax=ax)
        
        remove_handles,_ = ax.get_legend_handles_labels()

        sp = sns.stripplot(**plotting_parameters)
        
        all_handles,_ = ax.get_legend_handles_labels()

        if legend:
            ax.legend(handles=[h for h in all_handles if h not in remove_handles],
              labels=hue_order,
              fontsize=32, title=hue, title_fontsize=40, frameon=False, ncol=1, loc='center right', bbox_to_anchor=(1, 0.5))
        else: sp.legend_.remove()

    if stats:
        plotting_parameters.pop('ax')
        if hue == x:
            plotting_parameters['hue'] = None
            plotting_parameters['hue_order'] = None
        annotator = Annotator(ax, pairs, **plotting_parameters)
        annotator.set_pvalues(p_vals)
        annotator.annotate()

    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)

    plt.xlabel(xlabel, fontsize=40)
    plt.ylabel(ylab, fontsize=40)

    if xticks is not None: plt.xticks(xticks, fontsize=32)
    else:
        locs,_= plt.xticks()
        plt.xticks(fontsize=32)
        if xticklabs is not None: plt.xticks(locs,labels=xticklabs,fontsize=32)
    if yticks is not None: plt.yticks(yticks, fontsize=32)
    else: plt.yticks(fontsize=32)

    if y=='percentage_change': plt.axhline(0,color='r',linestyle='--',linewidth='2.0')
    if chance_perf is not None: plt.axhline(chance_perf,color='r',linestyle='--',linewidth='2.0')
    
    sns.despine()

    # set tight layout in case there are different subplots
    plt.tight_layout()

    if savefig:
        for ext in format:
            plt.savefig(fname=os.path.join(FIG_DIR, f'{title}.{ext}'),
                        transparent=True, bbox_inches='tight', dpi=dpi)
    if show: plt.show(block=block)
# def actual_boxplot(x, y, df, ylab=None, order=None, marker=None, yticks=None, xticks=None, xticklabs=None,
#             title=None, hue=None, hue_order=None, palette=None, xlabel=None, facet=None,
#             width=0.5, linewidth=1.5, xlim=None, ylim=None, chance_perf=None,
#             indexvar='sample', withinvars=['Age'], ci_lims=None,
#             stats=False,pairs=None,p_vals=None, dodge=0.5,
#             legend=True, figsize=(19.2, 9.43), show=False, savefig=False, format=['png'], block=True, **kwargs):
    


def parallel_plot(df, y1_var, y2_var, c, hue='genotype', hue_order=['WT','HE','KO'], cmap='viridis', xlabels=None, ylabel=None, ylim=None, chance_perf=None,
                  cb_norm=None, cb_ticks=None, cb_label=None, figsize=(19.2, 9.43), show=False, savefig=False, block=True, title=None, **kwargs):
    
    ncols = len(df[hue].unique())
    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    for subplot_n in range(0, ncols):
        plt.subplot(1, ncols, subplot_n+1)
        data = df[df[hue] == hue_order[subplot_n]].reset_index()
        x1_data = np.ones(len(data[y1_var])) * 0.25 + np.random.uniform(-0.1,0.1, size=len(data[y1_var])) # add jitter
        x2_data = np.ones(len(data[y2_var])) * 0.75 + np.random.uniform(-0.1,0.1, size=len(data[y2_var]))
        ax = axs[subplot_n]

        for y1, y2, x1, x2 in zip(data[y1_var], data[y2_var], x1_data, x2_data):
            ax.plot([x1, x2], [y1, y2], color='black')

        if cb_norm=='log':
            c_data = np.log10(data[c])
            # norm=colors.SymLogNorm(linthresh=0.01, base=10)
        else: norm=None

        s = plt.scatter(x1_data, data[y1_var], s=120, c=c_data, cmap=cmap)
        plt.scatter(x2_data, data[y2_var], s=120, c=c_data, cmap=cmap)
        
        ax.set_xticks([0.25, 0.75], labels=xlabels, fontsize=18)
        ax.set_title(hue_order[subplot_n], loc='center', fontsize=18)

        if chance_perf is not None:
            s.axhline(chance_perf,color='gray',linestyle='--',linewidth='2.0')

        plt.xlim([0,1])
        if ylim is not None: plt.ylim(ylim)
        if ylabel is not None: plt.ylabel(ylabel, fontsize=18)
        plt.yticks(fontsize=18)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # # add colorbar
    # if cb_lim is None:
    #     vmin = df[c][df[c] != 0].min()
    #     vmax = df[c].max()
    # else:
    #     vmin = cb_lim[0]
    #     vmax = cb_lim[1]

    # if cb_norm == 'log' and vmax/vmin > 10:
    #     # use log scale if data spread more than 1 magnitude
    #     pcm = ax.pcolormesh(df[c], norm=colors.Log(
    #         vmin=vmin, vmax=vmax), cmap=getattr(plt.cm, cmap))
    # else:
    #     # use linear scale by default
        # pcm = ax.pcolormesh(df[c][:,np.newaxis], cmap=cmap)
    
    # get one new axis
    for ax in axs:
        divider = make_axes_locatable(ax)
        divider.new_vertical(size='5%', pad=0.6, pack_start = True)

    cax = fig.add_axes([0.05, 0, 0.95, 0.05])
    # pcm = plt.pcolormesh(data[c][:,np.newaxis], norm=norm, cmap=cmap)
    cb = fig.colorbar(s, ax=axs, cax=cax, orientation='horizontal')
    if cb_ticks is not None:
        cb.ax.set_xticks(cb_ticks)
        cb.ax.set_xticklabels(cb_ticks, fontsize=18)
    else:
        cb.ax.tick_params(labelsize=18)
    if cb_label is not None:
        cb.ax.set_xlabel(cb_label, fontsize=18)
    else: cb.ax.set_xlabel(c, fontsize=18)
    
    # set tight layout in case there are different subplots
    plt.tight_layout()

    if savefig:
        plt.savefig(fname=os.path.join(FIG_DIR, f'{title}.png'),
                    transparent=False, bbox_inches='tight', dpi=dpi)
    if show: plt.show(block=block)

def barplot(data, channels, num, subplot=None, figsize=(19.2, 9.43), name=None, savefig=False):

    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    x_pos = np.arange(np.size(data,axis=1))

    plt.figure(num=num, figsize=figsize)
    if subplot is None:
        subplot = (1, 1, 1)
    plt.subplot(*subplot)
    
    ax = plt.gca()
    ax.bar(x_pos, mean, yerr=sd, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Weight')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(channels)
    # ax.set_title('')
    ax.yaxis.grid(False)

    # Save the figure and show
    if savefig:
        plt.tight_layout()
        plt.savefig(fname=os.path.join(FIG_DIR, f'{name}.png'))

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
                    transparent=True, bbox_inches='tight', dpi=dpi)
    if show: plt.show(block=block)
    # plt.close()