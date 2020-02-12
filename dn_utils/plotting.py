import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .glm_utils import Regressor

def plot_trial_modulation(beh, meta, sub, con, modulations) -> None:
    '''Plot model variable time-course for a single task.
    
    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        modulations (n_subjects x n_conditions x n_trials): 
            trialwise modulations values for all subjects and conditions
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', figsize=(20, 3))
    
    block = beh[sub, con, :, meta['dim4'].index('block')]
    n_trials = len(meta['dim3'])
    trials = np.arange(n_trials) + 1

    ax.plot(0)
    ax.plot(trials, modulations, color='grey')
    ax.plot(
        trials[modulations > 0], 
        modulations[modulations > 0],
        linewidth=0,
        marker='o',
        markersize=5,
        color='g',
    )
    ax.plot(
        trials[modulations < 0], 
        modulations[modulations < 0],
        linewidth=0,
        marker='o',
        markersize=5,
        color='r',
    )
    ax.plot(
        trials[modulations == 0],
        modulations[modulations == 0],
        linewidth=0,
        marker='o',
        markersize=5,
        color='orange',
    )

    ax.set_xlim([0, n_trials + 1])
    ax.set_xticks(np.concatenate((np.nonzero(np.diff(block))[0] + 2, [1, 110])))
    ax.grid()

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_correlation_between_regressors(regressors, figsize=(7, 7)) -> None:
    '''Plot correlation matrix for specific set of regressors.
    
    Args:
        regressors (list): list containing Regressor objects
        figsize (tuple, optional): figure size
    '''

    # Calculate correlation between expected probability of winning and expected value
    corr_between_regressors = np.zeros((len(regressors), ) * 2)

    for i, j in itertools.product(range(len(regressors)), repeat=2):
        if j > i:
            corr_between_regressors[i, j] = Regressor.corrcoef(regressors[i], regressors[j])

    corr_between_regressors += corr_between_regressors.T
    np.fill_diagonal(corr_between_regressors, 1)

    # Generate figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_between_regressors, cmap='seismic', vmin=-1, vmax=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    ax.set_xticks(np.arange(len(regressors)))
    ax.set_yticks(np.arange(len(regressors)))
    ax.set_xticklabels([regressor.name for regressor in regressors])
    ax.set_yticklabels([regressor.name for regressor in regressors])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i, j in itertools.product(range(len(regressors)), repeat=2):
        if abs(corr_between_regressors[i, j]) < .5:
            color = 'k'
        else:
            color = 'w'
        text = ax.text(j, i, format(corr_between_regressors[i, j], '.2f'), 
                       ha="center", va="center", color=color)