import matplotlib.pyplot as plt
import numpy as np
import itertools
from nilearn import plotting
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
        
def plot_stat_maps_grid(stat_maps, labels=None, threshold=None):
    '''Plots grid of statical maps.
    
    Args:
        stat_maps (list): List of nibabel.nifti1.Nifti1Image first level output statistical images.
        labels (list, optional): List of titles for grid images. Defaults to integers (1, 2, 3...)
        threshold (float, optional): Threshold for plotting. If nothing is passed, image will not be
            thresholded.    
    '''
    n = len(stat_maps)
    n_rows = (n - 1) // 4 + 1
    
    if labels is None:
        labels = [str(i) for i in range(n)]
    
    fig, ax = plt.subplots(
        nrows=n_rows, ncols=4, 
        facecolor='k', figsize=(15, 4 * n_rows))

    for cidx, stat_map in enumerate(stat_maps):
        if n_rows == 1:
            axes = ax[cidx]
        else:
            axes = ax[int(cidx / 4)][int(cidx % 4)]
        plotting.plot_glass_brain(
            stat_map, 
            colorbar=False, 
            threshold=threshold,
            title=labels[cidx],
            axes=axes,
            plot_abs=False,
            black_bg=True,
            display_mode='z')