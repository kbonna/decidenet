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


def plot_correlation_between_regressors(regressors, figsize=(7, 7), 
                                        output_file=None) -> None:
    '''Plot correlation matrix for specific set of regressors.
    
    Args:
        regressors (list): 
            List containing Regressor objects.
        figsize (tuple, optional): 
            Figure size.
        output_file (string, optional):
            The name of an image file to export the plot to.
    '''

    # Calculate correlation between expected probability of winning and expected value
    corr_between_regressors = np.zeros((len(regressors), ) * 2)

    for i, j in itertools.product(range(len(regressors)), repeat=2):
        if j > i:
            corr_between_regressors[i, j] = regressors[i].corr(regressors[j])

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
        
    if output_file:
        fig.savefig(output_file)
    
    plt.close(fig)
        
        
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
        
        
def plot_z_convergence(rhat_z: np.array) -> None:
    '''Plots r-hat barplot for indicator variable'''
    
    rhat_thr = 1.1
    n_subjects = np.size(rhat_z)

    fig, ax = plt.subplots(ncols=1, nrows=1, facecolor='w', figsize=(20, 4))
    b = ax.bar(
        range(n_subjects), 
        rhat_z,
        color=[.3, .3, .3]
    )

    for rect in b:
        if rect.get_height() > rhat_thr:
            rect.set_color([1, .3, .3])
    
    ax.set_title('Model indicator variable $z_i$')
    
    ax.set_xlim((-1, n_subjects))
    ax.set_xlabel('Subjects')
    ax.set_xticks(range(n_subjects))
    ax.set_xticklabels([f'm{sub:02}' for sub in range(2, n_subjects+2)],
                      rotation=-45);

    ax.set_ylim((1, 1.4))
    ax.set_ylabel(r'$\hat{r}$', rotation=0)

    ax.plot(ax.get_xlim(), (rhat_thr, rhat_thr), color='r')
    
    plt.tight_layout()
    plt.savefig('pygures/convergence.png')
    
    
def gen_logbf_barplot(bf, model_names, output_file=None, cmap_name='bone'):
    '''Create barplots for subjectwise model comparison using log-scale.

    Args:
        bf: 
            Bayes factor vector of size n_subjects.
        modelname: 
            Two-element list of modelnames.
        output_file (optional):
            The name of an image file to export the plot to.
        cmap_name (optional): 
            Name of matplotlib colormap used to discriminate evidence levels. 
    '''
    logbf = np.log10(bf)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', figsize=(20, 5))
    
    b = ax.bar(range(n_subjects), logbf)

    cmap = cm.get_cmap(cmap_name)
    color = {
        'extreme': cmap(0),
        'vstrong': cmap(1/5),
        'strong': cmap(2/5),  
        'moderate': cmap(3/5),
        'anecdotal': cmap(4/5),
    }
    
    for rect in b:
        if np.abs(rect.get_height()) < np.log10(3):
            rect.set_color(color['anecdotal'])
        elif np.abs(rect.get_height()) < np.log10(10):
            rect.set_color(color['moderate'])
        elif np.abs(rect.get_height()) < np.log10(30):
            rect.set_color(color['strong'])
        elif np.abs(rect.get_height()) < np.log10(100):
            rect.set_color(color['vstrong'])
        else:
            rect.set_color(color['extreme'])
        
    ax.set_title('Individual Bayes Factors')
    ax.set_xlim((-1, n_subjects))
    ax.set_xlabel('Subjects')
    ax.set_xticks(range(n_subjects))
    ax.set_xticklabels([f'm{sub:02}' for sub in range(2, n_subjects+2)],
                      rotation=-45)
    ax.set_ylim((-3, 3))
    ax.set_ylabel(r'$\log_{10}(BF)$', rotation=90)
    ax.set_axisbelow(True)
    ax.grid()

    legend_elements = [Patch(facecolor=col, edgecolor='k', label=key) 
                   for key, col in color.items()]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, .7))
    ax.annotate(r'$\uparrow$' + model_names[0], 
                [1.01, .8], xycoords='axes fraction', fontsize=20)
    ax.annotate(r'$\downarrow$' + model_names[1], 
                [1.01, .1], xycoords='axes fraction', fontsize=20)
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file)
        
def plot_stat_map_custom(stat_map, bg_img, height_control='fdr', p_val=0.01, 
                  cluster_threshold=15, title='', **plot_kwargs):
    '''Threshold and plot statistical map.
    
    Args:
        stat_map (nibabel.nifti1.Nifti1Image): 
            Statistical map from 1st or 2nd level GLM analysis.
        height_control (str, optional): 
            Specifies type of multiple comparison correction.
        p_val (float, optional): 
            Specifies p-value threshold for statistal map.
        cluster_threshold (int, optional): 
            Threshold for minimal size cluster of voxels.  
        title (str, optional):
            Figure title.
    '''
    cut_coords = (-15, -10, 3, 22, 38, 52)
    display_mode = 'z'
    
    _, threshold = map_threshold(
        stat_map,
        alpha=p_val,
        height_control=height_control,
        cluster_threshold=cluster_threshold)

    fig, ax = plt.subplots(facecolor='k', figsize=(20, 5))
    plotting.plot_stat_map(
        stat_map,
        bg_img=bg_img,
        axes=ax,
        threshold=threshold,
        colorbar=True,
        display_mode=display_mode,
        cut_coords=cut_coords,
        title=title + f' ({height_control}; p<{p_val})',
        **plot_kwargs
    )
    plt.show()    