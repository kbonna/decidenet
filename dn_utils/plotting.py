import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools

from nilearn import plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .glm_utils import Regressor
from matplotlib.colors import to_rgb, is_color_like


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
    
    
def barplot_annotate_brackets(ax, num1, num2, text, center, height, 
                              yerr=None, dh=.05, barh=.05, fs=None, 
                              maxasterix=None):
    ''' Annotate barplot with p-values.

    Args:
        ax: 
            Axes containing barplot.
        num1 (int): 
            Index of left bar to put bracket over.
        num2 (int): 
            index of right bar to put bracket over.
        text (str): 
            String used for annotation. 
        center: 
            Centers of all bars (like plt.bar() input)
        height: 
            Heights of all bars (like plt.bar() input)
        yerr (optional): 
            Yerrs of all bars (like plt.bar() input)
        dh (float, optional): 
            Height offset over bar / bar + yerr in axes coordinates (0 to 1)
        barh (float, optional): 
            Bar height in axes coordinates (0 to 1)
        font size (float, optional)
            Font size for annotated text.
    '''
    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr is not None:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    # Add lines
    ax.plot(barx, bary, c='black')

    # Add text
    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    ax.text(*mid, text, **kwargs)
    
def aligned_imshow_cbar(ax, im):
    '''Create nicely aligned colorbar for matrix visualisations.
    
    Args:
        ax:
            Axes object.
        im:
            Axes image object.
    '''
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return cax
    
def hex_to_rgb(h):
    '''Converts hex color string into RGB tuple with values from 0 to 1.'''
    if h.startswith('#'):
        h = h[1:]
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

def barplot_annotate(ax, text, bar_x, bar_y, offset_bar, offset_top, 
                     line_kwargs=None, annotate_kwargs=None):
    '''Annotate barplot with text and lines between two individual bars.
    
    Args:
        ax (Axes):
            Axes with barplot.
        text (str):
            Annotation text.
        bar_x (list):
            X-coordinates of bar centers. Should be of length 2.
        bar_y (list):
            Height of bars. Should be of length 2.
        offset_bar (float):
            Line offset from the top of bars.
        offset_top (float):
            Text offset from the top of the higher bar (excluding offset_bar).
        line_kwargs (dict, optional):
            Optional arguments for plt.plot used to create lines.
        annotate_kwargs (dict, optional):
            Optional arguments for plt.text used to create text annotation.
    '''
    line_kwargs = {'color': 'k'} if line_kwargs is None else line_kwargs
    annotate_kwargs = {'backgroundcolor': '#ffffff'} \
                      if annotate_kwargs is None else annotate_kwargs
    x_annotate = np.mean(bar_x)
    y_annotate = np.max(bar_y) + offset_bar + offset_top

    # Create lines
    ax.plot(bar_x, [y_annotate, y_annotate], **line_kwargs)
    
    for idx in range(2):
        ax.plot([bar_x[idx], bar_x[idx]], 
                [bar_y[idx] + offset_bar, y_annotate], 
                **line_kwargs)
    
    ax.annotate(
        s=text,
        xy=[x_annotate, y_annotate],
        ha='center',
        va='center',
        **annotate_kwargs
    )
    
def plot_design_matrix(X, colors=None, output_file=None):
    '''Visualise GLM design matrix. 
    
    Args:
        X (pd.Dataframe):
            Design matrix. Each column correspond to single GLM regressor. 
            Column names will be used to label subplots. Index should be scan
            time in seconds.
        colors (list of string, optional):
            Colors for each regressor. Use it to distinguish different regressor 
            types.
        output_file (string, optional):
            The name of an image file to export the plot to.    
    '''
    mpl.rcParams.update({'font.size': 8})
    colors = colors if colors is not None else ['b'] * X.shape[1]
    
    fig, axs = plt.subplots(
        nrows=X.shape[1], 
        sharex=True, 
        figsize=(13, 7), 
        facecolor='w'
    )

    for i, column in enumerate(X):
        axs[i].plot(X.index, X[column], color=colors[i], label=column)
        axs[i].legend(loc='upper right', framealpha=1)
        axs[i].set_xlim([0, max(X.index)])

    axs[0].set_title('PPI linear model regressors')
    axs[-1].set_xlabel('Time [seconds]')
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file)
        plt.close(fig)
    else:
        plt.show()
        

def plot_regressors_correlation(X, colors=None, output_file=None):
    '''Visualise correlation between individual regressors in GLM design matrix. 
    
    Args:
        X (pd.Dataframe):
            Design matrix. Each column correspond to single GLM regressor. 
            Column names will be used to label subplots. Index should be scan
            time in seconds.
        colors (list of string, optional):
            Colors for each regressor. Use it to distinguish different regressor 
            types.
        output_file (string, optional):
            The name of an image file to export the plot to.    
    '''
    mpl.rcParams.update({'font.size': 13})
    colors = colors if colors is not None else ['b'] * X.shape[1]
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='w')
    im = ax.imshow(X.corr(), cmap='RdBu_r', clim=[-1, 1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.set_xticks(np.arange(X.shape[1]))
    ax.set_yticks(np.arange(X.shape[1]))
    ax.set_xticklabels(X.columns, rotation=90)
    ax.set_yticklabels(X.columns)
    ax.set_title('Correlation between regressors')

    for xticklabel, yticklabel, tickcolor in zip(ax.get_xticklabels(), 
                                                 ax.get_yticklabels(), 
                                                 colors):
        xticklabel.set_color(tickcolor)
        yticklabel.set_color(tickcolor)

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file)
        plt.close(fig)
    else:
        plt.show()
        
def plot_matrix_old(mat, clim=[-1, 1], labels=None, annotate=False, title=None):
    '''Basic plotting utility for adjacency matices.
    
    Args:
        mat (np.array):
            Adjacency matrix.
        clim (list of floats, optional):
            Two-element list specifying color limits. Defaults to range from -1
            to 1 suitable for visualisation of adjacency matrices.
        labeles (list-like, optional):
            Labels for individual network nodes.
        annotate (bool, optional):
            Annotation of individual connections. Defaults to False.
        title (str, optional):
            Plot tilte.
            
    Return:
        None.
    '''
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='w')
    im = ax.imshow(mat, clim=clim, cmap='RdBu_r', interpolation='none')
    aligned_imshow_cbar(ax, im)
    
    if labels is not None:
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        
    if annotate:
        clim_range = clim[1] - clim[0]
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                if value < (clim[0] + clim_range * 0.2) or \
                   value > (clim[1] - clim_range * 0.2):
                    color = 'w'
                else:
                    color = 'k'
                text = ax.text(
                    j, 
                    i, 
                    f'{value:.2f}',
                    ha='center', 
                    va='center', 
                    color=color
                )
                
    if title:
        ax.set_title(title)

    plt.tight_layout()
    

def plot_matrix(mat, clim=[-1, 1], labels=None, annotate=False, 
                annotate_mask=None, title=None):
    """Basic matrix plotting utility suitable for small and large matrices.
    
    Args:
        mat (np.ndarray):
            2D array (matrix).
        clim (list; optional):
            Colorbar limits.
        labels (list-like; optional):
            Labels for each entry of the matrix (usually corresponding to node 
            or subnetwork). Label entries can be either regular strings or 
            strings representing colors (e.g. '#00ff00', 'red'). In the case of
            colors, axes will be annotated with colorbar with colors 
            corresponding to each entry. Type of annotation will be detected 
            automatically.
        annotate (bool or np.ndarray; optional):
            Text annotations for each matrix element. If True, matrix will be 
            annotaed with numerical representation of value for corresponding 
            matrix entry. If array is used, values from this array will be used
            for annotations instead values from original mat array.
        annotate_mask (np.ndarray; optional):
            Array of boolean values with shape equal to mat shape. True values
            indicate orignal array elements that will be annotated with values.
        title (str; optional):
            Plot title.
            
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='w')
    im = ax.imshow(mat, clim=clim, cmap='RdBu_r', interpolation='none')
    divider = make_axes_locatable(ax)

    # Create colorbar
    cbarax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cbarax)

    if labels is not None:
        if all(is_color_like(c) for c in labels):
            # Case 1: Labels are colors
            colors = [to_rgb(c) for c in labels]
            colors_h = np.array(colors)[np.newaxis, ...]
            colors_v = np.transpose(colors_h, axes=(1, 0, 2))
            ax.set_xticks([])
            ax.set_yticks([])

            # Create additional axes
            cax_h = divider.append_axes("bottom", size="2%", pad=0.07)
            cax_v = divider.append_axes("left", size="2%", pad=0.07)

            for cax in [cax_h, cax_v]:
                for spine in cax.spines.values():
                    spine.set_visible(False)
                cax.set_xticks([])
                cax.set_yticks([])

            # Plot colors
            cax_v.imshow(colors_v, aspect="auto")
            cax_h.imshow(colors_h, aspect="auto")
        else:
            # Case 2: Labels are text
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)

    if annotate is not False:
        # Annotation values & mask
        mask = annotate_mask
        mask = np.ones(mat.shape, dtype=bool) if mask is None else mask
        values = annotate if isinstance(annotate, np.ndarray) else mat

        clim_hi = clim[1] - (clim[1] - clim[0]) * 0.2
        clim_lo = clim[0] + (clim[1] - clim[0]) * 0.2
        for i, j in np.argwhere(mask):
            value = values[i, j]
            if value < clim_lo or value > clim_hi:
                c = 'w'
            else:
                c = 'k'
            text = ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=c)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
