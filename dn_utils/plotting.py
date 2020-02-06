import matplotlib.pyplot as plt
import numpy as np

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
