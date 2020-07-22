# -----------------------------------------------------------------------------#
#                           behavioral_models.py                               #
#------------------------------------------------------------------------------#

import numpy as np
import json

def load_behavioral_data(path: str, verbose=True) -> tuple:
    """Load aggregated behavioral files. 
    
    Function assumes that name of the file 
    is behavioral_data_clean_all with .npy extension for data file and .json 
    extension for metadata file. It also assumes that both files are within 
    the same directory specified as root. 
    
    Args:
        path: 
            Path to directory containing behavioral files
        verbose: 
            Do you want additional information regarding loaded files?
        
    Returns:
        A tuple of (ndarray, dict). First element contains ndarray with 
        behavioral data, second element is a dict decribing all fields of the 
        array.
    """
    import os
    
    beh_path = os.path.join(path, "behavioral_data_clean_all.npy")
    beh_meta_path = beh_path.replace("npy", "json")
    
    beh = np.load(beh_path)
    with open(beh_meta_path, "r") as f:
        meta = json.loads(f.read())
        
    if verbose:
        print("Shape of beh array:", beh.shape)
        print("Conditions", [(i, cond) for i, cond in enumerate(meta['dim2'])])
        print("Columns:", [(i, col) for i, col in enumerate(meta['dim4'])])        
        
    return beh, meta

def get_response_mask(beh, meta, subject, condition):
    """Return masking array indicating subject response.
    
    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
    
    Returns:
        np.array (n_trials x n_sides): 
            mask array for subject response 
            1: indicate that option was selected
            0: option was not selected
    """
    resp = (beh[subject, condition, :, meta['dim4'].index('response')] + 1) / 2
    mask_resp = np.hstack((1-resp[:, np.newaxis], resp[:, np.newaxis]))
    mask_resp[mask_resp == .5] = 0
    
    return mask_resp

def estimate_wbci(beh, meta, subject, condition, alpha):
    '''Implements TD learning model on side probabilities. Note that estimated 
    probability is the probability that side will be chosen (rewarded in 
    reward-seeking condition or punished in punishment-avoidance condition)

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        alpha (float): learning rate

    Returns:
        np.array (n_trials x n_sides): 
            reflects algorithm trialwise beliefs about probabilities that side 
            will be chosen (rewarded / punished)
    '''

    wbci = np.zeros((beh.shape[2], 2))
    wbci[0] = [.5, .5] # Initial beliefs (agnostic)

    side_bci = np.copy(
        beh[subject, condition, :, meta['dim4'].index('side_bci')][:-1]
    )

    for t, rbci in enumerate(side_bci):
        wbci[t+1, 1] = wbci[t, 1] + alpha * ((rbci + 1)/2 - wbci[t, 1])
        wbci[t+1, 0] = wbci[t, 0] + alpha * ((-rbci + 1)/2 - wbci[t, 0])

    return wbci

def estimate_wbci_pd(beh, meta, subject, condition, alpha_plus, alpha_minus):
    '''Implements TD learning model with separate learning rates for positive 
    and negative prediction errors.

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        alpha_plus (float): learning rate for positive PE
        alpha_minus (float): learning rate for negative PE

    Returns:
        (np.array): reflects algorithm trialwise beliefs about
            probabilities that box will be chosen (rewarded / punished)
    '''
    
    wbci = np.zeros((beh.shape[2], 2))
    wbci[0] = [.5, .5] # Initial beliefs (agnostic)

    response = np.copy(beh[subject, condition, :, meta['dim4'].index('response')][:-1])
    side_bci = np.copy(beh[subject, condition, :, meta['dim4'].index('side_bci')][:-1])
    side = np.copy(beh[subject, condition, :, meta['dim4'].index('side')][:-1])
    
    # establish trialwise learning rates
    alpha = alpha_plus * (side == response) \
          + alpha_minus * (side != response) 

    for trial, sbci in enumerate(side_bci):
        wbci[trial+1, 1] = wbci[trial, 1] \
                        + alpha[trial] * ((sbci + 1)/2 - wbci[trial, 1])
        wbci[trial+1, 0] = wbci[trial, 0] \
                        + alpha[trial] * ((-sbci + 1)/2 - wbci[trial, 0])
    
    return wbci

def estimate_util(beh, meta, subject, condition, gamma=1, delta=1):
    '''Implements function converting reward magnitude to experienced utility.

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        gamma (float): loss aversion parameter
        delta: (float): risk aversion parameter

    Returns:
        np.array (n_trials x n_sides): 
            reflects algorithm trialwise estimates of utility for both left and 
            right boxes
    '''
    util = np.zeros((beh.shape[2], 2))

    if condition == meta['dim2'].index('pun'):
        factor = (-1) * gamma
    else:
        factor = 1

    util[:, 0] = factor * np.power(
        np.abs(beh[subject, condition, :, meta['dim4'].index('magn_left')]),
        delta
    )
    util[:, 1] = factor * np.power(
        np.abs(beh[subject, condition, :, meta['dim4'].index('magn_right')]),
        delta
    )

    return util

def estimate_choice_probability(wbci, util, kind='simple', beta=None):
    '''Implements softmax decision rule reflecting choice probabilities

    Args:
        wbci (np.array): trialwise beliefs about probabilities that side will
            be rewarded / punished
        util (np.array): trialwise estimates of utility for both sides
        kind (str): either 'simple' or 'softmax'
        beta (float): inverse temperature for softmax function

    Returns:
        np.array (n_trials x n_sides): 
            trialwise choice probabilities
    '''

    # Calculate expected value for both options
    exvl = np.multiply(util, wbci)

    if kind == 'simple':
        prob = exvl / np.sum(exvl, axis=1)[:, np.newaxis]
        if np.sum(exvl) < 0:
            prob = np.fliplr(prob)

    elif kind == 'softmax':
        prob = np.exp(beta * exvl) / np.sum(np.exp(beta * exvl), axis=1)[:, np.newaxis]

    return prob

def g_square(beh, meta, subject, condition, prob):
    '''Calculate badness-of-fit quality measure. G-square is inversely
    related to log likelyhood.

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        prob (np.array): trialwise choice probabilities

    Returns:
        (float): g-square badness-of-fit
    '''

    ll = 0
    responses = beh[subject, condition, :, meta['dim4'].index('response')]

    for trial, response in enumerate(responses):
        if response == -1:
            if prob[trial, 0] > 0: 
                ll += np.log(prob[trial, 0])
            else:
                ll += np.log(np.finfo(float).eps)
        elif response == 1:
            if prob[trial, 1] > 0:
                ll += np.log(prob[trial, 1])
            else:
                ll += np.log(np.finfo(float).eps)

    return (-2) * ll


### Full Models ################################################################
def model1(beh, meta, subject, condition, alpha):
    '''Simple one-parameter model with variable learning rate.'''

    wbci = estimate_wbci(beh, meta, subject, condition, alpha)
    util = estimate_util(beh, meta, subject, condition)
    prob = estimate_choice_probability(wbci, util, kind='simple')

    return (wbci, util, prob)


def model2(beh, meta, subject, condition, alpha, beta):
    '''Two-parameter model  with variable learning rate and inverse T.'''

    wbci = estimate_wbci(beh, meta, subject, condition, alpha)
    util = estimate_util(beh, meta, subject, condition)
    prob = estimate_choice_probability(wbci, util, kind='softmax', beta=beta)

    return (wbci, util, prob)

def model3(beh, meta, subject, condition, alpha, beta, gamma, delta):
    '''Four-parameter model.

    Args:
        alpha (float): learning rate
        beta (float): inverse softmax temperature
        gamma (float): loss aversion
        delta (float): risk aversion
    '''

    wbci = estimate_wbci(beh, meta, subject, condition, alpha)
    util = estimate_util(beh, meta, subject, condition, gamma, delta)
    prob = estimate_choice_probability(wbci, util, kind='softmax', beta=beta)

    return (wbci, util, prob)

def estimate_modulation(beh, meta, subject, condition, wbci):
    '''Calculate trial-wise hidden model variables for chosen option. 
    These values are used to create fMRI model-based regressors.

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        wbci (np.array): reflects algorithm trialwise beliefs about
            probabilities that box will be chosen (rewarded / punished)

    Returns (3-tuple):
        wcor (np.arry): expected probability of choosing correct box
        exvl (np.arry): pascalian expected value for chosen option
        perr (np.array): prediction error during outcome
    '''
    magn = np.hstack((
        np.copy(beh[subject, condition, :, meta['dim4'].index('magn_left')])[:, np.newaxis],
        np.copy(beh[subject, condition, :, meta['dim4'].index('magn_right')])[:, np.newaxis],
    ))
    response = np.copy(beh[subject, condition, :, meta['dim4'].index('response')])
    side_bci = np.copy(beh[subject, condition, :, meta['dim4'].index('side_bci')])
    side = np.copy(beh[subject, condition, :, meta['dim4'].index('side')])

    response_mask = np.hstack((
        response.reshape(-1, 1) == -1, 
        response.reshape(-1, 1) == 1
    ))

    # Expected probability for side for being correct
    if condition == 1:
        wcor = np.sum(np.fliplr(wbci) * response_mask, axis=1)    
    else:
        wcor = np.sum(wbci * response_mask, axis=1)

    # Expected value for chosen option
    exvl = np.sum(magn * response_mask, axis=1) * wcor

    # Prediction error
    perr = (side == response) - wcor

    return wcor, exvl, perr



