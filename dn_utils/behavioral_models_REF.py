import numpy as np
import json

def load_behavioral_data(root: str, verbose=True) -> tuple:
    """Load aggregated behavioral files. 
    
    Function assumes that name of the file 
    is behavioral_data_clean_all with .npy extension for data file and .json 
    extension for metadata file. It also assumes that both files are within 
    the same directory specified as root. 
    
    Args:
        root: 
            Path to directory containing behavioral files
        verbose: 
            Do you want additional information regarding loaded files?
        
    Returns:
        A tuple of (ndarray, dict). First element contains ndarray with 
        behavioral data, second element is a dict decribing all fields of the 
        array.
    """
    import os
    
    beh_path = os.path.join(root, "behavioral_data_clean_all_REF.npy")
    beh_meta_path = beh_path.replace("npy", "json")
    
    beh = np.load(beh_path)
    with open(beh_meta_path, "r") as f:
        meta = json.loads(f.read())
        
    if verbose:
        print("Shape of beh array:", beh.shape)
        print("Conditions", [(i, cond) for i, cond in enumerate(meta['dim2'])])
        print("Columns:", [(i, col) for i, col in enumerate(meta['dim4'])])        
        
    return beh, meta

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

### <--- REFACTORING POINT ####################################################

def estimate_values_pd(beh, meta, subject, condition, alpha_plus, alpha_minus):
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
        val (np.array): reflects algorithm trialwise beliefs about
            probabilities that box will be chosen (rewarded / punished)
    '''
    
    val = np.zeros((beh.shape[2], 2))
    val[0] = [.5, .5] # Initial beliefs (agnostic)

    rewarded = np.copy(beh[subject, condition, :, meta['dim4'].index('rwd')][:-1])
    response = np.copy(beh[subject, condition, :, meta['dim4'].index('response')][:-1])

    # establish trialwise learning rates
    if condition == 0:
        alpha = alpha_plus * (rewarded == response) \
              + alpha_minus * (rewarded != response) 
    else:
        alpha = alpha_plus * ((-1)*rewarded == response) \
              + alpha_minus * ((-1)*rewarded != response) 

    for trial, rwd in enumerate(rewarded):
        val[trial+1, 1] = val[trial, 1] \
                        + alpha[trial] * ((rwd + 1)/2 - val[trial, 1])
        val[trial+1, 0] = val[trial, 0] \
                        + alpha[trial] * ((-rwd + 1)/2 - val[trial, 0])
    
    return val

def estimate_regressors(beh, meta, subject, condition, val):
    '''Calculate trial-wise regressors for chosen option.

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        val (np.array): reflects algorithm trialwise beliefs about
            probabilities that box will be chosen (rewarded / punished)

    Returns (3-tuple):
        anticip_pwin (np.arry): expected probability of choosing correct box
        anticip_rew (np.arry): pascalian expected value
        pred_err (np.array): prediction error
    '''
    rewarded = np.copy(beh[subject, condition, :, meta['dim4'].index('rwd')])
    response = np.copy(beh[subject, condition, :, meta['dim4'].index('response')])
    magn_both = np.hstack((
        np.copy(beh[subject, condition, :, meta['dim4'].index('magn_left')])[:, np.newaxis],
        np.copy(beh[subject, condition, :, meta['dim4'].index('magn_right')])[:, np.newaxis],
    ))

    response_mask = np.hstack((
        response.reshape(-1, 1) == -1, 
        response.reshape(-1, 1) == 1
    ))

    # Ensure correct val interpretation
    anticip_val = np.sum(np.multiply(val, response_mask), axis=1) # bci
    if condition == 1:
        val = np.fliplr(val) # change bci interpretation to correct interpretation
        rewarded *= (-1)
    anticip_pwin = np.sum(np.multiply(val, response_mask), axis=1) # correct interpretation
    
    anticip_rew = np.sum(magn_both * response_mask, 1) * anticip_val
    pred_err = (rewarded == response) - anticip_pwin

    return anticip_pwin, anticip_rew, pred_err



