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
    
    beh_path = os.path.join(root, "behavioral_data_clean_all.npy")
    beh_meta_path = beh_path.replace("npy", "json")
    
    beh = np.load(beh_path)
    with open(beh_meta_path, "r") as f:
        meta = json.loads(f.read())
        
    if verbose:
        print("Shape of beh array:", beh.shape)
        print("Conditions", [(i, cond) for i, cond in enumerate(meta['dim2'])])
        print("Columns:", [(i, col) for i, col in enumerate(meta['dim4'])])        
        
    return beh, meta

def estimate_values(beh, meta, subject, condition, alpha):
    '''Implements TD learning model on experienced probabilistic outcomes.

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        alpha (float): learning rate

    Returns:
        val (np.array): reflects algorithm trialwise beliefs about
            probabilities that box will be rewarded / punished
    '''

    val = np.zeros((beh.shape[2], 2))
    val[0] = [.5, .5] # Initial beliefs (agnostic)

    rewarded = np.copy(beh[subject, condition, :, meta['dim4'].index('rwd')][:-1])

    for trial, rwd in enumerate(rewarded):
        val[trial+1, 1] = val[trial, 1] + alpha * ((rwd + 1)/2 - val[trial, 1])
        val[trial+1, 0] = val[trial, 0] + alpha * ((-rwd + 1)/2 - val[trial, 0])

    return val

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

def estimate_pred_err(beh, meta, subject, condition, val):
    '''Calculate trial-wise prediction error for correct choice probability.
    
    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        val (np.array): reflects algorithm trialwise beliefs about
            probabilities that box will be chosen (rewarded / punished)
            
    Returns:
        pred_err (np.array): trial-wise prediction error values
    '''

    rewarded = np.copy(beh[subject, condition, :, meta['dim4'].index('rwd')])
    response = np.copy(beh[subject, condition, :, meta['dim4'].index('response')])

    if condition == 1:
        val = np.fliplr(val) # change bci interpretation to correct interpretation
        rewarded *= (-1)

    response_mask = np.hstack((
        response.reshape(-1, 1) == -1, 
        response.reshape(-1, 1) == 1
    ))

    anticip_val = np.sum(np.multiply(val, response_mask), axis=1)
    
    return (rewarded == response) - anticip_val 

def estimate_utilities(beh, meta, subject, condition, gamma=1, delta=1):
    '''Implements function converting reward magnitude to experienced utility.

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        gamma (float): loss aversion parameter
        delta: (float): risk aversion parameter

    Returns:
        util (np.array): reflects algorithm trialwise estimates of utility
            for both left and right boxes
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

def estimate_choice_probability(val, util, kind='simple', theta=None):
    '''Implements softmax decision rule reflecting choice probabilities

    Args:
        val (np.array): trialwise beliefs about probabilities that box will
            be rewarded / punished
        util (np.array): trialwise estimates of utility for both boxes
        kind (str): either 'simple' or 'softmax' for two different models
        theta (float): inverse temperature for softmax function

    Returns:
        p (np.array): trialwise choice probabilities
    '''

    # Calculate expected value for both options
    ev = np.multiply(util, val)

    if kind == 'simple':
        p = ev / np.sum(ev, axis=1)[:, np.newaxis]
        if np.sum(ev) < 0:
            p = np.fliplr(p)

    elif kind == 'softmax':
        p = np.exp(theta * ev) / np.sum(np.exp(theta * ev), axis=1)[:, np.newaxis]

    return p

def g_square(beh, meta, subject, condition, p):
    '''Calculate badness-of-fit quality measure. G-square is inversely
    related to log likelyhood.

    Args:
        beh (np.array): aggregated behavioral responses
        meta (dict): description of beh array coding
        subject (int): subject index
        condition (int): task condition index
        p (np.array): trialwise choice probabilities

    Returns:
        (float): g-square badness-of-fit
    '''

    ll = 0
    responses = beh[subject, condition, :, meta['dim4'].index('response')]

    for trial, response in enumerate(responses):
        if response == -1:
            ll += np.log(p[trial, 0])
        elif response == 1:
            ll += np.log(p[trial, 1])

    return (-2) * ll

### Behavioral Models #######################################################
def model1(beh, meta, subject, condition, alpha):
    '''Simple one-parameter model with variable learning rate.'''

    val = estimate_values(beh, meta, subject, condition, alpha)
    util = estimate_utilities(beh, meta, subject, condition)
    p = estimate_choice_probability(val, util, kind='simple')

    return (val, util, p)


def model2(beh, meta, subject, condition, alpha, theta):
    '''Two-parameter model  with variable learning rate and inverse T.'''

    val = estimate_values(beh, meta, subject, condition, alpha)
    util = estimate_utilities(beh, meta, subject, condition)
    p = estimate_choice_probability(val, util, kind='softmax', theta=theta)

    return (val, util, p)

def model3(beh, meta, subject, condition, alpha, theta, gamma, delta):
    '''Four-parameter model.

    Args:
        alpha (float): learning rate
        theta (float): inverse softmax temperature
        gamma (float): loss aversion
        delta (float): risk aversion
    '''

    val = estimate_values(beh, meta, subject, condition, alpha)
    util = estimate_utilities(beh, meta, subject, condition, gamma, delta)
    p = estimate_choice_probability(val, util, kind='softmax', theta=theta)

    return (val, util, p)
