### Functions ##################################################################

from random import randint, shuffle

def create_blocks(N_trials: int, N_reversal: int, N_min_stable: int) -> list:
    """Creates basic block structure for the task.

    Args:
        N_trials: Total number of trials.
        N_reversal: Total number of reward contingency reversal. Number of
            different block is N_reversal + 1.
        N_min_stable: Minimal number of trials per block. Reversals will be
            separated by at least N_min_stable trials.

    Returns:
        List containing more probable box for each trial. If i-th element is -1,
        then on i-th trial left box have higher probability for being chosen.
    """
    if N_min_stable * (N_reversal+1) > N_trials:
        raise Exception(f'You do not have enough trials to have at least '
                        f'{N_min_stable} trials in {N_reversal+1} blocks!')

    block_size = [N_min_stable for block in range(N_reversal + 1)]
    while sum(block_size) < N_trials:
        block_size[randint(0, N_reversal)] += 1

    trial_block = [(block % 2 * 2) - 1
                       for block in range(N_reversal+1)
                       for trial in range(block_size[block])]

    random_factor = (randint(0, 1) * 2 - 1)
    return [random_factor * trial for trial in trial_block]

def create_schedule(N_trials, trial_block, reward_probability):
    """Determines which box will be rewarded / punished on each trial.

    Args:
        N_trials (int): Total number of trials.
        trial_block (list): List of trial types. Accepted types are -1 if left
            box is more probable and 1 if right box is more probable.
        reward_probability (float): Probability that more probable side will
            actually be rewarded / punished. Should be greater than 0.5.

    Returns:
        List containing chosen box for each trial. If i-th element is -1, then
        on i-th trial left box is chosen.
    """
    if len(trial_block) != N_trials:
        raise Exception('Length of trial_block list does not match N_trials!')
    if reward_probability < 0.5:
        raise ValueError('reward_probability should be greater or equal to 0.5')

    N_rewarded = round(N_trials * reward_probability)

    trial_win = ([1 for i in range(N_rewarded)] +
                 [-1 for i in range(N_trials - N_rewarded)])
    shuffle(trial_win)

    return [trial_block[trial] * trial_win[trial] for trial in range(N_trials)]

def create_split(N_trials: int, reward_total: int, reward_minimum: int) -> list:
    """Determines reward magnitude for each box on each trial.

    Args:
        N_trials: Total number of trials.
        reward_total: Sum of reward magnitudes for both boxes.
        reward_minimum: Minimal reward magnitude that can be assigned to single
            box.

    Returns:
        List containing reward magnitude for one side (either left or right) for
        each trial.
    """
    if reward_minimum > reward_total:
        raise ValueError('Minimum reward cannot be greater than total reward!')

    return [randint(reward_minimum, reward_total-reward_minimum)
            for trial in range(N_trials)]

def gen_trialList(*args):
    """Converts arbitrary number of lists into trialList datatype.

    trialList data type is used to create TrialHandler.

    Args:
        (2-tuple): First value should be a list of objects (any type) used in
        trial. Second argument should be a string denoting name of the list.

    Returns:
        list of OrderedDict: used as keyword argument trialList in TrialHandler
            creation.
    """
    if len(args)==0: raise Exception('You need to pass at least one argument.')
    for arg in args:
        if type(arg) is not tuple:
            raise TypeError(f'{arg} is not a tuple!')
        if len(arg) != 2:
            raise IndexError(f'{arg} should have length 2!')
        if type(arg[0]) is not list:
            raise TypeError(f'{type(arg[0])} should be a list!')
        if type(arg[1]) is not str:
            raise TypeError(f'{type(arg[1])} should be a string!')
    if len(set([len(arg[0]) for arg in args])) != 1:
        raise IndexError('All lists should be of same size.')

    trialList = []
    keys = [arg[1] for arg in args]
    for x in zip(*tuple([arg[0] for arg in args])):
        trialList.append(OrderedDict([y for y in zip(keys, x)]))

    return trialList

def outcome_magn(thisTrial, keys, key_left, key_right, condition):
    """Determines points received by subject in reward condtion or taken from
    subject in a punishment condition.

    Args:
        thisTrial (OrderedDict): Contains information about current trial.
        keys (list): List of pressed keys.
        key_left (str): Key for pressing left button.
        key_right (str): Key for pressing right button.
        condition (str): 'rew' for reward condition and 'pun' for punishment
            condition.

    Returns:
        Integer denoting number of points rewarded / punished during thisTrial.
    """
    try:
        rewarded_side = thisTrial['rwd']
        magn_right = thisTrial['magn_right']
        magn_left = thisTrial['magn_left']
    except:
        print('thisTrial does not have required key(s)')
    if rewarded_side not in [-1, 1]:
        raise ValueError('rewarded_side should be either -1 or 1.')

    if keys == None:
        if condition == 'rew': return 0 # No reward in case of miss.
        if condition == 'pun': # Forced punishment in case of miss.
            if rewarded_side == -1: return magn_left
            else:                   return magn_right

    if keys[0][0] == key_left:
        keypressed = -1
        selected_reward = magn_left
    elif keys[0][0] == key_right:
        keypressed = 1
        selected_reward = magn_right
    else: raise ValueError('Forbidden key pressed. Aborting...')

    return (keypressed == rewarded_side) * selected_reward
