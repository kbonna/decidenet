"""This is PsychoPy implementation of probabilistic reversal learning (PRL) task
treining used before fMRI study in DecideNet project.

Author: Kamil Bonna
Version: 1.2.train
"""
from prl_DecideNet_training_config import *
from collections import OrderedDict
from psychopy import visual, core, event, data, gui
from random import randint, shuffle, uniform
import numpy as np
import os.path

### Functions ##################################################################
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

    trialList datatype is used to create TrialHandler.

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

def draw_objects(*args):
    """Drawing any number of objects."""
    for arg in args:
        arg.draw()

def draw_phase(phase, idx, win=None):
    """Draw objects for certain trial phase and update screen.

    Args:
        phase (str): Accepted values are:
            'dec': decision phase
            'out': outcome phase
            'isi': interval between decision and outcome phase
            'iti': interval between trials
        win (visual.Window): PsychoPy Window object.

    Note:
        For simplicity function uses bunch of PsychoPy objects from global
        scope.
    """
    # Objects specific to feedback phase.
    if idx <= N_trials_feedback:
        draw_objects(feedback_text, feedback_box)

    draw_objects(left_box, right_box, fix_circle,
                 down_bar, down_bar_fill,
                 down_bar_tick1, down_bar_tick2)
    if phase == 'dec' or phase == 'isi':
        draw_objects(left_text, right_text, fix_text, up_text)
    elif phase == 'out':
        draw_objects(left_text, right_text, fix_text)

    win.flip()

def set_phase(phase):
    """Setting proper object attributes before drawing them. This function
    changes specific properties of visual objects depending on trial phase.

    Args:
        phase (str): Accepted values are:
            'dec_before': decision phase before button press
            'dec_after': decision phase after button press
            'out': outcome phase
            'isi': interval between decision and outcome phase
            'iti': interval between trials

    Note:
        For simplicity function uses bunch of PsychoPy objects from global
        scope.
    """
    if phase not in ['iti', 'dec_before', 'dec_after', 'isi', 'out']:
        raise ValueError('Such phase does not exist.')

    # ITI phase
    if phase == 'iti':
        fix_circle.setLineWidth(0)
        fix_circle.setFillColor(circle_color)

    # Decision phase
    elif phase == 'dec_before':
        left_text.setText(thisTrial['magn_left'])
        right_text.setText(thisTrial['magn_right'])
        fix_text.setText('?')
    elif phase =='dec_after':
        if keys is not None:
            if keys[0][0] == key_left:
                fix_circle.setLineWidth(circle_line_width)
                fix_circle.setLineColor(box_left_color)
            elif keys[0][0] == key_right:
                fix_circle.setLineWidth(circle_line_width)
                fix_circle.setLineColor(box_right_color)
        else:
            up_text.setText('Za wolno!')
        fix_text.setText('')

    # ISI phase
    elif phase == 'isi': pass

    # Outcome phase
    elif phase == 'out':
        # Setting text within the fixation circle
        fix_text.setText(str(won_magn))
        # Setting fixation circle color
        if thisTrial['rwd'] == -1:
            fix_circle.setFillColor(box_left_color)
        elif thisTrial['rwd'] == 1:
            fix_circle.setFillColor(box_right_color)
        up_text.setText('')
        # Ensure that down_bar_fill will always fit within down_bar
        account_perceived = account
        if account > account_max:
            account_perceived = account_max
        elif account < 0:
            account_perceived = 0
        down_bar_fill.setWidth(bar_width*account_perceived/account_max)
        down_bar_fill.setPos([(bar_width/2)*(account_perceived/account_max-1),
                               bar_y])

def print_trial(thisTrial, keys, condition, account, TrialHandler):
    """Prints useful information about current trial to the python console."""
    if condition == 'rew': mark = '$'
    elif condition == 'pun': mark = '#'
    print(f'\n\n===== Trial {TrialHandler.thisIndex + 1} =====')
    if keys is None:
      if thisTrial['rwd'] == 1:
          print('| {0} | + |{2}{1}{2}|'.format(thisTrial['magn_left'],
              thisTrial['magn_right'], mark))
      else:
          print('|{2}{0}{2}| + | {1} |'.format(thisTrial['magn_left'],
              thisTrial['magn_right'], mark))
    elif keys[0][0] == key_left:
      if thisTrial['rwd'] == 1:
          print('< {0} > + |{2}{1}{2}|'.format(thisTrial['magn_left'],
              thisTrial['magn_right'], mark))
      else:
          print('<{2}{0}{2}> + | {1} |'.format(thisTrial['magn_left'],
              thisTrial['magn_right'], mark))
    elif keys[0][0] == key_right:
      if thisTrial['rwd'] == 1:
          print('| {0} | + <{2}{1}{2}>'.format(thisTrial['magn_left'],
              thisTrial['magn_right'], mark))
      else:
          print('|{2}{0}{2}| + < {1} >'.format(thisTrial['magn_left'],
              thisTrial['magn_right'], mark))
    print('Account = {}'.format(account))

### Global keys ################################################################
event.globalKeys.clear()
event.globalKeys.add(
    key=key_quit,
    func=core.quit)

### Task structure #############################################################
trial_block = create_blocks(
    N_trials=N_trials,
    N_reversal=N_reversal,
    N_min_stable=N_min_stable)
trial_rwd = create_schedule(
    N_trials=N_trials,
    trial_block=trial_block,
    reward_probability=reward_probability)
trial_magn_left = create_split(
    N_trials=N_trials,
    reward_total=reward_total,
    reward_minimum=reward_minimum)
trial_magn_right = [reward_total - magn for magn in trial_magn_left]
trialList = gen_trialList(
    (trial_block, 'block'),
    (trial_rwd, 'rwd'),
    (trial_magn_left, 'magn_left'),
    (trial_magn_right, 'magn_right'))

# Dialogue box
dlg = gui.Dlg(title="Probabilistic reversal learning training (DecideNet)")
dlg.addText('ENSURE THAT NUM LOCK IS OFF!')
dlg.addText('Subject info')
dlg.addField('Id:')
dlg.addText('Experiment settings')
dlg.addField('Condition:', choices=['rew','pun'])
dlg.addField('Group:', choices=[0, 1])
dlg_data = dlg.show()

if dlg.OK:
    subject_id = dlg_data[0]
    condition = dlg_data[1]
    group = dlg_data[2]
    filename = 'logs/'+ subject_id +'_prl_DecideNet_training' + '_' + condition
else:
    print('Canceled. Quitting...')
    core.quit()

### Objects ####################################################################
mywin = visual.Window(
    size=win_size,
    fullscr=win_fullscr,
    monitor=win_monitor,
    screen=win_screen,
    color=win_color,
    units=win_units)
mywin.mouseVisible = win_mouse_visible
left_box = visual.Rect(
    win=mywin,
    units=box_units,
    pos=[-box_separation/2, 0],
    width=box_width,
    height=box_height,
    lineWidth=0)
right_box = visual.Rect(
    win=mywin,
    units=box_units,
    pos=[box_separation/2, 0],
    width=box_width,
    height=box_height,
    lineWidth=0)
fix_circle = visual.Circle(
    win=mywin,
    pos=[0, 0],
    units=circle_units,
    radius=circle_radius,
    edges=circle_edges,
    fillColor=circle_color,
    lineWidth=0)
down_bar = visual.Rect(
    win=mywin,
    units=box_units,
    pos=[0, bar_y],
    width=bar_width,
    height=bar_height,
    fillColor=bar_color,
    lineWidth=0)
down_bar_tick1 = visual.Rect(
    win=mywin,
    units=box_units,
    pos=[0, bar_y],
    width=bar_tick_width,
    height=bar_height,
    fillColor=bar_tick1_color,
    lineWidth=0)
down_bar_tick2 = visual.Rect(
    win=mywin,
    units=box_units,
    pos=[0, bar_y],
    width=bar_tick_width,
    height=bar_height,
    fillColor=bar_tick2_color,
    lineWidth=0)
down_bar_fill = visual.Rect(
    win=mywin,
    units=box_units,
    pos=[0, bar_y],
    width=0,
    height=bar_height,
    fillColor=bar_fill_color,
    lineWidth=0)
left_text = visual.TextStim(
    win=mywin,
    units=text_units,
    pos=[-box_separation/2, 0],
    text='',
    color=text_digit_color,
    height=text_digit_height)
right_text = visual.TextStim(
    win=mywin,
    units=text_units,
    pos=[box_separation/2, 0],
    text='',
    color=text_digit_color,
    height=text_digit_height)
fix_text = visual.TextStim(
    win=mywin,
    units=text_units,
    pos=[0, 0],
    text='',
    color=text_digit_color,
    height=text_digit_height)
up_text = visual.TextStim(
    win=mywin,
    units=text_units,
    pos=[0, text_up_y],
    text='',
    color=text_letter_color,
    height=text_letter_height)
info_text = visual.TextStim(
    win=mywin,
    units=text_units,
    pos=[0, 0],
    wrapWidth=25,
    text='',
    color=text_letter_color,
    height=text_info_height)
feedback_text = visual.TextStim(
    win=mywin,
    units=text_units,
    pos=[0, text_feedback_y],
    text='',
    color=text_letter_color,
    height=text_feedback_height)
feedback_box = visual.Rect(
    win=mywin,
    units=box_units,
    pos=[0, box_feedback_y],
    width=box_feedback_width,
    height=box_feedback_height,
    lineWidth=0)

### Dialogue box dependent settings ############################################
# Account bar settings
if condition == 'rew':
    account = 0
    account_max = account_max_rew
    feedback_text.setText('Często wygrywający box:')
    down_bar_tick1.setPos([bar_width * (tick1_rew/account_max - .5), bar_y])
    down_bar_tick2.setPos([bar_width * (tick2_rew/account_max - .5), bar_y])
elif condition == 'pun':
    account_max = account_max_pun
    account = account_max
    feedback_text.setText('Często przegrywający box:')
    down_bar_tick1.setPos([bar_width * (tick1_pun/account_max - .5), bar_y])
    down_bar_tick2.setPos([bar_width * (tick2_pun/account_max - .5), bar_y])
down_bar_fill.setPos([(bar_width/2)*(account/account_max-1), bar_y])
down_bar_fill.setWidth(bar_width*account/account_max)

# Box color setting
if group == 0:
    box_left_color = box_color1
    box_right_color = box_color2
elif group == 1:
    box_left_color = box_color2
    box_right_color = box_color1
left_box.setFillColor(box_left_color)
right_box.setFillColor(box_right_color)

# Data handlers
th = data.TrialHandler(
    trialList=trialList,
    nReps=1,
    method='sequential')
exp = data.ExperimentHandler(
    name='prl_DecideNet_training',
    version='1.2',
    dataFileName=filename,
    extraInfo={'subject_id': subject_id,
               'condition': condition,
               'group': group})
exp.addLoop(th)

# Create clock
timer = core.Clock()

################################################################################
################################ TASK START ####################################
################################################################################

### Ask participant for readiness ##############################################
info_text.setText('Gdy będziesz gotowy(-wa) naciśnij przycisk "s".')
info_text.draw(); mywin.flip()
event.waitKeys(keyList=[key_pulse])

idx = -1 # Indexind trials
for thisTrial in th:
    idx += 1

    ### wait for trial (iti) ###################################################
    set_phase('iti')
    # Set proper color of a feedback box
    if thisTrial['block'] == -1:
        feedback_box.setFillColor(box_left_color)
    else:
        feedback_box.setFillColor(box_right_color)

    timer.reset(uniform(time_range_iti[0], time_range_iti[1]))
    while timer.getTime() < 0:
        draw_phase('iti', idx, win=mywin)

    ### decision phase #########################################################
    set_phase('dec_before')
    draw_phase('dec', idx, win=mywin)

    timer.reset(time_decision)
    keys = event.waitKeys(
        maxWait=time_decision,
        keyList=[key_left, key_right],
        timeStamped=False,
        clearEvents=True)

    set_phase('dec_after')
    while timer.getTime() < 0:
        draw_phase('dec', idx, win=mywin)

    ### wait phase (isi) #######################################################
    set_phase('isi')

    timer.reset(uniform(time_range_isi[0], time_range_isi[1]))
    while timer.getTime() < 0:
        draw_phase('isi', idx, win=mywin)

    ### outcome phase ##########################################################
    won_magn = outcome_magn(thisTrial, keys,
                            key_left=key_left, key_right=key_right,
                            condition=condition)
    won_bool = bool(won_magn)

    # Update account
    if condition == 'rew':      account += won_magn
    elif condition == 'pun':    account -= won_magn

    # Draw outcome screen ASAP
    timer.reset(time_outcome)
    set_phase('out')
    draw_phase('out', idx, win=mywin)

    # Save participant's responses
    th.addData('acc_after_trial', account)
    th.addData('won_bool', won_bool)
    th.addData('won_magn', won_magn)
    if keys is None:    th.addData('response', None)
    else:               th.addData('response', keys[0][0])
    exp.nextEntry()

    # Wait until next trial
    while timer.getTime() < 0:
        draw_phase('out', idx, win=mywin)

    # Info for the researcher
    print_trial(thisTrial, keys, condition, account, th)

### After experiment ###########################################################
exp.saveAsWideText(
    fileName=filename,
    delim=',')
print('\nTask logs saved!')

# Info screen after the task
if condition == 'rew':
    info_text.setText('Udało Ci się zebrać {} punktów.'.format(account))
elif condition == 'pun':
    info_text.setText('Pozostało Ci {} punktów.'.format(account))

timer.reset(time_decision)
while timer.getTime() < 0:
    info_text.draw()
    mywin.flip()
