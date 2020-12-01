"""This is PsychoPy implementation of probabilistic reversal learning (PRL) task
used in fMRI study in DecideNet project.

Participants are asked to make sequential choices of left or right box. They
have to learn boxes probabilities from experience and use it along reward size
displayed on the boxes to make accurate decisions. More probable box will change
several times during the task. Task consists of two separate conditions: reward
and punishment. In reward condition subjects have to acquire as many points as
possible by preferring box that is more frequently chosen and has higher
magnitude. Conversly, in punishment condition subjects are initially endowed
with some amount of points, and their goal is to loose as less points as
possible. In this condition best strategy is to choose box that is lower in
magnitudeless and less frequently chosen.

Task parameters are saved within separate file prl_DecideNet_config.py.
Parameters inlude graphicial options (colors and sizes of objects), task options
(e.g. number of trials, number of reversals, box probabilities) and timing
options (duration of phases and intervals).

Author: Kamil Bonna
Version: 1.2
Updates:
    08-04-2019: version 1.2 released
    -> changed entire timing method to non-slip timing
    -> glob_clock is used just for post experiment checking of Synchronisation
    -> fmri_clock is directly correlated with scanner pulse timing and it is
    meant to be used in GLM model analysis
    19-04-2019: minor changes
    -> added two separate account bar ticks representing two possible reward
    thresholds, their position as well as account capacity can be changed for
    both task conditions
"""
from prl_DecideNet_config_fmri import *
from utils import *
from collections import OrderedDict
from psychopy import visual, core, event, data, gui
from random import shuffle
from time import strftime
import pandas as pd
import numpy as np
import os.path


def draw_objects(*args):
    """Drawing any number of objects."""
    for arg in args:
        arg.draw()

def draw_phase(phase, win=None):
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

def time2frames(time: float, refresh_rate: int) -> int:
    """Converts time in s to number of frames ensuring even number of frames"""
    n_frames = refresh_rate * time
    return int(n_frames)

def StaffordRandFixedSum(n, u, nsets):
    """
    Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
    EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
    OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    The views and conclusions contained in the software and documentation are
    those of the authors and should not be interpreted as representing official
    policies, either expressed or implied, of Paul Emberson, Roger Stafford or
    Robert Davis.
    Includes Python implementation of Roger Stafford's randfixedsum implementation
    http://www.mathworks.com/matlabcentral/fileexchange/9700
    Adapted specifically for the purpose of taskset generation with fixed
    total utilisation value
    Please contact paule@rapitasystems.com or robdavis@cs.york.ac.uk if you have
    any questions regarding this software.
    """
    if n < u:
        return None

    #deal with n=1 case
    if n == 1:
        return np.tile(np.array([u]), [nsets, 1])

    k = min(int(u), n - 1)
    s = u
    s1 = s - np.arange(k, k - n, -1.)
    s2 = np.arange(k + n, k, -1.) - s

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, n + 1):
        tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
        w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
        tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + \
            (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

    x = np.zeros((n, nsets))
    rt = np.random.uniform(size=(n - 1, nsets))  # rand simplex type
    rs = np.random.uniform(size=(n - 1, nsets))  # rand position in simplex
    s = np.repeat(s, nsets)
    j = np.repeat(k + 1, nsets)
    sm = np.repeat(0, nsets)
    pr = np.repeat(1, nsets)

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0):
        e = rt[(n - i) - 1, ...] <= t[i - 1, j - 1]
        sx = rs[(n - i) - 1, ...] ** (1.0 / i)  # next simplex coord
        sm = sm + (1.0 - sx) * pr * s / (i + 1)
        pr = sx * pr
        x[(n - i) - 1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1, ...] = sm + pr * s

    #iterated in fixed dimension order but needs to be randomised
    #permute x row order within each column
    for i in range(0, nsets):
        x[..., i] = x[np.random.permutation(n), i]

    return x.T.tolist()

def gen_intervals(N_trials, n_range_interval):
    '''This function generate list of shuffled intervals for all trials.

    Parameters:
        N_trials (int): number of trials in experiment
        n_range_interval (list): possible durations of intervals as a number of
            frames

    Returns:
        intervals (list): list of intervals of length N_trials
    '''
    if (N_trials % len(n_range_interval)) != 0:
        raise ValueError('Interval cannot be properly counterbalanced!')

    mult_factor_interval = N_trials // len(n_range_interval)
    intervals = n_range_interval * mult_factor_interval

    try:
        shuffle(intervals)
    except AttributeError:
        print("shuffle not found. Try 'from random import shuffle'")

    return intervals

def getpulse():
    """Collecting scanner pulses."""
    global pulses
    global glob_clock
    pulses.append(glob_clock.getTime())

rgb2psy = lambda x : [val / 127.5 - 1 for val in x]

def save_pulses(pulses, filename):
    """Saves pulse onsets and spacing between them into a csv file"""
    if pulses:
        puldur = [p1-p2 for p1, p2 in zip(pulses[1:], pulses[:-1])]
        puldur = [0] + puldur
        df = pd.DataFrame(
            {'onset': pulses,
             'spacing': puldur})
        df.index = np.arange(1, len(df)+1)
        df.to_csv(filename + '_pulse.csv', sep=",", columns=['onset','spacing'])

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
event.globalKeys.add(
    key=key_pulse,
    func=getpulse,
    name='record_fmri_pulses')

### Task structure #############################################################
# Randomize intervals
isi = np.array(StaffordRandFixedSum(N_trials, N_trials/2, nsets=1)[0])
isi = isi * (time_range_isi[1] - time_range_isi[0]) + time_range_isi[0]
iti = np.array(StaffordRandFixedSum(N_trials, N_trials/2, nsets=1)[0])
iti = iti * (time_range_iti[1] - time_range_iti[0]) + time_range_iti[0]

onset_iti, onset_isi = [], []
onset_dec, onset_out = [], []

t = 0
for t_iti, t_isi in zip(iti, isi):
    onset_iti.append(t); t += t_iti
    onset_dec.append(t); t += time_decision
    onset_isi.append(t); t += t_isi
    onset_out.append(t); t += time_outcome

# Create trial structure
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
    (trial_magn_right, 'magn_right'),
    (onset_iti, 'onset_iti_plan'),
    (onset_isi, 'onset_isi_plan'),
    (onset_dec, 'onset_dec_plan'),
    (onset_out, 'onset_out_plan'))

# Scanner pulses
pulses = []

# Dialogue box
dlg = gui.Dlg(title="Probabilistic reversal learning (DecideNet)")
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
    filename = 'logs/'+ subject_id +'_prl_DecideNet' + '_' + condition
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

### Dialogue box dependent settings ############################################
# Account bar settings
if condition == 'rew':
    account = 0
    account_max = account_max_rew
    down_bar_tick1.setPos([bar_width * (tick1_rew/account_max - .5), bar_y])
    down_bar_tick2.setPos([bar_width * (tick2_rew/account_max - .5), bar_y])
elif condition == 'pun':
    account = account_max_pun
    account_max = account_max_pun
    down_bar_tick1.setPos([bar_width * (tick1_pun/account_max - .5), bar_y])
    down_bar_tick2.setPos([bar_width * (tick2_pun/account_max - .5), bar_y])
down_bar_fill.setPos([(bar_width/2) * (account/account_max - 1), bar_y])
down_bar_fill.setWidth(bar_width * account/account_max)

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
    name='prl_DecideNet',
    version='2.0',
    dataFileName=filename,
    extraInfo={'subject_id': subject_id,
               'condition': condition,
               'group': group})
exp.addLoop(th)

# Ask participant for readiness
info_text.setText('Gdy będziesz gotowy(-wa) naciśnij dowolny przycisk.')
info_text.draw(); mywin.flip()
event.waitKeys(keyList=[key_left, key_right])
print('Participant {} is ready.'.format(subject_id))
info_text.setText('Zadanie rozpocznie się za moment.')
info_text.draw(); mywin.flip()

# Create clocks
glob_clock = core.MonotonicClock() # Just for post experimental check
fmri_clock = core.Clock() # Main clock synchronised with fMRI trigger

################################################################################
################################ TASK START ####################################
################################################################################

# Wait for first scanner pulse
event.waitKeys(keyList=[key_pulse])
fmri_clock.reset() # Synchronisation with first pulse

idx = -1 # Indexing trials
for thisTrial in th:
    idx += 1

    ### wait for trial (iti) ###################################################
    set_phase('iti')
    th.addData('onset_iti', fmri_clock.getTime())
    th.addData('onset_iti_glob', glob_clock.getTime())

    while fmri_clock.getTime() < onset_dec[idx]:
        draw_phase('iti', win=mywin)

    ### decision phase #########################################################
    set_phase('dec_before')
    th.addData('onset_dec', fmri_clock.getTime())
    th.addData('onset_dec_glob', glob_clock.getTime())
    rt_onset = fmri_clock.getTime() # For RT calculation
    draw_phase('dec', win=mywin)

    keys = event.waitKeys(
        maxWait=time_decision,
        keyList=[key_left, key_right],
        timeStamped=fmri_clock,
        clearEvents=True)

    set_phase('dec_after')
    while fmri_clock.getTime() < onset_dec[idx] + time_decision:
        draw_phase('dec', win=mywin)

    ### wait phase (isi) #######################################################
    set_phase('isi')
    th.addData('onset_isi', fmri_clock.getTime())
    th.addData('onset_isi_glob', glob_clock.getTime())
    while fmri_clock.getTime() < onset_out[idx]:
        draw_phase('isi', win=mywin)

    ### outcome phase ##########################################################
    won_magn = outcome_magn(thisTrial, keys,
                            key_left=key_left, key_right=key_right,
                            condition=condition)
    won_bool = bool(won_magn)

    # Update account
    if condition == 'rew':      account += won_magn
    elif condition == 'pun':    account -= won_magn

    # Draw outcome screen ASAP
    set_phase('out')
    th.addData('onset_out', fmri_clock.getTime())
    th.addData('onset_out_glob', glob_clock.getTime())
    draw_phase('out', win=mywin)

    # Save participant's responses
    th.addData('acc_after_trial', account)
    th.addData('won_bool', won_bool)
    th.addData('won_magn', won_magn)
    if keys is None:
        th.addData('rt', None)
        th.addData('response', None)
    else:
        th.addData('rt', keys[0][1] - rt_onset)
        th.addData('response', keys[0][0])
    exp.nextEntry()

    # Wait until next trial
    while fmri_clock.getTime() < onset_out[idx] + time_outcome:
        draw_phase('out', win=mywin)

    # Info for the researcher
    print_trial(thisTrial, keys, condition, account, th)

### Save data ##################################################################
# Behavioral part
exp.saveAsWideText(
    fileName=filename,
    delim=',')
print('\nTask logs saved!')
# Scanner part
save_pulses(pulses, filename)
print('Pulse file saved!')

# Set and display info screen after the task
if condition == 'rew':
    if account >= tick1_rew:
        info_text.setText('Gratulacje! Zyskujesz 10 zł.'.format(account))
        print('Participant won 10 zl.')
    elif account >= tick2_rew:
        info_text.setText('Gratulacje! Zyskujesz 20 zł.'.format(account))
        print('Participant won 20 zl.')
    else:
        info_text.setText('Zyskujesz 0 zł.')
        print('Participant won 0 zl.')
elif condition == 'pun':
    if account >= tick1_pun:
        info_text.setText('Gratulacje! Pozostało Ci 20 zł.'.format(account))
        print('Participant won 20 zl.')
    elif account >= tick2_pun:
        info_text.setText('Gratulacje! Pozostało Ci 10 zł.'.format(account))
        print('Participant won 10 zl.')
    else:
        info_text.setText('Pozostało Ci 0 zł.'.format(account))
        print('Participant won 0 zl.')

for frame in range(time2frames(time_info_after, refresh_rate)):
    info_text.draw()
    mywin.flip()
print('Task ended.')
