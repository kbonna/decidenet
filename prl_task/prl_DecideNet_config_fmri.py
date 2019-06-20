"""This file contains stimulus and task setting for probabilistic reversal
learning task. Options are organized into sections corresponding to certain
task objects or mechanisms.

Version for fMRI scanning, do not change!
"""
# Convert RGB color representation into PsychoPy default one
rgb2psy = lambda x : [val / 127.5 - 1 for val in x]

### Task construction ##########################################################
N_trials = 110
N_reversal = 4
N_min_stable = 15
reward_total = 50
reward_minimum = 5
reward_probability = 0.8

# Account bar settings
account_max_rew = 2300
account_max_pun = 1200
tick1_rew = 1150
tick2_rew = 2300
tick1_pun = 600
tick2_pun = 0

# Task timing (in seconds)
time_decision = 1.5     # Time to make decision
time_outcome = 1.5      # Time to see outcome
time_range_isi = (3, 7) # Interval between decision and outcome
time_range_iti = (3, 7) # Interval between trials
time_info_after = 10    # Time to see experiment summary (collected money)
refresh_rate = 60
''' Total experiment time is:

-> mean trial time: mtt = 1.5 (dec) + 1.5 (out) + 5 (isi) + 5 (iti) = 13 seconds
-> dummy scan time: dst = 5 (N_dummy) * 2 (TR)                      = 10 seconds
-> time info after: iat = (time_info_after)                         = 10 seconds

-> total time: tt = 110 (N_trials) * mtt + dst + iat = 1430 + 10 + 10 =
                  = 1450 seconds = 24 minutes 10 seconds

-> extra time: 20 seconds
-> sequence duration = tt + extra_time = 24 minutes 30 seconds
'''

# Available keys
key_quit = 'q'
key_left = 'a'          # Left button SyncBox emulation
key_right = 'd'         # Right button SyncBox emulation
key_pulse = 's'         # Scanner pulse SyncBox emulation

### Visual properties ##########################################################
# Window (screen)
win_size = (800, 600)
win_monitor = 'testMonitor'
win_screen = 1          # For second screen, select 1.
win_fullscr = True
win_mouse_visible = False
win_color = [-1, -1, -1]
win_units = 'norm'

# Boxes to choose
box_units = 'cm'
box_width = 5
box_height = 5
box_separation = 15
box_color1 = rgb2psy([86, 180, 233])
box_color2 = rgb2psy([240, 228, 66])

# Account bar
bar_height = 1
bar_width = 20
bar_y = -4
bar_color = [-.8, -.8, -.8]
bar_tick_width = .15
bar_tick1_color = [1, 1, 1]
bar_tick2_color = [1, 1, 1]
bar_fill_color = [-.4, -.4, -.4]

# Fixation circle
circle_units = 'cm'
circle_radius = 2
circle_edges = 50
circle_color = [-.4, -.4, -.4]
circle_line_width = 10

# Digits and informations
text_units = 'cm'
text_digit_height = 1.5
text_letter_height = 1.5
text_digit_color = [-1, -1, -1]
text_letter_color = [1, 1, 1]
text_info_height = 1
text_up_y = 5
