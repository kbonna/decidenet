"""This file contains stimulus and task setting for probabilistic reversal
learning task training. Options are organized into sections corresponding to
certain task objects or mechanisms.
"""
# Convert RGB color representation into PsychoPy default one
rgb2psy = lambda x : [val / 127.5 - 1 for val in x]

### Task construction ##########################################################
N_trials = 50
N_reversal = 4
N_min_stable = 7
N_trials_feedback = 25
reward_total = 50
reward_minimum = 5
reward_probability = 0.8

# Account bar settings
account_max_rew = 1100
account_max_pun = 600
tick1_rew = 550
tick2_rew = 1100
tick1_pun = 300
tick2_pun = 0

# Task timing (in seconds)
time_decision = 1.5
time_outcome = 1.5
time_range_isi = (2, 3)
time_range_iti = (2, 3)
refresh_rate = 60

# Available keys
key_quit = 'q'
key_left = 'z'
key_right = 'm'
key_pulse = 's'

### Visual properties ##########################################################
# Window (screen)
win_size = (800, 600)
win_monitor = 'testMonitor'
win_screen = 0 # 1 is second screen.
win_fullscr = True
win_mouse_visible = True
win_color = [-1, -1, -1]
win_units = 'norm'

# Boxes to choose
box_units = 'cm'
box_width = 5
box_height = 5
box_separation = 15
box_feedback_y = -8.5
box_feedback_height = 3
box_feedback_width = 3
box_color1 = rgb2psy([86, 180, 233])
box_color2 = rgb2psy([240, 228, 66])

# Account bar
bar_height = 1
bar_width = 20
bar_y = -4
bar_color = [-.8, -.8, -.8]
bar_tick_width = .15
bar_tick_color = [1, 1, 1]
bar_tick1_color = [1, 1, 1]
bar_tick2_color = [1, 1, 1]
bar_fill_color = [-.4, -.4, -.4]

# Fixation circle
circle_units = 'cm'
circle_radius = 2
circle_edges = 50
circle_color = [-.4, -.4, -.4]
circle_line_width = 8

# Digits and informations
text_units = 'cm'
text_digit_height = 1.5
text_letter_height = 1.5
text_digit_color = [-1, -1, -1]
text_letter_color = [1, 1, 1]
text_info_height = 1
text_up_y = 5
text_feedback_y = -6
text_feedback_height = 1
