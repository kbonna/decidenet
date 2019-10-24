Behavioral data conventions

## Files
Behavioral data is stored in two separate files `behavioral_data_clean_all.npy` and `behavioral_data_clean_all.json`. File with *.npy extension is array containing actual data and corresponding *.json file contains names of corresponding fields. 
1. Aggregated behavioral data loaded from `behavioral_data_clean_all.npy` is called is represented as `beh` variable in the code. 
2. Corresponding metadata dictionary is called as `meta` variable in the code. 

## Fields
Aggregated behavioral data contains **all task information, all behavioral responses  / recordings and timing of all scanner events**.  Multidimensional array `beh` aggregates data for subjects, conditions, trials for different events. Shape of the array is $N_{subjects} \times N_{conditions} \times N_{trials} \times N_{variables}$. 
### Task structure variables
| variable name | code | values |
|--|--|--|
| Correct side (block) | `block` | -1 / 1 |
| Chosen side (block) | `block_bci` | -1 / 1 |
| Correct side | `side` | -1 / 1 |
| Chosen side | `side_bci` | -1 / 1 |
| Left-side magnitude | `magn_left` | `int` from -45 to 45 |
| Right-side magnitude | `magn_right` | `int` from -45 to 45 |

Note the distinction between correct and chosen side. Correct is related to human utility interpretation where correct means *rewarded or not punished*. Chosen is related to **being chosen interpretation** where chosen means *rewarded or punished* (it is used for RL algorithm calculations). In the reward condition all sides that are chosen are correct at the same time, whereas in the punishment condition all sides that are chosen are incorrect. In different parts of the code, different interpretation are used, so they are separately defined to avoid confusion.

### Subject's behavioral variables
| variable name | code | values |
|--|--|--|
| Response | `response` | -1 / 0 / 1 |
| Reaction time (s) | `rt` | `float` from 0 to 1.5 / `nan` |
| Correct choice label | `won_bool` | 0 / 1 |
| Account update | `won_magn` | `int` from -45 to 45 |
| Account balance (after trial) | `acc_after_trial` | `int` from 0 to 2300 |

Response variable is coded the same way as in task structure variables. It is often desirable to convert -1 / 1 coding into 0 / 1 coding, as it is more convenient for JAGS to represent direct probabilities that one side will be chosen. However, in order to incorporate intuitive representation of missing responses as 0, default coding is -1 / 0 / 1 for left / miss / right. 

### Scanner timing variables
| variable name | code |
|--|--|
| main fmri clock time (used for analysis)|`onset_[iti/dec/isi/out]`|
| planned time of stimulus presentation (planned time for stimulus presentation (always slightly behind registered time)| `onset_[iti/dec/isi/out]_plan`|
| time registered with global clock (only for synchronization validation purpose, not used for analysis) |  `onset_[iti/dec/isi/out]_glob` 

Subscripts represent different events related with single trial. `iti` reflects inter-trial-interval onset (before each trial), `dec` is decision phase onset, `isi` is inter-stimulus-interval onset (waiting phase onset) and `out` is outcome phase onset.

## Latent algorithm variables

### Algorithm parameters
| variable name | code | text |
|--|--|--|
| learning rate | `alpha` | $\alpha$ |
| learning rate for positive PE | `alpha_plus` / `alpha[first_index]` | $\alpha_+$ |
| learning rate for negative PE | `alpha_minus` / `alpha[second_index]` | $\alpha_-$ |
| learning rate for reward condition | `alpha_rew` / `alpha[first_index]` | $\alpha_{rew}$ |
| learning rate for punishment condition | `alpha_pun` / `alpha[second_index]` | $\alpha_{pun}$ |
| inverse-temperature | `beta` | $\beta$ |
| loss aversion (prospect theory) | `gamma` | $\gamma$ |
| risk aversion (prospect theory) | `delta` | $\delta$ |

Note that learning rates can be represented either as separate vectors or arrays with columns corresponding to different learning rates. `first_index` and `second_index` are used instead of specific values because in Python `first_index` is 0, whereas in MATLAB it is 1 (and so on). In case of four learning rates model first dimension corresponds to task condition and second dimension corresponds to PE sign. 


### Tracked and computed variables
| variable name | code | text |
|--|--|--|
| Expected probability for side for being chosen | `wbci` / `wbci_l` and `wbci_r` | $\rho$ |
| Expected probability for side for being correct | `wcor` / `wcor_l` and `wcor_r` | $p$|
| Expected value (utility) | `eval` / `eval_l` and `eval_r` | $v$|
| Reward magnitude | `magn` / `magn_l` and `magn_r` | $x$|
| Reward utility | `util` / `util_l` and `util_r` | $u$ |
| Choice probability | `prob` / `prob_l` and `prob_r` | $P$|

Whenever `l` and `r` suffixes are not used, **array representation** of tracked variables is assumed. In array representation $N_{trials} \times N_{sides}$ array is representing variable state across task time course for both sides simultaneously. 

