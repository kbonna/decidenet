# Activation analysis

## Files

Analysis of activation requires preprocessed functional data located within `<path_bids>/derivatives/fmriprep` directory, task events onsets stored within `behavioral_data_clean_all.npy` and individual model parameters estimated during behavioral analysis. For this model-based fMRI analysis prediction-error dependent and condition independent (PDCI) behavioral model is used to estimate modulated regressors. Point estimates of positive and negative learning rates are stored in `alpha_pdci_mle.npy` file containing array of size <img src="https://render.githubusercontent.com/render/math?math=$N_{subjects} \times 2$">.

## Fields

Activation analysis is divided into three steps:
1. Calculating parametric modulations.
2. Creating and estimating first level model.
3. Creating and estimating second level model.

### Parametric modulations

Parametric modulations are calculated using **prediction-error dependent condition independent (PDCI)** reinforcement learning model. This model was selected as the winning model using Bayesian cognitive modeling approach. It assumes two separate learning rates `alpha_plus` for rewarded / not punished trials and `alpha_minus` for punished / not rewarded trials and precision `beta` parameter controlling sensitivity to difference in value. For each subject point estimates of learning rates are calculated as the arguments maximizing posterior distributions (maximum likelihood estimates; MLE).

Then, these MLE estimates are used to estimate individual expected probability for correct choice in each trial (`wbci` for chosen side). Finally trial-wise prediction errors are calculated as `1 - wcor[t]` for rewarded / not punished trials or `0 - wcor[t]` for punished / not-rewarded trials. 

> Note that since expected probability for side for being correct is bounded between 0 and 1, sign of prediction error always reflects correct and incorrect choices.

Parametric modulations for different latent variables are named `modulation_<latent-variable-name>`. They are stored as aggregated arrays of size <img src="https://render.githubusercontent.com/render/math?math=$N_{subjects} \times N_{conditions} \times N_{trials}$"> within `<path_bids>/derivatives/nistats/modulations` directory.

### First level analysis

First level GLM analyis consist of three steps. First, parametrically modulated regressors are creating by convolving parametric modulations with hemodynamic response function (HRF). Second, full GLM design matrix is created from task regressors of interest, task regressors of no interest, confound regressors and drift regressors. Third, GLM is estimated, and individual statistical maps for predefined contrasts are calculated and saved.

Parametrically modulated regressors are implemented as custom `Regressor` class living within `dm_utils.glm_utils` module. `Regressor` class is a simple wrapper for nistats [function](https://nistats.github.io/modules/generated/nistats.design_matrix.make_first_level_design_matrix.html) for design matrix creation. Instance property `dm_column` corresponds to convolved regressor of interest. It allows to define and investigate modulated regressors isolation from the rest of design matrix. List of regressors objects is then passed to `my_make_first_level_design_matrix` custom function to create full design matrix. Regressors are named according to convention `reg_<phase>_<type>`. For example, `reg_res_lbp` represents regressors time-locked to the response onset and reflects all left button press events. Full table of regressors can be found within `dn_aa_01_first_level.ipynb` notebook.

After fitting GLM, individual contrasts are estimated and saved as statistical maps within `<path_bids>/derivatives/nistats/first_level_output` directory.

![Trial events figure](trial_events.png?raw=true "Trial Events")

### Second level analysis


