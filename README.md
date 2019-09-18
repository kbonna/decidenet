# DecideNet

Neuroimaging project aimed at understanding functional brain network dynamics during value-based decision making.

## Project description

Participants underwent single MRI scanning session during which they performed two conditions of probabilistic reversal learning (PRL) task.  They also completed Barratt Impulsiveness Scale BIS-11 (Patton et al. 1995) and Specific Risk Taking Scale DOSPERT (Blais and Weber 2006). Total number of N=32 subjects participated in the project. 

## Code organization

- `behavioral_analysis` directory contains script required to run entire analysis of response data acquired from the subjects: task log preprocessing, response visualisations and Bayesian modelling
- `fmri_preparation` directory enables data preparation for primary analysis. Raw NIfTI dataset is converted BIDS-compliant dataset. Then data is preprocessed by [fmriprep](https://github.com/poldracklab/fmriprep) package
- `activation analysis` directory contains code to replicate GLM findings
- `dn_utils` is small python module containing useful functions used throughout the analysis 
