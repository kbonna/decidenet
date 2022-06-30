# 💻 Code for DecideNet neuroimaging project

This repository contains code required to reproduce results from my Doctoral project "DecideNet." This project is dedicated to examining and describing functional network reconfiguration during prediction error processing. The analysis is focused on three perspectives: (1) behavioral analysis using the Bayesian modeling approach, (2) activation analysis using model-based fMRI approach, and (3) connectivity analysis using beta-series correlation approach. The analysis is conducted using open-source Python packages for neuroimaging like `nilearn` and `nibabel` and custom Python code.

## 📖 Thesis

[Here](https://umk.bip.gov.pl/fobjects/download/1266495/rozprawa-doktorska-pdf.html) you can find my PhD thesis based on the findings produced by this codebase.

## 📂 Directory structure:
- `activation_analysis`: code for BOLD activation analysis
- `behavioral_analysis`: code for behavioral modeling
  - `matjags-dn`: JAGS code for hierarchical latent mixture model and Bayesian model analysis
- `connectivity_analysis`: code for functional connectivity analysis
  - `parcellations` contains brain parcellation tables with MNI coordinates and ROI/LSN names useful for both types of connectivity analysis
  - `bsc` contains beta-series correlation analysis files
  - `ppi` contains psychophysiological interaction analysis files 
  - `signal_extraction` contains general functional data processing useful for connectivity analysis
- `fmri_preparation`: code for data preparation (BIDS structure, preprocessing)
- `dn_utils`: helper functions used throughout the project
- `prl_task`: PsychoPy code for a task used in fMRI scanner 
