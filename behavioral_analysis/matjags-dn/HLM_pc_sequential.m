clear; clc;

%% Load and process experiment data
% data location
root = '/home/kmb/Desktop/Neuroscience/Projects/BONNA_decide_net/data/main_fmri_study/sourcedata/behavioral/';
fname_beh = 'behavioral_data_clean_all.mat';
fname_meta = 'behavioral_data_clean_all.json';

% load behavioral and metadata
load(strcat(root, fname_beh));
fid = fopen(strcat(root, fname_meta)); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
meta = jsondecode(str);
clearvars -except beh meta

% split relevant data 
xa = beh(:, :, :, strcmp(meta.dim4, 'magn_left'));  % reward magnitude for left box
xb = beh(:, :, :, strcmp(meta.dim4, 'magn_right')); % reward magnitude for right box
resp = beh(:, :, :, strcmp(meta.dim4, 'response')); % chosen side
rwd = beh(:, :, :, strcmp(meta.dim4, 'rwd'));       % rewarded / punished side (bci)

nSubjects = numel(meta.dim1);
nConditions = numel(meta.dim2);
nTrials = numel(meta.dim3);
nPredErrSign = 2;
nModels = 4;                    
resp(resp == 0) = NaN;                              % set missing values to NaNs
resp = (resp + 1) / 2;                              % 0: left box; 1: right box; NaN: miss
rwd  = (rwd + 1) / 2;                               % 0: left box; 1: right box (bci)  
rwdwin = rwd;                                      
rwdwin(:, 2, :) = 1 - rwdwin(:, 2, :);              % convert from being chosen to favor interpretation

%% JAGS Setup
% Parameters
fname_model = fullfile(pwd, strcat(mfilename, '.txt'));
doparallel = 1;                                     % parallelization flag
thinning = 1;                                       % thinning parameter
nChains = 4;                                        
nBurnin = 2500;
nSamples = 7500;

% Initialize Markov chain values
for i=1:nChains
    S = struct;                                     % random initial values
    init0(i) = S;
end

% Assign MATLAB variables to the observed JAGS nodes
% Uncomment lines below to run model "diconnected" with the data
% resp(resp<Inf) = NaN;
% resp(1)=1;
datastruct = struct(...
    'xa', xa, ...
    'xb', xb, ...
    'rwdwin', rwdwin, ...
    'resp', resp, ...
    'nSubjects', nSubjects, ...
    'nConditions', nConditions, ...
    'nPredErrSign', nPredErrSign, ...
    'nTrials', nTrials, ...
    'nModels', nModels);

monitorparams = {...
    'z', ...
    'a_alpha_pici', 'b_alpha_pici', 'mu_beta_pici', 'sigma_beta_pici', ...
    'alpha_pici', 'beta_pici', ...
    'a_alpha_picd', 'b_alpha_picd', 'mu_beta_picd', 'sigma_beta_picd', ...
    'alpha_picd', 'beta_picd', ...
    'a_alpha_pdci', 'b_alpha_pdci', 'mu_beta_pdci', 'sigma_beta_pdci', ...
    'alpha_pdci', 'beta_pdci', ...
    'a_alpha_pdcd', 'b_alpha_pdcd', 'mu_beta_pdcd', 'sigma_beta_pdcd', ...
    'alpha_pdcd', 'beta_pdcd', ...
    };

%% RUN JAGS
fprintf('Running JAGS...\n');
tic 
[samples, stats, structArray] = matjags( ...
    datastruct, ...                     % Observed data
    fname_model, ...                    % File that contains model definition
    init0, ...
    'doparallel' , doparallel, ...      % Parallelization flag
    'nchains', nChains,...              % Number of MCMC chains
    'nburnin', nBurnin,...              % Number of burnin steps
    'nsamples', nSamples, ...           % Number of samples to extract
    'thin', thinning, ...               % Thinning parameter
    'monitorparams', monitorparams, ... % List of latent variables to monitor
    'savejagsoutput', 1 , ...           % Save command line output produced by JAGS?
    'verbosity', 2 , ...                % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup', 1 );                     % clean up of temporary files?
toc

