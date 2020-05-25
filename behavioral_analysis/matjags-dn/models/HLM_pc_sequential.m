clear; clc;

%% Load and process experiment data
% data location
path_root = getenv('DECIDENET_PATH');
path_beh = fullfile(path_root, 'data/main_fmri_study/sourcedata/behavioral');
fname_beh = 'behavioral_data_clean_all.mat';
fname_meta = 'behavioral_data_clean_all.json';

% load behavioral and metadata
load(fullfile(path_beh, fname_beh));
fid = fopen(fullfile(path_beh, fname_meta)); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
meta = jsondecode(str);
clearvars -except beh meta

% split relevant data 
magn_l = beh(:, :, :, strcmp(meta.dim4, 'magn_left'));  % reward magnitude for left box
magn_r = beh(:, :, :, strcmp(meta.dim4, 'magn_right')); % reward magnitude for right box
resp = beh(:, :, :, strcmp(meta.dim4, 'response'));     % chosen side
side_bci = beh(:, :, :, strcmp(meta.dim4, 'side_bci')); % rewarded / punished side (bci)

nSubjects = numel(meta.dim1);
nConditions = numel(meta.dim2);
nTrials = numel(meta.dim3);
nPredErrSign = 2;
nModels = 4;                    

% Change coding from {-1, 0, 1} to {0, NaN, 1}
resp(resp == 0) = NaN;              % set missing values to NaNs
resp = (resp + 1) / 2;              % 0: left box; 1: right box; NaN: miss
side_bci  = (side_bci + 1) / 2;     % 0: left box; 1: right box (bci)  
side = side_bci;                                      
side(:, 2, :) = 1 - side(:, 2, :);  % convert from being chosen to favor interpretation

%% JAGS Setup
% Parameters
fname_model = fullfile(pwd, strcat(mfilename, '.txt'));
doparallel = 1;                     % parallelization flag
thinning = 1;                       % thinning parameter
nChains = 4;                                        
nBurnin = 1000;
nSamples = 5000;

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
    'magn_l', magn_l, ...
    'magn_r', magn_r, ...
    'side', side, ...
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

