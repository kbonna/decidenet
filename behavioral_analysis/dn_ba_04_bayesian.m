clc; clear;

% Load behavioral data
fname = ['/home/kmb/Desktop/Neuroscience/Projects/BONNA_decide_net/' ...
         'data/main_fmri_study/behavioral/behavioral_data_clean_all.mat'];
load(fname)

% Load data
%TODO: permute beh first, remove redundant calls
nTrials = size(beh, 3);
nSubjects = size(beh, 1);
nConditions = size(beh, 2);
magnl = permute(squeeze(beh(:, :, :, 3)), [2, 1, 3]);           % magnitude for left box
magnr = permute(squeeze(beh(:, :, :, 4)), [2, 1, 3]);           % magnitude for right box
rwd = permute((squeeze(beh(:, :, :, 2)) + 1)/2, [2, 1, 3]);     % rewarded side
choice = permute((squeeze(beh(:, :, :, 5)) + 1)/2, [2, 1, 3]);  % subject choice
choice(choice==.5) = nan;                                       % missing responses

% JAGS Parameters
nChains = 2;
nBurnin = 100;
nSamples = 1000;
doparallel = 0;                             % do not use parallelization

% Set initial parameters
for i=1:nChains
    S.alpha_mu = .5;
    S.alpha_std = .2;
    init0(i) = S;
end

% Assign MATLAB variables to the observed JAGS nodes
datastruct = struct( ...
    'rwd', rwd, ...
    'choice', choice, ...
    'magnl', magnl, ...
    'magnr', magnr, ...
    'nTrials', nTrials, ...
    'nSubjects', nSubjects, ...
    'nConditions', nConditions ...
);


fprintf('Running JAGS...\n');

tic
[samples, stats, structArray] = matjags( ...
    datastruct, ...                     % Observed data
    fullfile(pwd, 'hierarchicalFull.txt'), ...    % File that contains model definition
    init0, ...
    'doparallel' , doparallel, ...      % Parallelization flag
    'nchains', nChains,...              % Number of MCMC chains
    'nburnin', nBurnin,...              % Number of burnin steps
    'nsamples', nSamples, ...           % Number of samples to extract
    'thin', 1, ...                      % Thinning parameter
    'monitorparams', {
        'alpha', 'alpha_mu', 'alpha_std' 
        'beta', 'gamma', 'delta'
        }, ...                          % List of latent variables to monitor
    'savejagsoutput' , 0 , ...          % Save command line output produced by JAGS?
    'verbosity' , 2 , ...               % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup' , 0 );                    % clean up of temporary files?
toc

