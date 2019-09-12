clear; clc;

% data location
root = '/home/kmb/Desktop/Neuroscience/Projects/BONNA_decide_net/data/main_fmri_study/behavioral/';
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
xl = beh(:, :, :, strcmp(meta.dim4, 'magn_left'));  % reward magnitude for left box
xr = beh(:, :, :, strcmp(meta.dim4, 'magn_right')); % reward magnitude for right box
resp = beh(:, :, :, strcmp(meta.dim4, 'response')); % chosen side
rwd = beh(:, :, :, strcmp(meta.dim4, 'rwd'));       % rewarded / punished side (bci)
nSubjects = numel(meta.dim1);
nConditions = numel(meta.dim2);
nTrials = numel(meta.dim3);
resp(resp == 0) = NaN;                              %TODO: deal with missing values
resp = (resp + 1) / 2;                              % 0: left box; 1: right box; Inf: miss
rwd  = (rwd + 1) / 2;                               % 0: left box; 1: right box  

% JAGS Parameters
fname_model = fullfile(pwd, strcat(mfilename, '.txt'));
doparallel = 0;                                     % parallelization flag
thinning = 2;                                       % thinning parameter
nChains = 1;
nBurnin = 500;
nSamples = 2000;


% Initialize Markov chain values
for i=1:nChains
    S.alpha = 0.5 * ones(nSubjects, 1);
    S.beta = 1 * ones(nSubjects, 1);
    init0(i) = S;
end

% Assign MATLAB variables to the observed JAGS nodes
datastruct = struct(...
    'xl', xl, ...
    'xr', xr, ...
    'rwd', rwd, ...
    'resp', resp, ...
    'nSubjects', nSubjects, ...
    'nConditions', nConditions, ...
    'nTrials', nTrials);


%% RUN JAGS
fprintf('Running JAGS...\n');

[samples, stats, structArray] = matjags( ...
    datastruct, ...                     % Observed data
    fname_model, ...                    % File that contains model definition
    init0, ...
    'doparallel' , doparallel, ...      % Parallelization flag
    'nchains', nChains,...              % Number of MCMC chains
    'nburnin', nBurnin,...              % Number of burnin steps
    'nsamples', nSamples, ...           % Number of samples to extract
    'thin', thinning, ...               % Thinning parameter
    'monitorparams', ...
        {'alpha', 'alpha_d', 'beta',...
         'z', ...
         'a_alpha', 'b_alpha'}, ...     % List of latent variables to monitor
    'savejagsoutput', 0 , ...           % Save command line output produced by JAGS?
    'verbosity', 2 , ...                % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup', 1 );                     % clean up of temporary files?


