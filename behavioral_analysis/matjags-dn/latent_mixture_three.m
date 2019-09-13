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
pz = [1/3, 1/3, .1/3];                             % prior for model selector                        
resp(resp == 0) = NaN;                              % set missing values to NaNs
resp = (resp + 1) / 2;                              % 0: left box; 1: right box; NaN: miss
rwd  = (rwd + 1) / 2;                               % 0: left box; 1: right box  

% JAGS Parameters
fname_model = fullfile(pwd, strcat(mfilename, '.txt'));
doparallel = 0;                                     % parallelization flag
thinning = 1;                                       % thinning parameter
nChains = 4;
nBurnin = 500;
nSamples = 2000;


% Initialize Markov chain values
for i=1:nChains
    S.a_alpha_sl = 2;
    S.b_alpha_sl = 2;
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
    'nTrials', nTrials, ...
    'pz', pz);


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
        {'z', 'pz', ...
        'alpha_sl', 'beta_sl', ...
        'alpha_dlo', 'beta_dlo', ...
        'alpha_dlp', 'beta_dlp'}, ...   % List of latent variables to monitor
    'savejagsoutput', 0 , ...           % Save command line output produced by JAGS?
    'verbosity', 2 , ...                % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup', 1 );                     % clean up of temporary files?


