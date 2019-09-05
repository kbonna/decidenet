clear; clc;

% Load data
k = [16, 18, 22, 25, 27];   % actual data
nmax = 500;                 % max n values
m = 5;                      % number of helpers

% jags parameters
nchains = 1;
nburnin = 100;
nsamples = 500;
nthin = 100;
doparallel = 0;

% create datastruct
datastruct = struct('k', k, 'nmax', nmax, 'm', m);

% set initial values for latent variables
for i=1:nchains
    S.theta = 0.5;
    S.n = 250;
    init0(i) = S;
end

% run matjags 
% run matjags
tic
[samples, stats, structArray] = matjags( ...
    datastruct, ...                     % Observed data
    fullfile(pwd, 'example4.txt'), ...  % File that contains model definition
    init0, ...                          % Initial values for latent vatiables
    'doparallel' , doparallel, ...      % Parallelization flag
    'nchains', nchains,...              % Number of MCMC chains
    'nburnin', nburnin,...              % Number of burnin steps
    'nsamples', nsamples, ...           % Number of samples to extract
    'thin', nthin, ...                  % Thinning parameter
    'monitorparams', ...                % List of latent variables to monitor
        {'n', 'theta'}, ...    
    'savejagsoutput' , 0 , ...          % Save command line output produced by JAGS?
    'verbosity' , 2 , ...               % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup' , 0 );                    % clean up of temporary files?
toc