clc; clear;

% Load data
n = 10;
k = 5;

% JAGS Parameters
nchains = 4;
nburnin = 0;
nsamples = 10000;

% Assign MATLAB variables to the observed JAGS nodes
datastruct = struct('n', n, 'k', k);

doparallel = 0; % do not use parallelization
fprintf('Running JAGS...\n');

% Initialize the values for each latent variable in each chain
for i=1:nchains
    S.theta = 0.5;
    init0(i) = S;
end

tic
[samples, stats, structArray] = matjags( ...
    datastruct, ...                     % Observed data
    fullfile(pwd, 'example1.txt'), ...  % File that contains model definition
    init0, ...                          % Initial values for latent vatiables
    'doparallel' , doparallel, ...      % Parallelization flag
    'nchains', nchains,...              % Number of MCMC chains
    'nburnin', nburnin,...              % Number of burnin steps
    'nsamples', nsamples, ...           % Number of samples to extract
    'thin', 1, ...                      % Thinning parameter
    'monitorparams', {'theta', 'thetaprior', 'priorpredk', 'postpredk'}, ...     
    'savejagsoutput' , 0 , ...          % Save command line output produced by JAGS?
    'verbosity' , 2 , ...               % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup' , 0 );                    % clean up of temporary files?
toc