clc; clear;

% Loading data
k1 = 5;
n1 = 10;
k2 = 5;
n2 = 10;

% jags parameters
nchains = 4;
nsamples = 1000;
nburnin = 100;
doparallel = 0;

% create object passed to matjags
datastruct = struct('k1', k1, 'n1', n1, 'k2', k2, 'n2', n2);

fprintf('Running jags...\n');

% initial values for latent parameters
for i=1:nchains
   S.theta = .5;
   init0(i) = S;
end

% run matjags
tic
[samples, stats, structArray] = matjags( ...
    datastruct, ...                     % Observed data
    fullfile(pwd, 'example3.txt'), ...  % File that contains model definition
    init0, ...                          % Initial values for latent vatiables
    'doparallel' , doparallel, ...      % Parallelization flag
    'nchains', nchains,...              % Number of MCMC chains
    'nburnin', nburnin,...              % Number of burnin steps
    'nsamples', nsamples, ...           % Number of samples to extract
    'thin', 1, ...                      % Thinning parameter
    'monitorparams', ...                % List of latent variables to monitor
        {'theta'}, ...    
    'savejagsoutput' , 0 , ...          % Save command line output produced by JAGS?
    'verbosity' , 2 , ...               % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup' , 0 );                    % clean up of temporary files?
toc