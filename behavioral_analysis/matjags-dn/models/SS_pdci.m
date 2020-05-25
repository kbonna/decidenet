% Bayesian model used for parameter recovery simulation
clear; clc;

%% Setup
% task data location
root = '/home/kmb/Desktop/Neuroscience/Projects/BONNA_decide_net/data/main_fmri_study/sourcedata/behavioral/';
root_resp_artif = '/home/kmb/Desktop/Neuroscience/Projects/BONNA_decide_net/code/behavioral_analysis/matjags-dn/data/parameter_recovery_synthetic_data/';
fname_beh = 'behavioral_data_clean_all.mat';
fname_meta = 'behavioral_data_clean_all.json';
   
% load behavioral and metadata
load(strcat(root, fname_beh));
fid = fopen(strcat(root, fname_meta)); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
meta = jsondecode(str);
clearvars -except beh meta root_resp_artif

nSubjects = numel(meta.dim1);
nConditions = numel(meta.dim2);
nTrials = numel(meta.dim3);
nPredErrSign = 2;

% Parameters
fname_model = fullfile(pwd, strcat(mfilename, '.txt'));
doparallel = 1;         % parallelization flag
thinning = 1;           % thinning parameter
nChains = 4;                                        
nBurnin = 50;
nSamples = 250;

% Initialize Markov chain values
for i=1:nChains
    S = struct;         % random initial values
    init0(i) = S;
end

monitorparams = {'alpha_pdci', 'beta_pdci'};

%% Loop over parameter space
nTaskSamples = 5;   % number of different task realizations
bt = 3;             % fixed inverse-temperature

samples_ap = zeros(21, 21, nChains*nSamples, nTaskSamples);
samples_am = zeros(21, 21, nChains*nSamples, nTaskSamples);
samples_bt = zeros(21, 21, nChains*nSamples, nTaskSamples);

tic 
counter = -1;
for ap = 1 : 21
    for am = 1 : 21
        
        % Generate information for the user
        time = toc;
        counter = counter + 1;
        time_left_minutes = ((441/counter) - 1) * time / 60;
        fprintf('Progress: %.2f %% (%i/441 models estimated)\n', 100*counter/441, counter);
        fprintf('Estimated time to go: %ih %imin\n', ...
            floor(time_left_minutes/60), ... 
            floor(mod(time_left_minutes, 60)));

        % Draw random five task realizations
        subjects_idx = 1:32;
        subjects_idx = subjects_idx(randperm(length(subjects_idx)));
        subjects_idx = subjects_idx(1:nTaskSamples);

        for i = 1 : nTaskSamples 

            s = subjects_idx(i);

            % artificial response data location
            fname_resp_artif = strcat( ...
                root_resp_artif, ...
                strcat('response_synthetic_sub-', meta.dim1{s}, '.mat'));
            load(fname_resp_artif);

            % pool task data & synthetic response data
            xl = squeeze(beh(s, :, :, strcmp(meta.dim4, 'magn_left')));     % reward magnitude for left box
            xr = squeeze(beh(s, :, :, strcmp(meta.dim4, 'magn_right')));    % reward magnitude for right box
            side = squeeze(beh(s, :, :, strcmp(meta.dim4, 'side')));        % correct side
            resp = squeeze(response_synthetic(ap, am, bt, :, :));           % synthetic response

            % minor preprocessing
            resp(resp == 0) = NaN;              % set missing values to NaNs
            resp = (resp + 1) / 2;              % 0: left box; 1: right box; NaN: miss
            side = (side + 1) / 2;              % 0: left box; 1: right box (correct)

            % Assign MATLAB variables to the observed JAGS nodes
            datastruct = struct(...
                'xl', xl, ...
                'xr', xr, ...
                'side', side, ...
                'resp', resp, ...
                'nConditions', nConditions, ...
                'nPredErrSign', nPredErrSign, ...
                'nTrials', nTrials);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % RUN JAGS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                'savejagsoutput', 0 , ...           % Save command line output produced by JAGS?
                'verbosity', 0 , ...                % 0=do not produce any output; 1=minimal text output; 2=maximum text output
                'cleanup', 1 );                     % clean up of temporary files?

            % Write outputs
            samples_ap(ap, am, :, i) = reshape(samples.alpha_pdci(:,:,1), 1, []);
            samples_am(ap, am, :, i) = reshape(samples.alpha_pdci(:,:,2), 1, []);
            samples_bt(ap, am, :, i) = reshape(samples.beta_pdci, 1, []);
        end
        
    end
end