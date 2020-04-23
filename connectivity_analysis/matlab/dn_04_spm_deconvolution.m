path_root = getenv('DECIDENET_PATH');

path_out = fullfile(path_root, 'data/main_fmri_study/derivatives/ppi');
path_timeseries = fullfile(path_out, 'timeseries');

load(fullfile(path_timeseries, 'time_series_raw_all.mat'));
load(fullfile(path_timeseries, 'confounds_hmp_all.mat'));

%--- set parameters
RT = 2;         % repetition time (TR)
dt = 0.125;     % upsampling time interval (TR/16)
fMRI_T0 = 0;    % scanning onset
NT = RT/dt;     % upsampling rate (16 times)
N = 730;        % number of volumes
M = 25;         % number of confounds
k = 1:NT:N*NT;  % original volumes indices (in upsampled signal)

%--- select subject and task
sub = 1;
con = 1;
roi = 1;

%--- timeseries and get confounds
Y = squeeze(time_series_raw_all(sub, con, :, roi));
X0 = squeeze(confounds_hmp_all(sub, con, :, :));
X0 = [X0, ones(length(X0), 1)]; % adds constant term

%--- create and convolve cosine basis set
hrf = spm_hrf(dt);
xb  = spm_dctmtx(N*NT + 128,N);
Hxb = zeros(N,N);
for i = 1:N
    Hx       = conv(xb(:,i),hrf);
    Hxb(:,i) = Hx(k + 128);
end
xb = xb(129:end,:);

% Specify covariance components; assume neuronal response is white
% treating confounds as fixed effects
Q = speye(N,N)*N/trace(Hxb'*Hxb);
Q = blkdiag(Q, speye(M,M)*1e6  );
W = speye(N,N); % whitening matrix

% Create structure for spm_PEB
P{1}.X = [W*Hxb X0];        % Design matrix for lowest level
P{1}.C = speye(N,N)/4;      % i.i.d assumptions
P{2}.X = sparse(N + M, 1);   % Design matrix for parameters (0's)
P{2}.C = Q;

%%% Deconvolve
C  = spm_PEB(Y,P);
xn = xb*C{2}.E(1:N);
xn = spm_detrend(xn);

