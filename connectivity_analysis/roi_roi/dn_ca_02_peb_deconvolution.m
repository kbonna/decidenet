path_root = getenv('DECIDENET_PATH');

path_out = fullfile(path_root, 'data/main_fmri_study/derivatives/ppi');
path_timeseries = fullfile(path_out, 'timeseries');

load(fullfile(path_timeseries, 'time_series_raw_all.mat'));
load(fullfile(path_timeseries, 'confounds_filtered_all.mat'));

%--- set parameters
RT = 2;         % repetition time (TR)
dt = 0.125;     % upsampling time interval (TR/16)
fMRI_T0 = 0;    % scanning onset
NT = RT/dt;     % upsampling rate (16 times)
N = 730;        % number of volumes
M = 33;         % number of confounds
k = 1:NT:N*NT;  % original volumes indices (in upsampled signal)
Nroi = 300;     % number of rois
Nsub = 32;      % number of subjects
Ncon = 2;       % number of conditions

%--- create and convolve cosine basis set
hrf = spm_hrf(dt);
xb  = spm_dctmtx(N*NT + 128,N);
Hxb = zeros(N,N);
for i = 1:N
    Hx       = conv(xb(:,i),hrf);
    Hxb(:,i) = Hx(k + 128);
end
xb = xb(129:end,:);

%--- specify covariance components 
% assume neuronal response is white
% treat confounds as fixed effects
Q = speye(N,N)*N / trace(Hxb'*Hxb);
Q = blkdiag(Q, speye(M,M)*1e6);

time_series_neural_all = zeros(Nsub, Ncon, N*NT, Nroi);

for con = 1 : Ncon
    for sub = 1 : Nsub
        fprintf('sub %i\n', sub)
        %--- get confounds
        X0 = [squeeze(confounds_filtered_all(sub, con, :, :)), ones(N, 1)];
        
        for roi = 1 : Nroi        
            %--- get timeseries
            Y = squeeze(time_series_raw_all(sub, con, :, roi));

            %--- create structure for spm_PEB
            P{1}.X = [Hxb X0];           % design matrix for lowest level
            P{1}.C = speye(N, N) / 4;    % i.i.d assumptions
            P{2}.X = sparse(N+M, 1);     % design matrix for parameters (0's)
            P{2}.C = Q;

            %--- deconvolve
            C  = spm_PEB(Y, P);
            xn = xb * C{2}.E(1:N);
            xn = spm_detrend(xn);

            %--- store result
            time_series_neural_all(sub, con, :, roi) = xn;
            
        end
    end
end

%--- save aggregated data
save(fullfile(path_timeseries, 'time_series_neural_hmp_csf_wm_all.mat'))

