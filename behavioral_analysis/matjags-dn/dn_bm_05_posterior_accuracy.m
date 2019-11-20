load('data/condition_prediction_run1_pi_only.mat', 'samples');
theta_pi = samples.theta_pi;
clear samples
load('data/condition_prediction_run1_pd_only.mat', 'samples', 'resp');
theta_pd = samples.theta_pd;
clear samples
nSubjects = 32;

%% Load behavioral data
% data location
root = '/home/kmb/Desktop/Neuroscience/Projects/BONNA_decide_net/data/main_fmri_study/sourcedata/behavioral/';
fname_beh = 'behavioral_data_clean_all.mat';
fname_meta = 'behavioral_data_clean_all.json';

% load behavioral and metadata
load(strcat(root, fname_beh));
fid = fopen(strcat(root, fname_meta)); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
meta = jsondecode(str);

%% Calculate predictions
theta_pi_mean = abs(squeeze(mean(mean(theta_pi, 1), 2)));
theta_pd_mean = abs(squeeze(mean(mean(theta_pd, 1), 2)));

err_pi = zeros(nSubjects, nSubjects, 2);
err_pd = zeros(nSubjects, nSubjects, 2);

for j = 1:2
    for i1=1:nSubjects
        for i2=1:nSubjects

        err_pi(i1, i2, j) = nansum(squeeze(abs(theta_pi_mean(i1, j, :) - resp(i2, j, :)))) / 110;  
        err_pd(i1, i2, j) = nansum(squeeze(abs(theta_pd_mean(i1, j, :) - resp(i2, j, :)))) / 110;
        resp_corr(i1, i2, j) = nansum(squeeze(abs(resp(i1, j, :) - resp(i2, j, :)))) / 110;

        end
    end
end

%% Plot prediction accuracy
f = figure('Position', [0, 0, 1000, 800]);

subplot(2, 2, 1)
imagesc(err_pi(:,:,1))
colorbar
colormap(flipud(hot))
caxis([0, .5])
title('PI reward condition')

subplot(2, 2, 2)
imagesc(err_pi(:,:,2))
colorbar
colormap(flipud(hot))
caxis([0, .5])
title('PI punishment condition')

subplot(2, 2, 3)
imagesc(err_pd(:,:,1))
colorbar
colormap(flipud(hot))
caxis([0, .5])
title('PD reward condition')

subplot(2, 2, 4)
imagesc(err_pd(:,:,2))
colorbar
colormap(flipud(hot))
caxis([0, .5])
title('PD punishment condition')

saveas(f, 'figures/accuracy.png')

%%
f = figure('Position', [0, 0, 1000, 400]);

subplot(1, 2, 1)
imagesc(resp_corr(:,:,1))
colorbar
colormap(jet)
caxis([0, 1])
title('reward condition')

subplot(1, 2, 2)
imagesc(resp_corr(:,:,2))
colorbar
colormap(jet)
caxis([0, 1])
title('punishment condition')

saveas(f, 'figures/response_similarity_original.png')

%% REORDER proof
resolution = 1.01;
while numel(unique(m2)) ~= 2
    [m2, ~] = community_louvain(resp_corr(:,:,2), resolution);
end
resp_corr2_reord = resp_corr(reorder_mod(resp_corr(:,:,2), m2), reorder_mod(resp_corr(:,:,2), m2), 2);

f = figure('Position', [0, 0, 500, 400]);

imagesc(resp_corr2_reord);
colorbar
colormap(jet)
caxis([0, 1])
title('response similarity')

saveas(f, 'figures/response_similarity_reordered.png')