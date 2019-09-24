%load('full4c_3m_run1.mat')

% Posterior model probabilities
pmp = [];
for i = 1:nModels 
    pmp = [pmp, squeeze(mean(mean(samples.z==i, 1), 2))];    
end
pmp(pmp==0) = eps;
pmp(pmp==1) = 1-eps;

% Bayes factors
bfactor = [];
for i = 1:nModels
    bfactor(:, :, i) = repmat(pmp(:, i), 1, 2) ./ pmp(:, (1:nModels ~= i));
end
bfactor(bfactor > 100) = 100;

for i=1:nSubjects
   ytl(i) = strcat("m", num2str(i+1, '%02i'));
end

%% PMP cute
imagesc(pmp);
colormap(flipud(bone));
colorbar;
title('Posterior model probabilities')

xlabel('Model');
xticks([1, 2]);
xticklabels({'PI', 'PD'});

for i=1:nSubjects
   ytl(i) = strcat("m", num2str(i+1, '%02i'));
end
ylabel('Participants');
yticks(1:nSubjects)
yticklabels(ytl)

%% BF for DLP
imagesc(bfactor(:,:,3));
colormap(flipud(hot));
title('Bayes Factors for DLP');

xlabel('Model');
xticks([1, 2]);
xticklabels({'vs SL','vs DLO'});

for i=1:nSubjects
   ytl(i) = strcat("m", num2str(i+1, '%02i'));
end
ylabel('Participants');
yticks(1:nSubjects)
yticklabels(ytl)

%% Alpha posteriors
Nrow = 8;
Ncol = 4;
cmap = colormap(flipud(bone));
figure('Position', [0 0 600 1200])
[ha, pos] = tight_subplot(8, 4, [0.04 0.04]);

i = 1;
for row=1:Nrow
    for col=1:Ncol
       
        z = samples.z(:, :, i);
        lr = samples.alpha_sl(:, :, i);
        
        if ~isempty(lr(z==1))
            histogram(ha(i), lr(z==1), ...
                      'EdgeColor', 'none', ...
                      'FaceColor', 'k')
        end
                  
        str = ytl(i);
        annotation(gcf, 'textbox', pos{i}, 'String', str);
        
        % Visual formatting
        yticklabels(ha(i), [])
        xticklabels(ha(i), [])
        
        i = i+1;
    end
end

%% show posteriors alpha DLO
Nrow = 8;
Ncol = 4;
cmap_rew = colormap(flipud(bone));
cmap_pun = colormap(flipud(pink));
figure('Position', [0 0 600 1200]);
[ha, pos] = tight_subplot(8, 4, [0.04 0.04]);

i = 1;
for row=1:Nrow
    for col=1:Ncol

        z = samples.z(:, :, i);
        lr_r = samples.alpha_dlo(:, :, i, 1);
        lr_p = samples.alpha_dlo(:, :, i, 2);
        
        if ~isempty(lr_r(z==2)) 
        histogram(ha(i), lr_r(z==2), ...
                  'EdgeColor', 'none', ...
                  'FaceColor', 'g', ...
                  'Normalization', 'probability')
        set(ha(i), 'nextplot', 'add')
        end
        
        if ~isempty(lr_p(z==2))
        histogram(ha(i), lr_p(z==2), ...
                  'EdgeColor', 'none', ...
                  'FaceAlpha', .4, ...
                  'FaceColor', 'r', ...
                  'Normalization', 'probability')
        end
        
        str = ytl(i);
        annotation(gcf, 'textbox', pos{i}, 'String', str);
        
        % Visual formatting
        yticklabels(ha(i), [])
        xticklabels(ha(i), [])
       
        i = i+1;
        
    end
end

%% show posteriors alpha DLP
Nrow = 8;
Ncol = 4;
cmap_rew = colormap(flipud(bone));
cmap_pun = colormap(flipud(pink));
figure('Position', [0 0 600 1200]);
[ha, pos] = tight_subplot(8, 4, [0.04 0.04]);

i = 1;
for row=1:Nrow
    for col=1:Ncol

        z = samples.z(:, :, i);
        lr_plus = samples.alpha_dlp(:, :, i, 1);
        lr_minu = samples.alpha_dlp(:, :, i, 2);
        
        if ~isempty(lr_plus(z==3)) 
        histogram(ha(i), lr_plus(z==3), ...
                  'EdgeColor', 'none', ...
                  'FaceColor', 'g', ...
                  'Normalization', 'probability')
        set(ha(i), 'nextplot', 'add')
        end
        
        if ~isempty(lr_minu(z==3))
        histogram(ha(i), lr_minu(z==3), ...
                  'EdgeColor', 'none', ...
                  'FaceAlpha', .4, ...
                  'FaceColor', 'r', ...
                  'Normalization', 'probability')
        end
        
        str = ytl(i);
        annotation(gcf, 'textbox', pos{i}, 'String', str);
        
        % Visual formatting
        yticklabels(ha(i), [])
        xticklabels(ha(i), [])
       
        i = i+1;
        
    end
end
