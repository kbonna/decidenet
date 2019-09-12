% correlation between responses and difference in reward magnitudes
mag_resp_corr = diag(corr(squeeze(resp(:, 1, :))', ... 
                          squeeze(xr(:, 1, :)-xl(:, 1, :))', 'rows', 'complete'));

%% show model posteriors
pmp = [squeeze(sum(samples.z == 1))/nSamples, squeeze(sum(samples.z == 2))/nSamples];
imagesc(pmp);
colormap(flipud(bone));
colorbar;
title('Posterior model probabilities')

xlabel('Model');
xticks([1, 2]);
xticklabels({'single lr','double lr'});

for i=1:nSubjects
   ytl(i) = strcat("m", num2str(i+1, '%02i'));
end
ylabel('Participants');
yticks(1:nSubjects)
yticklabels(ytl)

%% show posteriors on alpha
Nrow = 8;
Ncol = 4;
cmap = colormap(flipud(bone));
figure('Position', [0 0 600 1200])
[ha, pos] = tight_subplot(8, 4, [0.01 0.01]);

i = 1;
for row=1:Nrow
    for col=1:Ncol
       
        histogram(ha(i), samples.alpha(1, :, i), 'FaceColor', cmap(floor(pmp(i, 1)*64), :))
        
        str = ytl(i);
        annotation(gcf, 'textbox', pos{i}, 'String', str);
        
        % Visual formatting
        yticklabels(ha(i), [])
        xticklabels(ha(i), [])
        
        i = i+1;
        
    end
end

