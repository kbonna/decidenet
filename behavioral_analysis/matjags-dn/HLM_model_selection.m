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
        xi_alpha_pd = samples.xi_alpha_pd(:, :, i, 2);
        
        if ~isempty(xi_alpha_pd(z==2)) 
        histogram(ha(i), xi_alpha_pd(z==2), ...
                  'EdgeColor', 'none', ...
                  'FaceColor', 'r', ...
                  'Normalization', 'probability')
        xlim(ha(i), [-1, 1]) 
        set(ha(i), 'nextplot', 'add')
        end
        
        str = ytl(i);
        annotation(gcf, 'textbox', pos{i}, 'String', str);
        
        % Visual formatting
        yticklabels(ha(i), [])
        xticklabels(ha(i), [])
       
        i = i+1;
        
    end
end
