%% Load MCMC results and filter samples for model selection
if 0
    
load('condition_prediction_run1.mat', 'samples')
s1.z = samples.z;
s1.xi_alpha_pi = samples.xi_alpha_pi;
s1.xi_alpha_pd = samples.xi_alpha_pd; 
clear samples
load('condition_prediction_run1.mat', 'samples')
s0.z = samples.z;
s0.xi_alpha_pi = samples.xi_alpha_pi;
s0.xi_alpha_pd = samples.xi_alpha_pd; 
clear samples

% Additional variables
nModels = 2;
nSubjects = 32;

save('model_selection_samples')
end
load('model_selection_samples.mat')

%% Calculate Bayes factors for PI vd PD model comparison
% Posterior model probabilities
pmp = [];
for i = 1:nModels 
    pmp = [pmp, squeeze(mean(mean(s1.z==i, 1), 2))];    
end
pmp(pmp==0) = eps;
pmp(pmp==1) = 1-eps;

[posterior, out] = VBA_groupBMC(log(pmp)');
PEP = (1-out.bor)*out.ep + out.bor/length(out.ep)

% Bayes factors
bfactor_main = [pmp(:, 1) ./ pmp(:, 2)];

% Subject labels
for i=1:nSubjects
   ytl(i) = strcat("m", num2str(i+1, '%02i'));
end

gen_pep_barplot(PEP, {'PI', 'PD'}, 'figures/pep_pi_vs_pd.png')
gen_logbf_barplot(log10(bfactor_main), {'PI', 'PD'}, 'figures/logbf_pi_vs_pd.png')

%% draw pmp
figure('Renderer', 'painters', 'Position', [10 10 600 800])

subplot(1, 2, 1)
imagesc(pmp);
colormap(gca, flipud(bone));
colorbar;
title('Model indicator samples z_i')
xlabel('Model');
xticks([1, 2]);
xticklabels({'PI', 'PD'});
ylabel('Participants');
yticks(1:nSubjects)
yticklabels(ytl)

% draw bayes factors
subplot(1, 2, 2)
x = repmat(1:nModels, nSubjects, 1);                % generate x-coordinates
y = repmat(1:nSubjects, 1, nModels);                % generate y-coordinates
t = num2cell(bfactor);                              % extact values into cells
t = cellfun(@mynum2str, t, 'UniformOutput', false); % convert to string
imagesc(bfactor)
text(x(:), y(:), t, 'HorizontalAlignment', 'Center', 'Color', [.5,.5,.5]);
colormap(gca, flipud(bone));
colorbar;
xlabel('Model');
xticks([.5, 1.5, 2.5]);
xticklabels({'               PI vs PD', '               PD vs PI'});
yticks([1:nSubjects] - .5)
yticklabels([])
set(gca,'TickLength',[0, 0], 'GridAlpha', 1)
grid on
saveas(gcf, 'figures\pi_vs_pd.png')

%% Calculate and show Bayes factors for CI and CD model comparison
bf_thr = 10;
modelSelector = sum((bfactor > bf_thr) .* repmat(1:nModels, nSubjects, 1), 2);

% Draw subplots
nRow = 8;
nCol = 4;
figure('Position', [0 0 600 1200]);
[ha, pos] = tight_subplot(nRow, nCol, [0.04 0.04]);
set(gcf, 'Color', [1 1 1]);
set(gcf, 'InvertHardCopy', 'off');

edges = [-1:0.025:1]; 

for i=1:nSubjects

    z = s1.z(:, :, i);
    xi_pi = s1.xi_alpha_pi(:, :, i);
    xi_pd = s1.xi_alpha_pd(:, :, i, :);
    xi_pd_plus = s1.xi_alpha_pd(:, :, i, 1);
    xi_pd_minus = s1.xi_alpha_pd(:, :, i, 2);
    xi0 = s0.xi_alpha_pi(:, :, i);
    
    if modelSelector(i) == 1
            % prior
            prior = histogram(ha(i), xi0(z==1), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'k', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .33);   
            set(ha(i), 'nextplot', 'add')
            % posterior
            post = histogram(ha(i), xi_pi(z==1), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'b', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .66); 
            % bf annotations
            bf = post.Values(ceil(end/2)) / prior.Values(ceil(end/2));
            annotation(gcf, ... 
                'textbox', pos{i}, ...
                'String', num2str(bf, '%.02f'), ...
                'Color', 'b', 'LineStyle', 'None');

    elseif modelSelector(i) == 2
            % priors
            prior = histogram(ha(i), xi0(z==2), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'k', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .33);   
            set(ha(i), 'nextplot', 'add')
            % posterior
            post_plus = histogram(ha(i), xi_pd_plus(z==2), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'g', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .66); 
            set(ha(i), 'nextplot', 'add')
            post_minus = histogram(ha(i), xi_pd_minus(z==2), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'r', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .66); 
            % bf annotations
            bf = [post_plus.Values(ceil(end/2)) / prior.Values(ceil(end/2)), ...
                  post_minus.Values(ceil(end/2)) / prior.Values(ceil(end/2))];
            annotation(gcf, ... 
                'textbox', pos{i}, ...
                'String', num2str(bf(1), '%.02f'), ...
                'Color', 'g', 'LineStyle', 'None');    
            annotation(gcf, ... 
                'textbox', pos{i} + [0, -.02, 0, 0], ...
                'String', num2str(bf(2), '%.02f'), ...
                'Color', 'r', 'LineStyle', 'None');    

    else
            % prior
            prior = histogram(ha(i), xi0(:), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'k', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .33);   
            set(ha(i), 'nextplot', 'add')
            % posterior
            post = histogram(ha(i), xi_pd(:), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'k', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .66); 
            % bf annotations
            bf = post.Values(ceil(end/2)) / prior.Values(ceil(end/2));
            annotation(gcf, ... 
                'textbox', pos{i}, ...
                'String', num2str(bf, '%.02f'), ...
                'Color', 'k', 'LineStyle', 'None');
                   
    end
    
    % formatting
    set(ha(i), 'YTickLabel', []) 
    set(ha(i), 'YLim', [0, 0.15])
    % highliht subjects with "rejected H0"
    if any(bf < (1/bf_thr))
       set(ha(i), 'Color', 'y'); 
    end
    clear bf
    % subject labels
    annotation(gcf, ... 
    'textbox', pos{i} + [0, +0.023, 0, 0], ...
    'String', ytl{i}, ...
    'Color', 'k', 'LineStyle', 'None', 'FontSize', 8);    
    
end
annotation(gcf, ... 
    'textbox', [.47, .95, .05, .05], ...
    'String', '\xi\alpha', ...
    'LineStyle', 'None', 'FontSize', 12)
saveas(gcf, 'figures\ci_vs_cd.png')

