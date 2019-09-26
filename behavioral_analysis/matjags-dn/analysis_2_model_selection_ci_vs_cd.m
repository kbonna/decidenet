%% Load MCMC results and filter samples for model selection
if 0
    
load('condition_prediction_run1_pi_only.mat', 'samples')
spi.xi_alpha_pi = samples.xi_alpha_pi;
spi.z = samples.z;
clear samples

load('condition_prediction_run1_pd_only.mat', 'samples')
spd.xi_alpha_pd = samples.xi_alpha_pd;
spd.z = samples.z;
clear samples

load('condition_prediction_run1_null.mat', 'samples')
s0.z = samples.z;
s0.xi_alpha_pi = samples.xi_alpha_pi;
s0.xi_alpha_pd = samples.xi_alpha_pd; 
clear samples

% Additional variables
nModels = 2;
nSubjects = 32;

% Subject labels
for i=1:nSubjects
   ytl(i) = strcat("m", num2str(i+1, '%02i'));
end

end

%% Calculate and show Bayes factors for CI and CD model comparison
mId = 1; % MODEL ID

bf_thr = 10;
modelSelector = mId * ones(1, nSubjects);

% Draw subplots
nRow = 8;
nCol = 4;
f = figure('Position', [0 0 600 1200]);
[ha, pos] = tight_subplot(nRow, nCol, [0.04 0.04]);
set(gcf, 'Color', [1 1 1]);
set(gcf, 'InvertHardCopy', 'off');

edges = [-1:0.025:1]; 

for i=1:nSubjects

    if mId == 1
        xi_pi = spi.xi_alpha_pi(:, :, i);
        z = spi.z(:, :, i);
    else
        xi_pd = spd.xi_alpha_pd(:, :, i, :);
        xi_pd_plus = spd.xi_alpha_pd(:, :, i, 1);
        xi_pd_minus = spd.xi_alpha_pd(:, :, i, 2);
        z = spd.z(:, :, i);
    end
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
            pmp_zero(i, :) = [post.Values(ceil(end/2)), prior.Values(ceil(end/2))]; 
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
            pmp_dual(i, :, 1) = [post_plus.Values(ceil(end/2)), prior.Values(ceil(end/2))];
            pmp_dual(i, :, 2) = [post_minus.Values(ceil(end/2)), prior.Values(ceil(end/2))];
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
    % save bfs
    if mId == 1
        bfactor_single(i) = bf;
    else
        bfactor_dual(i, :) = bf;
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

% Save results
if mId == 1
    saveas(f, 'figures/posterior_xi_single.png');
    gen_logbf_barplot(log10(bfactor_single), {'CI_{0}', 'CD_{0}'}, ...
        'figures/logbf_ci_vs_cd_zero.png')
    % calculate posterior exceedance probability
    [~, out] = VBA_groupBMC(log(pmp_zero)');
    PEP = (1-out.bor)*out.ep + out.bor/length(out.ep);
    gen_pep_barplot(PEP, {'CI_{0}', 'CD_{0}'}, 'figures/pep_ci_vs_cd_zero.png')
else
    
    saveas(f, 'figures\posterior_xi_dual.png');
    gen_logbf_barplot(log10(bfactor_dual(:, 1)), {'CI_{+}', 'CD_{+}'}, ...
        'figures/logbf_ci_vs_cd_plus.png')
    gen_logbf_barplot(log10(bfactor_dual(:, 2)), {'CI_{-}', 'CD_{-}'}, ...
        'figures/logbf_ci_vs_cd_minus.png')
    % calculate posterior exceedance probabilities
    pmp_dual(pmp_dual==0) = eps;    
    [~, out] = VBA_groupBMC(log(pmp_dual(:, :, 1))');
    PEP = (1-out.bor)*out.ep + out.bor/length(out.ep);
    gen_pep_barplot(PEP, {'CI_{+}', 'CD_{+}'}, 'figures/pep_ci_vs_cd_plus.png');
    [~, out] = VBA_groupBMC(log(pmp_dual(:, :, 2))');
    PEP = (1-out.bor)*out.ep + out.bor/length(out.ep);
    gen_pep_barplot(PEP, {'CI_{-}', 'CD_{-}'}, 'figures/pep_ci_vs_cd_minus.png');
end

