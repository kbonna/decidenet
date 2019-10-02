load('data/pc_factorial_run1_pi_only.mat', 'samples');
alpha_single_rew = samples.alpha_pi(:, :, :, 1);
alpha_single_pun = samples.alpha_pi(:, :, :, 2);
clear samples
load('data/pc_factorial_run1_pd_only.mat', 'samples');
alpha_plus_rew = samples.alpha_pd(:, :, :, 1, 1);
alpha_plus_pun = samples.alpha_pd(:, :, :, 2, 1);
alpha_minus_rew = samples.alpha_pd(:, :, :, 1, 2);
alpha_minus_pun = samples.alpha_pd(:, :, :, 2, 2);
clear samples
nSubjects = 32;

%% Draw learning rates for PD model (4 LR's)
nRow = 8;
nCol = 4;
f = figure('Position', [0 0 600 1200]);
[ha, pos] = tight_subplot(nRow, nCol, [0.04 0.04]);
set(gcf, 'Color', [1 1 1]);
set(gcf, 'InvertHardCopy', 'off');

edges = [0:0.01:1]; 

mle_plus_rew = zeros(1, nSubjects);
mle_plus_pun = zeros(1, nSubjects);
mle_minus_rew = zeros(1, nSubjects);
mle_minus_pun = zeros(1, nSubjects);

for i=1:nSubjects

            h_plus_rew = histogram(ha(i), reshape(alpha_plus_rew(:, :, i), [], 1), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'g', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .33);   
            set(ha(i), 'nextplot', 'add')
            h_plus_pun = histogram(ha(i), reshape(alpha_plus_pun(:, :, i), [], 1), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'g', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .66); 
            set(ha(i), 'nextplot', 'add')
            h_minus_rew = histogram(ha(i), reshape(alpha_minus_rew(:, :, i), [], 1), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'r', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .33);   
            set(ha(i), 'nextplot', 'add')
            h_minus_pun = histogram(ha(i), reshape(alpha_minus_pun(:, :, i), [], 1), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'r', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .66); 
            
            % formatting
            set(ha(i), 'YTickLabel', []) 
            set(ha(i), 'YLim', [0, 0.1])

            % grab MLE alpha values
            mle_plus_rew(i) = h_plus_rew.BinEdges(find(h_plus_rew.Values == max(h_plus_rew.Values), 1, 'first'));
            mle_plus_pun(i) = h_plus_pun.BinEdges(find(h_plus_pun.Values == max(h_plus_pun.Values), 1, 'first'));
            mle_minus_rew(i) = h_minus_rew.BinEdges(find(h_minus_rew.Values == max(h_minus_rew.Values), 1, 'first'));
            mle_minus_pun(i) = h_minus_pun.BinEdges(find(h_minus_pun.Values == max(h_minus_pun.Values), 1, 'first'));
          
end
saveas(gca, 'figures/posterior_alpha_pd_all.png');

%% Draw learning rates for PD model (4 LR's)
nRow = 8;
nCol = 4;
f = figure('Position', [0 0 600 1200]);
[ha, pos] = tight_subplot(nRow, nCol, [0.04 0.04]);
set(gcf, 'Color', [1 1 1]);
set(gcf, 'InvertHardCopy', 'off');

edges = [0:0.01:1]; 

mle_single_rew = zeros(1, nSubjects);
mle_single_pun = zeros(1, nSubjects);

for i=1:nSubjects

            h_single_rew = histogram(ha(i), reshape(alpha_single_rew(:, :, i), [], 1), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'b', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .33);   
            set(ha(i), 'nextplot', 'add')
            h_single_pun = histogram(ha(i), reshape(alpha_single_pun(:, :, i), [], 1), edges, ...
                'EdgeColor', 'none', ...
                'FaceColor', 'b', ...
                'Normalization', 'probability', ...
                'FaceAlpha', .66); 
            
            % formatting
            set(ha(i), 'YTickLabel', []) 
            set(ha(i), 'YLim', [0, 0.1])

            % grab MLE alpha values
            mle_single_rew(i) = h_single_rew.BinEdges(find(h_single_rew.Values == max(h_single_rew.Values), 1, 'first'));
            mle_single_pun(i) = h_single_pun.BinEdges(find(h_single_pun.Values == max(h_single_pun.Values), 1, 'first'));

end
saveas(gca, 'figures/posterior_alpha_pi_all.png');


%% show alphas in parameter space
f = figure('Position', [0 0 1200 300]);

subplot(1, 3, 1)

plot(mle_single_rew, mle_single_pun, 'ko', 'MarkerFaceColor', 'b')
hold on
plot(0:0.1:1, 0:0.1:1, 'k') 
set(gca, 'XLim', [0, 1], 'YLim', [0, 1])
xlabel('\alpha_{rew}');
ylabel('\alpha_{pun}');
title('PE invariant')
grid on

subplot(1, 3, 2)

plot(mle_plus_rew, mle_plus_pun, 'ko', 'MarkerFaceColor', 'g')
hold on
plot(0:0.1:1, 0:0.1:1, 'k') 
set(gca, 'XLim', [0, 1], 'YLim', [0, 1])
xlabel('\alpha_{rew}');
ylabel('\alpha_{pun}');
title('Positive PE')
grid on

subplot(1, 3, 3)

plot(mle_minus_rew, mle_minus_pun, 'ko', 'MarkerFaceColor', 'r')
hold on
plot(0:0.1:1, 0:0.1:1, 'k') 
set(gca, 'XLim', [0, 1], 'YLim', [0, 1])
xlabel('\alpha_{rew}');
ylabel('\alpha_{pun}');
title('Negative PE')
grid on

saveas(gca, 'figures\posterior_alpha_correlations.png')