% Load variables 
if ~exist('stats')
    load('condition_prediction_run1.mat')
end

%% Show R-hat values for model indicator variable
close

bardata = stats.Rhat.z;

f = figure;
ax = axes;
hold on
for i=1:nSubjects
    h = bar(ax, i, bardata(i));
    if bardata(i) > 1.1
        set(h, 'FaceColor', 'r')
    else
        set(h, 'FaceColor', 'k')
    end
end

xlim(ax, [0, nSubjects+1]);
if sum(stats.Rhat.z) == 0
    yupper = 1.2;
else
    yupper = max(stats.Rhat.z)*1.1;
end
ylim(ax, [1, yupper])
line(get(ax, 'XLim'), [1.1, 1.1], 'Color', [1, 0, 0, .5], 'LineWidth', 1);
xticklabels(ax, [])
xlabel(ax, 'Subjects')
ylabel(ax, '$\hat{R}\quad$','Interpreter', 'latex', 'Rotation', 0)

title('MCMC convergence for model indicator variable');
grid on;
box on;
saveas(f, 'figures\rhat_z.png')

%% Report convergence for other variables
fprintf('Number of variables exceeding convergence threshold:\n')

varNames = fieldnames(stats.Rhat);
for i=1:length(varNames)
    log_rhat(stats, varNames{i})
end



    




