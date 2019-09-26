function [] = gen_pep_barplot(PEP, models, outpath)
% generate barplot for models protected exceedance proabilities
if nargin > 2
    ifSave = true;
else
    ifSave = false;
end

nModels = numel(models);
f = figure;
ax = axes;

bar(ax, PEP, 'FaceColor', [.5, .5, .5]);

xlims = [0.25, nModels+.75];
xlim(ax, xlims);
xticklabels(models);
xlabel('Models')
title('Protected Exceedance Probability')
line(xlims, [.95, .95], 'Color', 'r');
text(1:nModels, PEP, num2str(PEP', '%0.3f'), ...
    'vert', 'bottom', 'horiz', 'center')

box off
grid on

if ifSave
   saveas(f, outpath);
end
