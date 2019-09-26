nSubjects = 32;

%% BF x BF_single
close
x = log10(bfactor_single);
y = log10(bfactor_main);

filter = 1:nSubjects;
xf = x(filter(~(abs(y)>5 | abs(x)>5)));
yf = y(filter(~(abs(y)>5 | abs(x)>5)));
x = xf;
y = yf;

plot(x, y, 'bx', 'MarkerSize', 10)
ylabel('PD < log_{10}(BF) > PI')
xlabel('CD < log_{10}(BF_{single}) > CI')
line(get(gca, 'XLim'), [0, 0], 'Color', [.5, .5, .5]);
line([0, 0], get(gca, 'YLim'), 'Color', [.5, .5, .5]);
[rho, pval] = corr(x, y);
annotation('textbox', [.6, .6, .5, .3], ...
    'String', strcat('rho=', num2str(rho, '%0.2f'), ', p=', num2str(pval, '%0.2f')), ...
    'LineStyle', 'None');

saveas(gcf, 'figures/corr_bf_single_filtered.png')

%% BF x BF_dual+
close
x = log10(bfactor_dual(:, 1));
y = log10(bfactor_main);

filter = 1:nSubjects;
xf = x(filter(~(abs(y)>5 | abs(x)>5)));
yf = y(filter(~(abs(y)>5 | abs(x)>5)));
x = xf;
y = yf;

plot(x, y, 'gx','MarkerSize', 10)
ylabel('PD < log_{10}(BF) > PI')
xlabel('CD < log_{10}(BF_{dual+}) > CI')
line(get(gca, 'XLim'), [0, 0], 'Color', [.5, .5, .5]);
line([0, 0], get(gca, 'YLim'), 'Color', [.5, .5, .5]);
[rho, pval] = corr(x, y);
annotation('textbox', [.6, .6, .5, .3], ...
    'String', strcat('rho=', num2str(rho, '%0.2f'), ', p=', num2str(pval, '%0.2f')), ...
    'LineStyle', 'None');

saveas(gcf, 'figures/corr_bf_plus_filtered.png')

%% BF x BF_dual-
close
x = log10(bfactor_dual(:, 2));
y = log10(bfactor_main);

plot(x, y, 'rx', 'MarkerSize', 10)
ylabel('PD < log_{10}(BF) > PI')
xlabel('CD < log_{10}(BF_{dual-}) > CI')
line(get(gca, 'XLim'), [0, 0], 'Color', [.5, .5, .5]);
line([0, 0], get(gca, 'YLim'), 'Color', [.5, .5, .5]);
[rho, pval] = corr(x, y);
annotation('textbox', [.6, .6, .5, .3], ...
    'String', strcat('rho=', num2str(rho, '%0.2f'), ', p=', num2str(pval, '%0.2f')), ...
    'LineStyle', 'None');

saveas(gcf, 'figures/corr_bf_minus.png')