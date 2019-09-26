function [] = gen_logbf_barplot(logbf, modelnames, outpath)

    if nargin > 2
        ifSave = true;
    else
        ifSave = false;
    end

    % Prepare data
    nSubjects = numel(logbf);
    for i=1:nSubjects
       ytl(i) = strcat("m", num2str(i+1, '%02i'));
    end

    f = figure('Renderer', 'painters', 'Position', [10 10 900 300]);
    ax = axes;
    colors = bone;

    hold on;
    % Plot actual bars
    for i=1:nSubjects
        h = bar(ax, i, logbf(i), 'LineStyle', 'None');
        if abs(logbf(i)) < 1
            set(h, 'FaceColor', colors(48, :))
        elseif abs(logbf(i)) < 2
            set(h, 'FaceColor', colors(16, :))
        else 
            set(h, 'FaceColor', colors(1, :))
        end
    end

    % Cosmetics
    xlim([.5, nSubjects+.5])
    xlabel('Subjects')
    xticks(1:nSubjects)
    set(ax,'XTickLabel', ytl, 'FontSize', 8, 'TickLength', [0, 0])
    xtickangle(90)
    ylim([-3, 3]);
    ylabel('log_{10}(BF)')
    box on;
    grid on;

    % Annotate modelnames
    annotation(f, 'textbox', [0.05 .66 .1 .1], 'String', modelnames{1}, ...
        'Color', 'k', 'LineStyle', 'None', 'FontSize', 12);
    annotation(f, 'textbox', [0.05 .33 .1 .1], 'String', modelnames{2}, ...
        'Color', 'k', 'LineStyle', 'None', 'FontSize', 12);
    
    if ifSave
       saveas(f, outpath); 
    end
    
end