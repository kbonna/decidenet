function [] = print_rhats(stats, varName)
% Prints number of rhat values exceeding threshold for a 

    rhat_thr = 1.1;
    rstats = eval(sprintf('stats.Rhat.%s', varName));
    exceeding = rstats(rstats > rhat_thr);

    % Logging format string
    format_string = strcat( ...
        '---> %s: %i [', ...
        repmat('%0.2f ', 1, numel(exceeding)), ...
        ']\n');
        
    if exceeding
        fprintf(format_string, varName, numel(exceeding), exceeding);
    else
        fprintf(format_string, varName, numel(exceeding));
    end
    
end