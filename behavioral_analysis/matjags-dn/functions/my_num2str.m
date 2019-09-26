function str_formatted = my_num2str(num)
% Return formatted string from number, if it is in range 1/thr < num < thr
    thr = 100;
    if num >= thr
        str_formatted = strcat('>', num2str(thr));
    elseif num <= 1/thr
        str_formatted = strcat('<', num2str(1/thr));
    else
        str_formatted = num2str(num, '%.02f');
    end
end