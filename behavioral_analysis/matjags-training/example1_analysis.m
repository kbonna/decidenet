nchains = 4;

for i=1:nchains
    subplot(nchains, 1, i)
    histogram(samples.theta(i,:), 30)
    xlim([0, 1])
end

low = .4;
high = .6;

probab_low_high = sum(sum((samples.theta < high) & (samples.theta > low)))/numel(samples.theta);
