n = samples.n(1, :);
theta = samples.theta(1, :);

subplot(2, 1, 1);
scatter(n, theta);

% Draw from marginal distributions separately
for i=1:nsamples
    npostprdk = datasample(n, nsamples);
    thetapostprdk = datasample(theta, nsamples);
end

subplot(2, 1, 2);
scatter(npostprdk, thetapostprdk);