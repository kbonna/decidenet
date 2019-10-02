%
% Calculate protected exceedance probabilities using VBA toolbox
%
load('data/pmp_all.mat')

% PI vs PD
[~, out] = VBA_groupBMC(log(pmp_main));
pep_main = (1-out.bor)*out.ep + out.bor/length(out.ep);

% CI vs CD (assumed PI)
[~, out] = VBA_groupBMC(log(pmp_single));
pep_single = (1-out.bor)*out.ep + out.bor/length(out.ep);

% CI vs CD (assumed PD, positive PE)
[~, out] = VBA_groupBMC(log(pmp_plus));
pep_plus = (1-out.bor)*out.ep + out.bor/length(out.ep);

% CI vs CD (assumed PI, negative PE)
[~, out] = VBA_groupBMC(log(pmp_minu));
pep_minu = (1-out.bor)*out.ep + out.bor/length(out.ep);

save('data/pep_all.mat', 'pep_main', 'pep_single', 'pep_plus', 'pep_minu');