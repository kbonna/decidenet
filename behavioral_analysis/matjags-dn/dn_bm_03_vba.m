%%
%
% Calculate protected exceedance probabilities using VBA toolbox
%
% pmp_modelname: (nModels x nSubjects) z counts or point posterior estimate
% pep_modelname: (nModels)
%
%% Factorial HLM
load('data/pmp_factorial.mat');

% PI vs PD
pmp_main(pmp_main==0) = eps;
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

save('data/pep_factorial.mat', 'pep_main', 'pep_single', 'pep_plus', 'pep_minu')

%% Sequential HLM
load('data/pmp_seq.mat')

% PICI vs PICD vs PDCI vc PDCD
pmp_seq(pmp_seq==0) = eps; % avoid NaNs in BOR measure
[~, out] = VBA_groupBMC(log(pmp_seq));
pep_seq = (1-out.bor)*out.ep + out.bor/length(out.ep);

save('data/pep_seq.mat', 'pep_seq');