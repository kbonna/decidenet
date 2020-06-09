%
% Calculate protected exceedance probabilities using VBA toolbox
%
% pmp: shape (nModels x nSubjects) z counts or point posterior estimate
% pep: shape (nModels)
%
%% Sequential HLM
path_root = getenv('DECIDENET_PATH');
path_vba = fullfile(path_root, 'data/main_fmri_study/derivatives/jags/vba');
load(fullfile(path_vba, 'pmp_HLM_sequential_split.mat'));

% PICI vs PICD vs PDCI vc PDCD
pmp(pmp == 0) = eps; % avoid NaNs in BOR measure
[~, out] = VBA_groupBMC(log(pmp));

% Calculate protected exceedance probability
pep = (1 - out.bor) * out.ep + out.bor / length(out.ep);

save(fullfile(path_vba, 'pep_HLM_sequential_split.mat'), 'pep');