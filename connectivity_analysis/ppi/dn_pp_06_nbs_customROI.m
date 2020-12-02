
design_matrix = cat(2, [ones(29, 1); zeros(29, 1)], [zeros(29, 1); ones(29, 1)], [eye(29); eye(29)]);
contrast = [1, 0, zeros(1, 29)];

sim = zeros(size(nbs_beta_mats, 3))
for i = 1 : size(nbs_beta_mats, 3)
    for j = 1 : size(nbs_beta_mats, 3)
        a = nbs_beta_mats(:, :, i);
        b = nbs_beta_mats(:, :, j);
        sim(i, j) = corr(a(:), b(:)); 
    end
end

imagesc(sim)