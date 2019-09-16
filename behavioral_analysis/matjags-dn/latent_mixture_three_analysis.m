chain = 2;

pmp = [squeeze(mean(samples.z(chain,:,:)==1)), ...
       squeeze(mean(samples.z(chain,:,:)==2)), ...
       squeeze(mean(samples.z(chain,:,:)==3)), ...
       squeeze(mean(samples.z(chain,:,:)==4))];
   
imagesc(pmp)
colormap(flipud(bone))

sum(pmp==repmat(max(pmp')', [1, 4]))