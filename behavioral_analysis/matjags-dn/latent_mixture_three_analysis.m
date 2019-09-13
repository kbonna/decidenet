chain = 1;

pmp = [squeeze(mean(samples.z(chain,:,:)==1)), ...
       squeeze(mean(samples.z(chain,:,:)==2))]%, ...
       %squeeze(mean(samples.z(chain,:,:)==3))];
   
imagesc(pmp)
colormap(flipud(bone))
