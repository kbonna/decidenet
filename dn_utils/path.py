import os
from os.path import join

# Top level
path_root = os.environ.get("DECIDENET_PATH")
path_bids = join(path_root, "data/main_fmri_study")
path_derivatives = join(path_bids, "derivatives")
path_sourcedata = join(path_bids, "sourcedata") 

# \--- sourcedata
path_behavioral = join(path_sourcedata, "behavioral")

# \--- derivatives
path_bsc = join(path_derivatives, "bsc")
path_data_paths = join(path_derivatives, "data_paths")
path_figures = join(path_derivatives, "figures")
path_fmridenoise = join(path_derivatives, "fmridenoise")
path_fmriprep = join(path_derivatives, "fmriprep")
path_jags = join(path_derivatives, "jags")
path_nistats = join(path_derivatives, "nistats")
path_parcellations = join(path_derivatives, "parcellations")
path_ppi = join(path_derivatives, "ppi")

path = {
    "root": path_root,
    "bids": path_bids,
    "derivatives": path_derivatives,
    "sourcedata": path_sourcedata,
    "behavioral": path_behavioral,
    "bsc": path_bsc,
    "data_paths": path_data_paths,
    "figures": path_figures,
    "fmridenoise": path_fmridenoise,
    "fmriprep": path_fmriprep,
    "jags": path_jags,
    "nistats": path_nistats,
    "parcellations": path_parcellations,
    "ppi": path_ppi
}