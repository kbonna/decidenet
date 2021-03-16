from scipy.stats import zscore
import numpy as np

def zscore_network(arr):
    '''Trnaform adjacency matrix into new one applying z-score transform to 
    each individual element.'''
    new_arr = .5 * np.log((1 + arr) / (1 - arr))
    new_arr[np.diag_indices_from(new_arr)] = np.nan
    return new_arr
    
def standardize_network(arr):
    '''Transforms adjacency matrix into new one applying standarization to 
    flat vector of connection values. Diagonal remain unchanged.
    
    Note:
        Automatically detect if array is symmetric or not.
    '''
    n = arr.shape[0]
    
    is_symmetric = np.allclose(arr, arr.T, rtol=1e-05, atol=1e-08)
    new = np.zeros((n, n))
    
    if is_symmetric:
        v_triu = arr[np.triu_indices(n, k=1)]
        new[np.triu_indices(n, k=1)] = zscore(v_triu)
        new = new + new.T + np.diag(np.diag(arr))
    else:
        v_triu = arr[np.triu_indices(n, k=1)]
        v_tril = arr[np.tril_indices(n, k=-1)]
        v = zscore(np.concatenate((v_triu, v_tril)))
        v_triu, v_tril = np.split(v, 2)
        new[np.triu_indices(n, k=1)] = v_triu
        new[np.tril_indices(n, k=-1)] = v_tril
        new = new + np.diag(np.diag(arr))
        
    return new

def networks_mean(mat, net_names):
    '''Calculate mean values of matrix for between-networks blocks.
    
    Args:
        mat (array):
            Adjacency matrix representing any measure.
        net_names (Series):
            Vector of network labels.
            
    Returns:
        Tuple of two elements. First is list of large-scale networks. Second is
        lsn-adjacency matrix. Each row / column correspond to single lsn.
    '''
    n_rois = len(net_names)

    if mat.shape[0] != n_rois:
        raise ValueError(f'number of rois inferred from partition {n_rois} ' + \
                         f'is different shape of the adjacency matrix ' + \
                         f'{len(net_names)}')


    unique_nets = list(net_names.unique())
    n_unique_nets = len(unique_nets)

    mat_mean = np.zeros((n_unique_nets, n_unique_nets))

    for i, net_i in enumerate(unique_nets):
        for j, net_j in enumerate(unique_nets):
            mat_mean[i, j] = np.nanmean(
                mat[net_names == net_i, :][:, net_names == net_j]
            )

    return unique_nets, mat_mean