from scipy.stats import zscore
import numpy as np

def zscore_network(arr):
    '''Transforms adjacency matrix into new one applying zscore tranform to flat 
    vector of connection values. Diagonal remain unchanged.
    
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