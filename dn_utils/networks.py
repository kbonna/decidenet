import numpy as np
import pandas as pd

from scipy.stats import zscore
from statsmodels.stats.multitest import fdrcorrection

def fdrcorrection_matrix(arr, include_diagonal=True):
    """Apply FDR correction for matrix elements including diagonal entries.
    
    Args:
        arr (np.array):
            Matrix containing p-values.
        include_diagoonal (bool, optional):
            Whether diaganal elements should also be corrected. Defaults to 
            True.
            
    Returns:
        Matrix containing corrected p-values.    
    """
    n = arr.shape[0]
    k = 0 if include_diagonal else 1
    
    # Vectorize
    v_triu = arr[np.triu_indices(n, k=k)]

    # Restore 2D matrix
    new = np.zeros((n, n))
    new[np.triu_indices(n, k=k)] = fdrcorrection(v_triu)[1]
    new = new + np.tril(new.T, k=-1)
    
    return new


def zscore_matrix(arr):
    '''Transform adjacency matrix into new one applying z-score transform to 
    each individual element.'''
    new_arr = .5 * np.log((1 + arr) / (1 - arr))
    new_arr[np.diag_indices_from(new_arr)] = np.nan
    return new_arr
    
def zscore_vector(arr):
    """Transform 1D vector into new one calculating z-score for each element."""
    return (arr - np.mean(arr)) / np.std(arr)
    
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


def agreement_networks(m, nets, unique_nets):
    """Calculate agreement between set of a priori communities given network 
    division.
    
    Args:
        m (np.array):
            Community asignment vector.
        nets (np.array):
            A priori defined communities.
        unique_nets (np.array):
            Unique entries of nets. Resulting matrix rows and columns will 
            correspond to these.
    """
    return np.array(
        [
            [
                np.mean(m[nets == i] == m[nets == j][:, np.newaxis])
                - (1 / np.sum(nets == i)) * (i == j)
                for i in unique_nets
            ]
            for j in unique_nets
        ]
    )


def communities_overlap(m1, m2):
    """Calculate overlap matrix between two community vectors.
    
    Args:
        m1, m2 (pd.Series):
            Community vectors. Lenght should be equal to number of network 
            nodes. 
    """
    
    if len(m1) != len(m2): 
        raise ValueError(
            f"m1 and m2 should have equal length ({len(m1)} != {len(m2)}) ")
    if (m1.index != m2.index).any():
            f"m1 and m2 should have consistent index"

    m1_unique = m1.unique()
    m2_unique = m2.unique()

    df = pd.DataFrame(index=m1_unique, columns=m2_unique)
    for c in m2_unique:
        df[c] = m1[m2 == c].value_counts()        
    df = df.fillna(0)

    return df.astype(int)