import matplotlib.pyplot as plt
import numpy as np

def report_convergence(mat: dict) -> None:
    """Check for convergence of MCMC chains.

    Args:
        mat: loaded MATLAB structure containing stats field (JAGS output)
    """
    n_monit_nodes = 0    # total number of monitored nodes
    n_failed_nodes = 0   # number of nodes that failed to converge
    
    print('Looking for failed convergence...')
    for node in mat['stats']['Rhat'].item().dtype.names:
        
        rhats = mat['stats']['Rhat'].item()[node].item()
        rhats = np.array(rhats)
        ind_failed = rhats > 1.1 
        
        if np.any(ind_failed):
            n_failed_nodes += np.sum(ind_failed)
            message = (
                f"[--] node(s) {node} failed to converge\n"
                f"{5*' '}– {np.sum(ind_failed)}/{np.size(rhats)} fails, "
                f"max = {np.max(rhats[ind_failed]):0.2f}, "
                f"median = {np.median(rhats[ind_failed]):0.2f}"
            )
            print(message)
        else:
            print(f'[OK] node(s) {node} converged')
        n_monit_nodes += np.size(rhats)
        
    print('\nSummary:')
    print(f'{100*(1 - (n_failed_nodes/n_monit_nodes)):0.2f}% nodes converged\n\n')
    
