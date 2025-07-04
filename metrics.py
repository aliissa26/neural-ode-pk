import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, solve_triangular
from typing import Literal

def npde(
    y_obs: np.ndarray,
    sims:  np.ndarray,   
    ids:   np.ndarray,
    *,
    cov_jitter: float = 1e-8,           
    clip_jitter: float | None = None,   
    rank_method: Literal["classic", "mid"] = "classic",
) -> np.ndarray:
    
    y_obs = np.asarray(y_obs, float)
    sims  = np.asarray(sims,  float)
    ids   = np.asarray(ids,   int)

    N, R = y_obs.size, sims.shape[0]
    out  = np.full(N, np.nan, float)

    if clip_jitter is None:
        clip_jitter = 0.5 / R

    for sid in np.unique(ids):
        idx = np.where(ids == sid)[0]
        if idx.size < 2:
            continue  

        sims_sub = sims[:, idx]       
        y_sub    = y_obs[idx]        

       
        mu      = sims_sub.mean(axis=0)  
        Z_sims  = sims_sub - mu           
        Z_obs   = y_sub    - mu         

    
        cov = np.cov(Z_sims, rowvar=False)   
        cov += np.eye(cov.shape[0]) * cov_jitter
        L   = cholesky(cov, lower=True)      

        Z_sims = solve_triangular(L, Z_sims.T, lower=True).T
        Z_obs  = solve_triangular(L, Z_obs,    lower=True)

      
        for j, global_j in enumerate(idx):
            less = np.sum(Z_sims[:, j] < Z_obs[j])
            if rank_method == "classic":
                cdf = (less + 1) / (R + 1)
            else:  
                ties = np.sum(Z_sims[:, j] == Z_obs[j])
                cdf  = (less + 0.5 * ties) / R

            cdf = np.clip(cdf, clip_jitter, 1 - clip_jitter)
            out[global_j] = norm.ppf(cdf)

    return out