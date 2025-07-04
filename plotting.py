from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

_FIG_DIR = Path(__file__).parent / "figures"
_FIG_DIR.mkdir(exist_ok=True)

def _save(filename: str, tag: str | None = None):
    prefix = f"{tag}_" if tag else ""
    path = _FIG_DIR / f"{prefix}{filename}"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {path}")

def loss_curve(tr: list[float], va: list[float], logscale: bool = True):
    plt.figure(figsize=(7, 4))
    plt.plot(tr, "o-", label="Train")
    plt.plot(va, "o-", label="Val")
    if logscale:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood")
    plt.title("Training/Validation NLL")
    plt.legend()
    plt.tight_layout()
    _save("loss_curve.png")

def _gof(
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    title: str,
    ylabel: str,
    tag: str | None = None,
    logxy: bool = True,
    smooth: bool = True,
    line: bool = True,
    xlabel_override: str | None = None,   
):
    by_id = defaultdict(list)
    for xi, yi, sid in zip(x, y, ids):
        by_id[int(sid)].append((xi, yi))

    fig, ax = plt.subplots(figsize=(6, 5))

    if line:
        for pts in by_id.values():
            pts.sort(key=lambda p: p[0])
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="lightgrey", alpha=0.5, lw=1)

    ax.scatter(x, y, alpha=0.6, label="Data", zorder=3)
    if smooth:
        sm = lowess(endog=y, exog=x, return_sorted=True)
        ax.plot(sm[:,0], sm[:,1], color="red", lw=2, label="LOWESS")

    if logxy:
        
        if np.all(x > 0) and np.all(y > 0):
            ax.set_xscale("log")
            ax.set_yscale("log")
            
            lx = np.log10(x)
            ly = np.log10(y)
            r2 = np.corrcoef(lx, ly)[0,1]**2
            ax.text(0.05, 0.95, f"log-R²={r2:.2f}", transform=ax.transAxes,
                    va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        else:
            
            logxy = False
    else:
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() > 1:
            r2 = np.corrcoef(x[mask], y[mask])[0,1]**2
            ax.text(0.05, 0.95, f"R²={r2:.2f}", transform=ax.transAxes,
                    va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    if ylabel == "Observed" and not logxy:
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5)

   
    ax.set_xlabel(xlabel_override if xlabel_override is not None else "Predicted")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    _save(f"{title.lower().replace(' ', '_')}.png", tag=tag)

def obs_pred(p, obs, ids, tag=None, smooth=True, line=True, logxy=True):
   
    _gof(p, obs, ids, "Obs vs Pred", "Observed", tag=tag,
         logxy=logxy, smooth=smooth, line=line)

def resid_plot(p, y, sims, ids, label, resid_fn, tag=None, smooth=True, line=True):
    
    r = resid_fn(y, sims, ids)

    
    title = label
    ylabel, xlabel = label.split(" vs ", 1)

  
    _gof(
        p,
        r,
        ids,
        title,
        ylabel,
        tag=tag,
        logxy=False,
        smooth=smooth,
        line=line,
        xlabel_override=xlabel,  
    )

def vpc(time, obs, sim, *,
        n_bins=10,
        ci=95,
        lloq=None,
        pc=False,
        pred=None,
        tag=None):
    
    time = np.asarray(time, dtype=float)
    obs  = np.asarray(obs,  dtype=float)
    sim  = np.asarray(sim,  dtype=float)   
    R, N = sim.shape
    if pc:
        if pred is None:
            raise ValueError("pred must be supplied for pc-VPC")
        pred = np.asarray(pred, dtype=float)

        
        corr = np.clip(pred, 1e-8, np.inf)  
        sim  = sim  / corr[None, :]        
        obs  = obs  / corr                 

        if lloq is not None:
            
            lloq = lloq / np.median(corr)

   
    if lloq is not None:
        obs = np.maximum(obs, lloq)
        sim = np.maximum(sim, lloq)

    
    edges = np.quantile(time, np.linspace(0, 1, n_bins + 1))
    mids  = 0.5 * (edges[:-1] + edges[1:])

  
    q_obs  = {10: [], 50: [], 90: []}
    q_sim  = {10: [], 50: [], 90: []}
    ci_lo  = {10: [], 50: [], 90: []}
    ci_hi  = {10: [], 50: [], 90: []}
    lo_pct = (100 - ci) / 2
    hi_pct = 100 - lo_pct

    
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])): 
        if i == n_bins - 1:
            m = (time >= lo) & (time <= hi)
        else:
            m = (time >= lo) & (time <  hi)
        if m.sum() == 0:
            continue

        for p in (10, 50, 90): 
            q_obs[p].append(np.percentile(obs[m], p))

        bin_sim = sim[:, m]   
        for p in (10, 50, 90): 
            
            q_R = np.percentile(bin_sim, p, axis=1)   
            print(f"p={p}: min={q_R.min():.1f}, max={q_R.max():.1f}")
          
            q_sim[p].append(np.median(q_R))
            ci_lo[p].append(np.percentile(q_R, lo_pct))
            ci_hi[p].append(np.percentile(q_R, hi_pct))




   
    print(f"[DEBUG VPC] sim 10th   = {q_sim[10]}")
    print(f"[DEBUG VPC] sim median = {q_sim[50]}")
    print(f"[DEBUG VPC] sim 90th   = {q_sim[90]}")
    print(f"[DEBUG VPC] obs median = {q_obs[50]}")

   
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.set_yscale('log')

   
    plt.fill_between(
        mids,
        ci_lo[50],
        ci_hi[50],
        color='skyblue',
        alpha=0.4,
        label=f'{ci}% CI (median)'
    )

   
    plt.plot(
        mids,
        q_sim[50],
        color='blue',
        linewidth=2,
        label='Simulated median'
    )

   
    plt.plot(
        mids,
        q_sim[10],
        color='blue',
        linewidth=1,
        linestyle='--',
        label='Simulated 10th/90th'
    )
    plt.plot(
        mids,
        q_sim[90],
        color='blue',
        linewidth=1,
        linestyle='--'
    )

  
    if pc:
       
        plt.scatter(
            mids,
            q_obs[50],
            facecolors='none',
            edgecolors='black',
            s=50,
            label='Observed median'
        )
    else:
       
        plt.scatter(
            mids,
            q_obs[50],
            facecolors='none',
            edgecolors='black',
            s=50,
            label='Obs 50%'
        )
        plt.scatter(
            mids,
            q_obs[10],
            color='red',
            marker='v',
            zorder=3,
            label='Obs 10%'
        )
        plt.scatter(
            mids,
            q_obs[90],
            color='red',
            marker='^',
            zorder=3,
            label='Obs 90%'
        )

   
    if (not pc) and (lloq is not None):
        plt.axhline(
            lloq,
            linestyle='--',
            color='black',
            label='LLOQ'
        )

   
    start, end = edges[0], edges[-1]
    plt.xlabel('Time (h)')
    plt.ylabel('Prediction-corrected concentration' if pc else 'Concentration')
    plt.xlim(start, end)
    plt.xticks(np.arange(start, end + 1, 24))
    plt.title('PC-VPC' if pc else ' VPC')
    plt.legend(frameon=False)
    plt.grid(False)

  

    _save('vpc_pc.png' if pc else 'vpc.png', tag=tag)
