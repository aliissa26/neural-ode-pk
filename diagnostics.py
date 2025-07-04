import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import time
import scipy.stats as st
from tqdm import trange
from metrics import npde
from plotting import obs_pred, resid_plot, vpc, _save
import matplotlib.pyplot as plt
from tqdm import tqdm
from functorch import make_functional
from torch.func import functional_call             

def compute_pop_pred(model, loader, device):
   
    model.eval()
    real_encode = model._encode_z0

    def pop_encode(times_norm, obs_logdv):
        print(f"[DEBUG] pop_encode called; times_norm shape = {times_norm.shape}")
       
        if times_norm.numel() == 0:
            print("[DEBUG] → No obs: returning zero latent")
            return torch.zeros(model.latent_dim, device=device), torch.tensor(0.0, device=device)

        
        x_seq   = torch.stack([times_norm, obs_logdv], dim=-1).unsqueeze(1)
        _, h_n  = model.gru(x_seq)             
        h_last  = h_n.squeeze(0).squeeze(0)     
        mu_z    = model.fc_mu(h_last)           
        print(f"[DEBUG] → Computed mu_z mean = {mu_z.mean().item():.4f}")
        return mu_z, torch.tensor(0.0, device=device)

    model._encode_z0 = pop_encode
    all_logC = []
    for batch in loader:
        batch = [
            {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in subj.items()
            }
            for subj in batch
        ]
        with torch.no_grad():
            logC_pred_batch, _, _ = model(batch, conditioned=True)
        all_logC.append(logC_pred_batch)
    logC_pred = torch.cat(all_logC, dim=0)
    conc_pred = torch.exp(logC_pred * model.σ + model.μ)
    model._encode_z0 = real_encode
    return conc_pred.cpu().numpy()


def simulate_population(model, loader, device, n_rep, conditional=False):
  
    model.eval()

   
    state = {**dict(model.named_parameters()),
             **dict(model.named_buffers())}

    def fwd(batch, cond):
        
        return functional_call(model, state, (batch,), {'conditioned': cond})

   
    batch0 = next(iter(loader))
    batch0 = [{k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in s.items()}
              for s in batch0]

   
    if conditional:
        sims = []
        for _ in range(n_rep):
            with torch.no_grad():
                logC_pred_i, logσ_pred_i, logC_true_i = fwd(batch0, True)
              
                logσ_pred_i = logσ_pred_i.clamp(
                    min = math.log(1e-3),
                    max = math.log(2.0)
                )
                sd_i = torch.exp(logσ_pred_i)            
          
            zero_frac = (sd_i == 0).float().mean().item()
            if zero_frac > 0:
                print(f"[SIM] conditional sim sd==0 fraction: {zero_frac:.1%}")
            eps_i = torch.randn_like(sd_i) * sd_i
            sims.append((logC_pred_i + eps_i).cpu().numpy())
        sims_array = np.stack(sims, axis=0)
        print(f"[DEBUG simulate_population▷cond] sims_array shape={sims_array.shape}, "
              f"min={sims_array.min():.3f}, max={sims_array.max():.3f}")
        return np.stack(sims, axis=0)   


  
    preds, sds = [], []
    start_time = time.time()

  
    for _ in tqdm(range(n_rep), desc="simulate_population (unconditional)", unit="sim"):
       
        with torch.no_grad():
            logC_pred_i, logσ_pred_i, logC_true_i = fwd(batch0, False)
            logσ_pred_i = logσ_pred_i.clamp(
                min = math.log(1e-3),
                max = math.log(2.0)
            )
            sd_i = torch.exp(logσ_pred_i)
        preds.append(logC_pred_i.view(1, -1))
        sds.append(sd_i.view(1, -1))

  
    pred_rep = torch.cat(preds, 0)   
    sd_rep   = torch.cat(sds,   0)   
    eps      = torch.randn_like(sd_rep) * sd_rep
    total_time = time.time() - start_time

    print(f"    → simulate_population unc done: {n_rep} sims in {total_time:.1f}s")
    
    sims_unc = (pred_rep + eps).cpu().numpy()
    struct   = pred_rep.cpu().numpy()
    return sims_unc, struct




def collect(model, loader, device=torch.device("cpu")):
    """
    Run the model in full-encode mode and collect predicted vs observed
    for each subject’s entire profile.
    """
    Ys, Ps, Ss, IDs, Ts = [], [], [], [], []
    model.eval()

    # temporarily disable the reparameterization noise 
    real_encode = model._encode_z0
    def det_encode(times_norm, obs_logdv):
        """
        Deterministic posterior encoder: just return the mean (no eps)
        """
        device = obs_logdv.device
        if times_norm.numel() == 0:
            return torch.zeros(model.latent_dim, device=device), torch.tensor(0.0, device=device)
        x_seq = torch.stack([times_norm, obs_logdv], dim=-1).unsqueeze(1)
        _, h_last = model.gru(x_seq)
        h_last = h_last.squeeze(0).squeeze(0)
        mu_z = model.fc_mu(h_last)
        return mu_z, torch.tensor(0.0, device=device)

    model._encode_z0 = det_encode
    
    with torch.no_grad():
        for batch in loader:
            
            batch_dev = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in subj.items()}
                for subj in batch
            ]
            logC_pred, logσ_pred, logC_true = model(batch_dev, conditioned=True)
            p_np = logC_pred.cpu().numpy()
            y_np = logC_true.cpu().numpy()
            s_np = np.zeros_like(p_np)

           
            offset = 0
            for subj in batch:
                n = subj["obs_times"].shape[0]
                times = subj["obs_times"].cpu().numpy()

                Ps.append(p_np[offset : offset + n])
                Ys.append(y_np[offset : offset + n])
                Ss.append(s_np[offset : offset + n])
                IDs.extend([subj["id"]] * n)
                Ts.append(times)

                offset += n

    print(
        f"[DEBUG collect] collected {len(Ys)} points "
        f"from {len(np.unique(IDs))} subjects; times shape = {np.concatenate(Ts).shape}"
    )

    # restoring real encoder before returning
    model._encode_z0 = real_encode
    return (
        np.concatenate(Ys),
        np.concatenate(Ps),
        np.array(IDs, dtype=int),
        np.concatenate(Ts),
        np.concatenate(Ss),
    )




def run_full_diagnostics(
    model,
    loaders,
    names,
    n_sim: int = 1000,
    device: torch.device = torch.device("cpu"),
    lloq: float | None = None
):
   
    real_encode = model._encode_z0

    for loader, tag in zip(loaders, names):
        print(f"\n── Diagnostics: {tag} ──")

       
        
        y_obs, p_typ, ids, times, _ = collect(model, loader, device)
        
        print(f"[DEBUG GOF] subject count = {len(loader.dataset)}, "
              f"points y_obs={y_obs.shape}, p_typ={p_typ.shape}")
        C_pred = np.exp(p_typ * model.σ + model.μ)
        C_obs  = np.exp(y_obs  * model.σ + model.μ)
       
        print(f"[DEBUG GOF] C_pred min/max = {C_pred.min():.1f}/{C_pred.max():.1f}, "
              f"C_obs min/max = {C_obs.min():.1f}/{C_obs.max():.1f}")


        obs_pred(
            np.exp(p_typ*model.σ+model.μ),
            np.exp(y_obs*model.σ+model.μ),
            ids,
            tag=f"{tag}_INDIV",
            smooth=True,
            line=True,
            logxy=True               
        )

      
    

       
     
        full_loader = DataLoader(
            loader.dataset,
            batch_size=len(loader.dataset),
            shuffle=False,
            collate_fn=loader.collate_fn
        )

        
        print(f"  >> drawing {n_sim} unconditional sims (for VPC)…")
        t0 = time.time()
        sims_unc, _ = simulate_population(
            model, full_loader, device,
            n_rep=n_sim, conditional=False
        )
        print(f"[DEBUG run_full▷unc sims] sims_unc shape={sims_unc.shape}, "
              f"min={sims_unc.min():.3f}, max={sims_unc.max():.3f}")
       
        sims_unc = np.clip(sims_unc,
                           np.log(1e-3),  
                           np.log(1e5))   
        sims_unc = np.exp(sims_unc * model.σ + model.μ)

     
        pop_pred = compute_pop_pred(model, full_loader, device)
       


        dt = time.time() - t0
        print(f"  >> {n_sim} unconditional replicates in {dt/60:.1f} min")

      
        model._encode_z0 = real_encode
   

        held_out_loader = DataLoader(
            full_loader.dataset,
            batch_size=len(full_loader.dataset),
            shuffle=False,
            collate_fn=full_loader.collate_fn
        )

        
        real_enc = model._encode_z0
        def first_obs_encode(times_norm, obs_logdv):
            
            return real_enc(times_norm[:1], obs_logdv[:1])
        model._encode_z0 = first_obs_encode

        print(f"  >> drawing {n_sim} conditional sims (for NPDE)…")
        sims_pred = simulate_population(
            model, held_out_loader, device,
            n_rep=n_sim, conditional=True
        )

       
        model._encode_z0 = real_enc

        print(f"[DEBUG run_full▷cond sims_pred] shape={sims_pred.shape}, "
              f"min={sims_pred.min():.3f}, max={sims_pred.max():.3f}")



    
        y_pred, p_pred, ids_pred, times_pred, _ = collect(model, held_out_loader, device)
        print(f"[DEBUG run_full▷held-out collect] y_pred.shape={y_pred.shape}, "
              f"p_pred.shape={p_pred.shape}, ids_pred unique={len(np.unique(ids_pred))}, "
              f"times_pred.shape={times_pred.shape}")
        print(f"[DEBUG run_full▷align] sims_pred[1dim]={sims_pred.shape[1]} vs y_pred.len={y_pred.shape[0]}")

        print(f"  >> sims_pred.shape={sims_pred.shape}, y_pred.shape={y_pred.shape}")

    
        eps = 1e-8
     
        y_clipped    = y_pred
        sims_clipped = sims_pred

    
        C_obs  = np.exp(y_pred  * model.σ + model.μ)
        C_sims = np.exp(sims_pred * model.σ + model.μ)

       
        cov_jitter = 1e-6
        clip_jitter = 0.5 / C_sims.size
        while True:
            try:
                r_np = npde(
                    C_obs,
                    C_sims,
                    ids_pred,
                    cov_jitter=cov_jitter,
                    clip_jitter=clip_jitter,
                    rank_method='mid'
                )
                break
            except np.linalg.LinAlgError as e:
                print(f"[WARN] NPDE covariance not PD at jitter={cov_jitter:.1e}, bumping…")
                cov_jitter *= 10
                clip_jitter *= 2



        print(f"[DEBUG run_full▷NPDE] mean={np.nanmean(r_np):.3f}, std={np.nanstd(r_np, ddof=1):.3f}")

       
     


        resid_plot(
            p_pred,        
            y_pred,        
            r_np,          
            ids_pred,
            label="NPDE vs log(PRED)",
            resid_fn=lambda y, s, i: s,   
            tag=tag,
            smooth=True,
            line=True
        )

 
        r_np_clipped = np.clip(r_np, -4, 4)  
        plt.figure(figsize=(6, 5))
        for sid in np.unique(ids_pred):
            mask = ids_pred == sid
            if mask.sum() > 1:
                order = np.argsort(times_pred[mask])
                plt.plot(times_pred[mask][order], r_np[mask][order],
                         color="lightgrey", alpha=0.5, lw=1)
        plt.scatter(times_pred, r_np, alpha=0.6, zorder=3)
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(endog=r_np, exog=times_pred, return_sorted=True)
        plt.plot(sm[:, 0], sm[:, 1], color="red", lw=2)
        plt.xlabel("Time")
        plt.ylabel("NPDE")
        plt.title("NPDE vs Time")
        _save("npde_vs_time.png", tag=tag)

        plt.figure(figsize=(6, 4))
        mask = ~np.isnan(r_np)
        plt.hist(r_np[mask], bins=30, density=True, alpha=0.7)
        x = np.linspace(-4, 4, 200)
        plt.plot(x, st.norm.pdf(x), 'r--', label='N(0,1)')
        plt.title("NPDE Histogram")
        plt.legend()
        _save("npde_hist.png", tag=tag)

     
        plt.figure(figsize=(6, 6))
        st.probplot(r_np[mask], dist="norm", plot=plt)
        plt.title("NPDE QQ-Plot")
        _save("npde_qq.png", tag=tag)

       
        y_full_std, _, ids_full, times_full, _ = collect(model, full_loader, device)
        print(f"[DEBUG run_full▷full collect] y_full_std.len={y_full_std.shape[0]}, "
              f"ids_full unique={len(np.unique(ids_full))}, times_full.shape={times_full.shape}")
        print(f"[DEBUG run_full▷VPC align] pop_pred.len={pop_pred.shape[0]} vs times_full.len={len(times_full)}")
    
        y_full = np.exp(y_full_std * model.σ + model.μ)

        print(f"[DEBUG run_full▷VPC data] times_full={times_full.shape}, "
              f"sims_unc={sims_unc.shape}, pop_pred={pop_pred.shape}")
        


        print(f"[DEBUG VPC bins] time min/max = {times_full.min()}/{times_full.max()}, "
              f"y_full min/max = {y_full.min():.1f}/{y_full.max():.1f}")



      
        vpc(
            times_full,
            y_full,
            sims_unc,
            n_bins=6,
            ci=95,
            pc=False,
            tag=tag
        )

        
        model._encode_z0 = real_encode