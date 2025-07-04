import torch 
import torch.nn as nn
from torchdiffeq import odeint


# Global clamp settings
RAW_SIGMA_CLAMP = dict(min=-15.0, max=5.0)
DEBUG_CLAMP     = False


class DoseEncoder(nn.Module):
    def __init__(self, hidden=32, sigma_phys=10.0):
        super().__init__()
        # learnable width
        self.log_sigma_phys = nn.Parameter(torch.log(torch.tensor(sigma_phys)))
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, t_abs, dose_times_abs, amts, ss, ii, span_phys):
        if dose_times_abs.numel() == 0:
            return torch.zeros_like(t_abs)

        dt_full = (t_abs.unsqueeze(-1) - dose_times_abs.unsqueeze(0)) / span_phys
        
        dt_full_masked = torch.where(dt_full >= 0.0, dt_full, torch.full_like(dt_full, float("inf")))
        dt_last = dt_full_masked.amin(dim=-1)

        if t_abs.dim() == 0:
            dt = (t_abs - dose_times_abs) / span_phys
            mask = (dt >= 0).float()
            ss_norm = ss / (span_phys + 1e-6)
            ii_norm = ii / (span_phys + 1e-6)
            feats = torch.stack([
                dt,
                torch.log1p(amts),
                dt_last.expand_as(dt),
                ss_norm,
                ii_norm
            ], dim=-1)
            sigma_phys = torch.exp(self.log_sigma_phys).to(dt.dtype)
            w = mask * torch.exp(-0.5*(dt/sigma_phys)**2)
            scores = self.net(feats).squeeze(-1)
            
            return (scores * w).sum()

       
        # normalize the same way as in the scalar branch
        dt = (t_abs.unsqueeze(-1) - dose_times_abs.unsqueeze(0)) / span_phys
        mask = (dt >= 0.0).float()

        
        dt_masked = torch.where(mask.bool(),
                                dt,
                                torch.full_like(dt, float("inf")))
        dt_last_vec = dt_masked.amin(dim=-1, keepdim=True)

       
        ss_norm = ss / (span_phys + 1e-6)
        ii_norm = ii / (span_phys + 1e-6)
        ss_feat = ss_norm.unsqueeze(0).expand_as(dt)
        ii_feat = ii_norm.unsqueeze(0).expand_as(dt)
        feats = torch.stack([
            dt,
            torch.log1p(amts.unsqueeze(0).expand_as(dt)),
            dt_last_vec,
            ss_feat,
            ii_feat
        ], dim=-1)
        sigma_phys = torch.exp(self.log_sigma_phys).to(dt.dtype)
        w = mask * torch.exp(-0.5 * (dt / sigma_phys) ** 2)
        scores = self.net(feats).squeeze(-1)
        
        return (scores * w).sum(dim=-1)


class ODEFunc(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim)
        )
       

    def forward(self, t_norm, z, u):
       
        t_feat = t_norm.unsqueeze(-1)
        xu = torch.cat([z, u.unsqueeze(-1), t_feat], dim=-1)
        return self.net(xu)


class RHS(nn.Module):
    def __init__(self, dose_enc, odefunc, times_phys, amts, ss, ii, t0_obs, tN_obs):
        super().__init__()
        
        self.register_buffer("times_abs", times_phys)
        self.register_buffer("amts", amts)
        self.register_buffer("ss", ss)
        self.register_buffer("ii", ii)
        self.t0 = t0_obs
        self.tN = tN_obs
        self.span_phys = max(float(tN_obs - t0_obs), 1e-3)
        self.dose_enc = dose_enc
        self.odefunc = odefunc

    def forward(self, t_abs, z):
       
        t_norm = (t_abs - self.t0) / self.span_phys
       
        u      = self.dose_enc(t_abs, self.times_abs, self.amts, self.ss, self.ii, self.span_phys)
        
        dz     = self.odefunc(t_norm, z, u)
        dz = torch.nan_to_num(dz, nan=0.0, posinf=0.0, neginf=0.0)
        return dz


class NeuralPK(nn.Module):
    def __init__(
        self,
        latent_dim=8,
        gru_hidden=64,
        aug_dim=0,
        μ_log=0.0,
        σ_log=1.0,
        ode_hidden=128,
        dose_hidden=32,
        dose_sigma=1e-3,
        tol=1e-6,
    ):
        super().__init__()
         
        self.latent_dim = latent_dim
        self.aug_dim = aug_dim
        self.μ, self.σ = μ_log, σ_log
        self.tol = tol
        

       
        self.gru = nn.GRU(input_size=2, hidden_size=gru_hidden)
        self.fc_mu = nn.Linear(gru_hidden, latent_dim)
        self.fc_logvar = nn.Linear(gru_hidden, latent_dim)

        
        self.log_sigma_prop_lin = nn.Parameter(torch.log(torch.tensor(0.3)))
        self.log_sigma_add_lin  = nn.Parameter(torch.log(torch.tensor(0.2)))

        # Dose encoder + ODE function
        self.dose_enc = DoseEncoder(dose_hidden, sigma_phys=dose_sigma)
        self.odefunc = ODEFunc(latent_dim + aug_dim, ode_hidden)

     
       
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # [μ_pred, raw_logσ]
        )

   
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)  
            if m.bias is not None:
                nn.init.zeros_(m.bias) 

    def _encode_z0(self, times_norm, obs_logdv):
        device = obs_logdv.device
        n_obs = times_norm.size(0)
        if n_obs == 0:
            z0 = torch.zeros(self.latent_dim, device=device)
            kl = torch.tensor(0.0, device=device)
            return z0, kl

        x_seq = torch.stack([times_norm, obs_logdv], dim=-1).unsqueeze(1)
        _, h_last = self.gru(x_seq)  
        h_last = h_last.squeeze(0).squeeze(0)  

        mu_z = self.fc_mu(h_last)
        logvar_z = self.fc_logvar(h_last)

       
        if torch.rand(1).item() < 0.01:
            print(f"[ENCODE] μ_z mean={mu_z.mean().item():.4f}, std={mu_z.std().item():.4f}; "
                  f"logvar mean={logvar_z.mean().item():.4f}")
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        z0_latent = mu_z + eps * std_z
        kl = 0.5 * torch.sum(mu_z ** 2 + torch.exp(logvar_z) - 1.0 - logvar_z)
        return z0_latent, kl

    def forward(self, batch, conditioned: bool = True):
        device = next(self.parameters()).device
        total_kl = torch.tensor(0.0, device=device)

        means = []
        logvars = []
        trues = []

        for subj in batch:
            t_phys = subj["obs_times"]  
            y_log = subj["obs_logdv"]  

          
            t_enc, y_enc   = t_phys,    y_log
            t_pred, y_hold = t_phys,    y_log


           
            if t_enc.numel() > 1:
                span_enc = (t_enc[-1] - t_enc[0]).clamp(min=1e-6)
                times_norm_enc = (t_enc - t_enc[0]) / span_enc
            else:
                times_norm_enc = torch.zeros_like(t_enc)
            if conditioned:
                z0_latent, kl = self._encode_z0(times_norm_enc, y_enc)
            else:
                z0_latent = torch.randn(self.latent_dim, device=device)
                kl = torch.tensor(0.0, device=device)

            total_kl = total_kl + kl
            z0 = torch.cat([z0_latent, torch.zeros(self.aug_dim, device=device)], dim=0)

            if t_pred.numel() > 0:
               
                t_pred, _ = torch.sort(t_pred)
                t_pred    = torch.unique_consecutive(t_pred).double()

               
                dose_ts = subj["dose_times"].to(device, dtype=torch.float64)
               
                z0 = z0.double()

                
                z0 = torch.nan_to_num(z0, nan=0.0, posinf=0.0, neginf=0.0)

              
                span   = (t_pred[-1] - t_pred[0]).clamp(min=1e-6)
                min_dt = max(torch.finfo(torch.float64).eps * span, 1e-6)

               
                while True:
                    gaps = torch.diff(t_pred)
                    mask = gaps < min_dt
                    if not mask.any():
                        break
                    t_pred[1:] = torch.where(mask, t_pred[1:] + min_dt, t_pred[1:])

               
                if t_pred.numel() == 1:
                    t_pred = torch.cat([t_pred, t_pred + min_dt])


               
                t_span = t_pred.clone()

                
                rhs_instance = RHS(
                    self.dose_enc,
                    self.odefunc,
                    subj["dose_times"].to(device).double(),
                    subj["dose_amts"].to(device).double(),         
                    subj["dose_ss"].to(device).double(),
                    subj["dose_ii"].to(device).double(),
                    t0_obs=t_phys[0].item(),
                    tN_obs=t_phys[-1].item()
                ).double()



               
                dose_ts_full = subj["dose_times"].to(device).double()
                amts_full    = subj["dose_amts"].to(device).double()
                mask = (dose_ts_full >= t_pred[0]) & (dose_ts_full <= t_pred[-1])
                dose_ts      = dose_ts_full[mask]
                amts         = amts_full[mask]

                
                z_ode = odeint(
                    rhs_instance,
                    z0.double(),
                    t_pred,
                    method="dopri5",
                    rtol=self.tol,
                    atol=self.tol * 0.01,
                    options={"first_step": float(min_dt),
                             "max_num_steps": int(t_pred.numel() * 50)}
                ).float()



                
               
                z_ode = z_ode.float()


           
                lat       = z_ode[:, :self.latent_dim]
               
               
                dec_out = self.decoder(lat)
                
                μ_pred   = dec_out[:, 0].clamp(min=-6.0, max=6.0)
                raw_logσ = dec_out[:, 1].clamp(**RAW_SIGMA_CLAMP)
                
                if DEBUG_CLAMP:
                    hits_low  = (raw_logσ <= RAW_SIGMA_CLAMP['min']).sum().item()
                    hits_high = (raw_logσ >= RAW_SIGMA_CLAMP['max']).sum().item()
                    if hits_low or hits_high:
                        print(
                            f"[CLAMP σ_raw @{RAW_SIGMA_CLAMP['min']},{RAW_SIGMA_CLAMP['max']}] "
                            f"low_hits={hits_low}, high_hits={hits_high}"
                        )
             

                σ_raw    = torch.exp(raw_logσ)
                # global proportional + additive
                σ_prop   = torch.exp(self.log_sigma_prop_lin).clamp(min=1e-3, max=5.0)
                σ_add    = torch.exp(self.log_sigma_add_lin).clamp(min=1e-3, max=5.0)
                # combine into total sigma
                σ_pred_lin = torch.sqrt((σ_raw * σ_prop)**2 + σ_add**2).clamp(min=1e-6, max=1e3)
                logσ_pred  = torch.log(σ_pred_lin)
             
                if torch.isnan(μ_pred).any() or torch.isinf(μ_pred).any():
                    print("🔍 [DEBUG] μ_pred has NaN/Inf:",
                          torch.isnan(μ_pred).any().item(),
                          torch.isinf(μ_pred).any().item())


            else:
                μ_pred    = torch.zeros(0, device=device)
                logσ_pred = torch.zeros(0, device=device)

            means.append(μ_pred)
            logvars.append(logσ_pred)
            trues.append(y_hold)

    
        self.kl_sum = total_kl
        μ_all     = torch.cat(means)
        logσ_all  = torch.cat(logvars)
        y_all     = torch.cat(trues)
        return μ_all, logσ_all, y_all
