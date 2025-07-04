import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

class PKDataset(Dataset):

    def __init__(self, df, ids, mu_log, sigma_log, device):
        self.rows = []
        for sid in ids:
            sub = df[df.id == sid].sort_values("time")
            dose = sub[sub.evid == 1][["time", "amt", "ss", "ii"]].to_numpy()

            obs = sub[(sub.evid == 0) & (sub.time >= 0)][["time", "dv"]].to_numpy()

            
            times = obs[:, 0].astype(float)
            dv    = np.log(np.clip(obs[:, 1].astype(float), 1e-3, None))
            dv_std = (dv - mu_log) / sigma_log

            self.rows.append({
                "id":          int(sid),
                "dose_times":  torch.tensor(dose[:, 0], dtype=torch.float32, device=device),
                "dose_amts":   torch.tensor(dose[:, 1], dtype=torch.float32, device=device),
                "dose_ss":     torch.tensor(dose[:, 2], dtype=torch.float32, device=device),
                "dose_ii":     torch.tensor(dose[:, 3], dtype=torch.float32, device=device),
                "obs_times":   torch.tensor(times, dtype=torch.float32, device=device),
                "obs_logdv":   torch.tensor(dv_std, dtype=torch.float32, device=device),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        subj = self.rows[idx].copy()

        return subj

def make_loader(id_list, df, mu_log, sigma_log, device, batch_size, shuffle=False):
    ds = PKDataset(df, id_list, mu_log, sigma_log, device)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: b
    )
def load_data(
    csv_path: str,
    drug_name: str,
    device="cpu",
    batch_size: int = 16
):
    df = pd.read_csv(csv_path)
    df = df[df.dataset == drug_name].copy()
    df[["time", "dv", "evid", "amt"]] = df[["time", "dv", "evid", "amt"]].apply(pd.to_numeric)
    
    df[["ss", "ii"]] = df[["ss", "ii"]].apply(pd.to_numeric)

    all_ids = df.id.unique()
    train_ids, tmp = train_test_split(all_ids, train_size=0.6, random_state=1234)
    val_ids,   test_ids = train_test_split(tmp, test_size=0.5, random_state=1234)

    
    train_obs = df[
        (df.id.isin(train_ids))
        & (df.evid == 0)
        & (df.time  > 0)
    ].dv.clip(lower=1e-3)
    mu_log    = np.log(train_obs).mean()
    sigma_log = np.log(train_obs).std()

    train_dl = make_loader(train_ids, df, mu_log, sigma_log, device, batch_size, shuffle=True)
    val_dl   = make_loader(val_ids,   df, mu_log, sigma_log, device, batch_size)


    test_dl  = make_loader(test_ids,  df, mu_log, sigma_log, device, batch_size)  

    return train_dl, val_dl, test_dl, mu_log, sigma_log

if __name__ == "__main__":
    dl, _, _, mu, sd = load_data("sim.dat.csv", "Bolus_2CPTMM_rich", device="cpu", batch_size=4)
    b = next(iter(dl))
    print("Batch size:", len(b), "mu_log:", mu, "sd_log:", sd)