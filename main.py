import torch
import random
import numpy as np
import torch.nn as nn

import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import train_test_split

from data         import load_data, PKDataset
from model        import NeuralPK
from train        import train
from plotting     import loss_curve
from diagnostics  import run_full_diagnostics

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  
        if m.bias is not None:
            nn.init.zeros_(m.bias) 
def main():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">▶ Device: {device}\n")

    train_dl, val_dl, test_dl, mu_log, sigma_log = load_data(
        csv_path   = "sim.dat.csv",
        drug_name  = "Bolus_2CPTMM_rich",
        device     = device,
        batch_size = 8
    )

    model = NeuralPK(
        latent_dim  = 3,
        aug_dim     = 0,
        gru_hidden  = 192,
        μ_log       = mu_log,
        σ_log       = sigma_log,
        ode_hidden  = 256,
        dose_hidden = 64,
        dose_sigma  = 30.0,
        tol         = 1e-6, 
    ).to(device)

   
    model.apply(init_weights)  

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, 'min',
        patience=5, factor=0.5, verbose=True
    )

    tr_hist, va_hist = train(
        model,
        train_dl,
        val_dl,
        optimizer           = optimizer,
        scheduler           = scheduler,   
        epochs              = 300,
        clip                = 1.0,
        kl_weight           = 1e-2,
        early_stop_patience = 30
    )

   
    prop_err = torch.exp(model.log_sigma_prop_lin).item()
    print(">>> learned σ_prop =", prop_err)

    add_err = torch.exp(model.log_sigma_add_lin).item()
    print(">>> learned σ_add  =", add_err)

    
    print(f"Initial sigma was: 30.0")
    print(">>> Learned σ_phys (h):", torch.exp(model.dose_enc.log_sigma_phys).item())

    loss_curve(tr_hist, va_hist)

    #  Build data loaders for diagnostics
    df = pd.read_csv("sim.dat.csv")
    df = df[df.dataset == "Bolus_2CPTMM_rich"].copy()
    df[["time","dv","evid","amt","ss","ii"]] = df[["time","dv","evid","amt","ss","ii"]].apply(pd.to_numeric)
    all_ids = df.id.unique()
    train_ids, tmp      = train_test_split(all_ids, train_size=0.6, random_state=seed)
    val_ids,   test_ids = train_test_split(tmp,      test_size=0.5, random_state=seed)

    train_full_ds = PKDataset(df, train_ids, mu_log, sigma_log, device)
    val_full_ds   = PKDataset(df, val_ids,   mu_log, sigma_log, device)
    tv_full_ds    = ConcatDataset([train_full_ds, val_full_ds])

    train_full_dl = DataLoader(
        train_full_ds,
        batch_size=train_dl.batch_size,
        shuffle=False,
        collate_fn=train_dl.collate_fn
    )
    tv_full_dl = DataLoader(
        tv_full_ds,
        batch_size=train_dl.batch_size,
        shuffle=False,
        collate_fn=train_dl.collate_fn
    )

    
    df_all = pd.read_csv("sim.dat.csv")
    df_all = df_all[df_all.dataset == "Bolus_2CPTMM_rich"]
    df_all[["time","dv","evid","amt","ss","ii"]] = df_all[["time","dv","evid","amt","ss","ii"]].apply(pd.to_numeric)
    counts = df_all[df_all.evid == 0].groupby("id").size()
    good_ids = counts[counts >= 5].index.values.astype(int)

    def filter_loader(loader):
        ds = loader.dataset
        if isinstance(ds, PKDataset):
            filtered_ids = [row["id"] for row in ds.rows if row["id"] in good_ids]
        else:
            filtered_ids = []
            for sub_ds in ds.datasets:
                filtered_ids += [row["id"] for row in sub_ds.rows if row["id"] in good_ids]
        new_ds = PKDataset(df_all, filtered_ids, mu_log, sigma_log, device)
        return DataLoader(
            new_ds,
            batch_size=loader.batch_size,
            shuffle=False,
            collate_fn=loader.collate_fn
        )

    train_full_dl = filter_loader(train_full_dl)
    tv_full_dl    = filter_loader(tv_full_dl)
    test_dl       = filter_loader(test_dl)

    run_full_diagnostics(
        model,
        loaders = [train_full_dl, tv_full_dl, test_dl],
        names   = ["TRAIN", "TRAIN+VAL", "TEST"],
        n_sim   = 200,      
        device  = device,
        lloq    = None
    )


if __name__ == "__main__":
    main()
