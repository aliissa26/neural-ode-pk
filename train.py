import torch
import math
import torch.nn.functional as F

LOG_TWO_PI = math.log(2 * math.pi)
from torch.optim import AdamW
from tqdm import trange


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    epochs: int = 200,
    kl_weight: float = 1e-2,
    clip: float = 1.0,  
    early_stop_patience: int = 20
):
    device = next(model.parameters()).device


   

    best_val = float("inf")
    no_improve = 0
    best_state = None
    tr_hist = []
    va_hist = []

    for epoch in trange(1, epochs + 1, desc="Epoch"):
        
        warmup_epochs = int(0.3 * epochs)
        β = kl_weight * min(1.0, epoch / warmup_epochs)

        # TRAINING
        model.train()
        running_loss = 0.0
        total_points = 0

       
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            μ_pred, logσ_pred, y_true = model(batch, conditioned=True)

          
            σ_floor = 1e-3
            σ_pred  = torch.exp(logσ_pred).clamp(min=σ_floor)
            logσ_pred = torch.log(σ_pred)

      
            mse_term  = ((y_true - μ_pred)**2) / (σ_pred**2 + 1e-6)
            nll       = 0.5 * (mse_term + 2.0 * logσ_pred + LOG_TWO_PI)
            nll_loss  = nll.mean()

            kl   = getattr(model, "kl_sum", torch.tensor(0.0, device=device))
           
            loss = nll_loss + β * (kl / len(batch))

            loss.backward()

         
            if batch_idx % 50 == 0:
              
                kl_term = (getattr(model, "kl_sum", torch.tensor(0.0, device=device)) / len(batch)).item()
                print(f"[TRAIN] Epoch {epoch}, Batch {batch_idx}: NLL={nll_loss.item():.4f}, KL={kl_term:.4f}, β={β:.5f}")

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip) 
            optimizer.step()

            batch_points = y_true.numel()
            running_loss += nll_loss.item() * batch_points
            total_points += batch_points

        train_epoch_loss = running_loss / total_points
        tr_hist.append(train_epoch_loss)

        # VALIDATION 
        model.eval()
        running_val = 0.0

        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                μ_pred, logσ_pred, y_true = model(batch, conditioned=True)
               
                σ_floor = 1e-3
                σ_pred  = torch.exp(logσ_pred).clamp(min=σ_floor)
                logσ_pred = torch.log(σ_pred)


                mse_term  = ((y_true - μ_pred)**2) / (σ_pred**2 + 1e-6)
                nll       = 0.5 * (mse_term + 2.0 * logσ_pred + LOG_TWO_PI)
                nll_loss  = nll.mean()

                running_val += nll_loss.item() * y_true.numel()
                total_val += y_true.numel()

        val_epoch_loss = running_val / total_val
        va_hist.append(val_epoch_loss)

       
        if scheduler is not None:
            scheduler.step(val_epoch_loss)
      

        if val_epoch_loss < best_val - 1e-8:
            best_val = val_epoch_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return tr_hist, va_hist