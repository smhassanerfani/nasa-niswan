import os

# 1. Set Env Vars BEFORE other imports
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
# If you use Pandas/NumExpr, this is needed too:
os.environ["NUMEXPR_NUM_THREADS"] = "4" 

# GPU Config
# os.environ['CUDA_VISIBLE_DEVICES'] = "MIG-b5356651-0d8e-5cd1-bdf3-ccbb8b221031"
os.environ['CUDA_VISIBLE_DEVICES'] = "MIG-214cfb66-c8e5-57f2-b101-90f2cca83fad"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 2. Enforce PyTorch internal limit
torch.set_num_threads(4)

import csv
from tqdm import tqdm

from dataset import E33OMA
from utils import LatitudeWeightedMSE
from src.sfno import SphericalFourierNeuralOperator


def main():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Config
    MAX_ROLLOUT = 48
    IMAGE_SIZE = [90, 144] 
    PADDING = [0, 0]

    # Model
    model = SphericalFourierNeuralOperator(
        img_size=(90, 144),
        grid="equiangular",
        num_layers=4,
        scale_factor=6,
        embed_dim=16,
        residual_prediction=False, # DISABLED: We do manual residual
        pos_embed="spectral",
        use_mlp=True,
        normalization_layer="none",
        in_chans=6,   # Input: 1 BCB + 5 Weather
        out_chans=1   # Output: 1 Delta BCB
    ).to(device)

    # Dataset & Loader (Assuming E33OMA is imported)
    dataset_train = E33OMA(period='train', padding=PADDING, rollout_steps=MAX_ROLLOUT) 
    dataset_val = E33OMA(period='val', padding=PADDING, rollout_steps=MAX_ROLLOUT)

    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False)

    # Loss Functions
    criterion = LatitudeWeightedMSE(lat_res=90, device=device)

    # Training Schedule
    TEACHER_FORCING_EPOCHS = 40
    WARMUP_EPOCHS = 10
    COSINE_EPOCHS = 30
    PEAK_LR = 1e-4
    BASE_LR = 1e-5

    best_val_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=PEAK_LR)

    # Phase 1: Linear Warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=BASE_LR/PEAK_LR, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    # Phase 2: Cosine Annealing
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=COSINE_EPOCHS, eta_min=BASE_LR
    )
    # Phase 3: Constant
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=BASE_LR/PEAK_LR, total_iters=200 
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
        milestones=[WARMUP_EPOCHS, TEACHER_FORCING_EPOCHS]
    )

    # ==========================================
    # 3. Training Loop
    # ==========================================

    # TOTAL_EPOCHS = TEACHER_FORCING_EPOCHS + (MAX_ROLLOUT - 1)
    TOTAL_EPOCHS = TEACHER_FORCING_EPOCHS
    print(f"Starting training for {TOTAL_EPOCHS} epochs...")

    for epoch in range(TOTAL_EPOCHS):
        
        # --- Determine Phase & Rollout Steps ---
        if epoch < TEACHER_FORCING_EPOCHS:
            rollout_steps = 1
            phase_name = "Teacher Forcing"
        else:
            rollout_steps = (epoch - TEACHER_FORCING_EPOCHS) + 2
            rollout_steps = min(rollout_steps, MAX_ROLLOUT)
            phase_name = f"Curriculum ({rollout_steps} steps)"
        
        # --- Training Step ---
        model.train()
        train_loss_accum = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Ep [{epoch+1}/{TOTAL_EPOCHS}] {phase_name}", leave=True)
        
        for batch_idx, (inputs, targets) in enumerate(train_loop):
            inputs = inputs.to(device) # Shape: (B, Time, 6, H, W) -> 6 chans = [BCB, Weather...]
            targets = targets.to(device) # Shape: (B, Time, 1, H, W) -> Ground Truth BCB
            
            optimizer.zero_grad()
            loss = 0
            
            # Initialize State with T=0
            # We assume Channel 5 is BCB, Channels 0-4 are Weather
            current_bcb = inputs[:, 0, -1:] # (B, 1, H, W) - The BCB at T=0
            
            for k in range(rollout_steps):
                # 1. Prepare Input
                # We need the Weather at T=k (the current step driving the change)
                # inputs[:, k, :-1] extracts channels 0-4 (Weather)
                current_weather = inputs[:, k, :-1] 
                
                # Stack: (B, 1, H, W) + (B, 5, H, W) -> (B, 6, H, W)
                model_input = torch.cat([current_weather, current_bcb], dim=1)

                # 2. Forward Pass (Predict Delta)
                delta_bcb = model(model_input)
                
                # 3. Manual Residual Connection (Physics Step)
                # New State = Old State + Change
                pred_bcb = current_bcb + delta_bcb

                # 4. Calculate Loss
                # Compare Predicted State vs Ground Truth State at T=k
                gt_bcb = targets[:, k] # The actual BCB at this step
                loss += criterion(pred_bcb, gt_bcb)
                
                # 5. Update State for Next Step (Autoregressive)
                # Crucial: We must detach() if we don't want gradients to flow infinitely back
                # (Though for training we usually keep graph connected for BPTT, Backpropagation Through Time)
                # In curriculum learning, we usually feed the PREDICTION into the next step
                if k < rollout_steps - 1:
                    current_bcb = pred_bcb.detach()
            
            # Normalize Loss
            loss = loss / rollout_steps
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            train_loop.set_postfix(batch_loss=f"{loss.item():.4f}")

        # --- Step Scheduler ---
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # --- Validation Step ---
        model.eval()
        val_loss_accum = 0.0
        val_loop = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs = inputs.to(device)
                targets = targets.to(device)
                loss = 0
                
                # Reset State for Validation Batch
                current_bcb = inputs[:, 0, -1:]
                
                for k in range(rollout_steps):
                    current_weather = inputs[:, k, :-1]
                    model_input = torch.cat([current_weather, current_bcb], dim=1)

                    # delta_bcb = model(model_input)
                    delta_bcb = model(model_input)
                    
                    # Manual Residual Update
                    pred_bcb = current_bcb + delta_bcb
                    
                    gt_bcb = targets[:, k]
                    loss += criterion(pred_bcb, gt_bcb)

                    if k < rollout_steps - 1:
                        current_bcb = pred_bcb
                
                val_loss_accum += (loss / rollout_steps).item()
                
        avg_val_loss = val_loss_accum / len(val_loader)
        
        # --- Logging ---
        tqdm.write(f"--> Summary Ep {epoch+1}: LR: {current_lr:.6f} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        with open("./01-22-2026-sfno/training_log.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, rollout_steps, current_lr, avg_train_loss, avg_val_loss])
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./01-22-2026-sfno/best_model.pth")
            tqdm.write(f"    [Saved Best Model] Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()