# -*- coding: utf-8 -*-
"""
PyTorch training script for AE-based anomaly detection in sounds.

Migrated from 01_train.py (Chainer → PyTorch).

Reference:
  Y. Koizumi, et al., "ToyADMOS: A Dataset of Miniature-Machine Operating
  Sounds for Anomalous Sound Detection," WASPAA 2019.
"""
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import scipy.signal.windows
from tqdm import tqdm

# ── Ensure parent directory is on sys.path ────────────────────────────────────
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.join(_parent_dir, 'Modules'))

# ── Import our PyTorch modules ───────────────────────────────────────────────
from torch_version.config import load_config
from torch_version.model import FCN_AE
from torch_version.modules import wav_read_trn, melFilterBank, list_to_device
from torch_version.gpu_funcs import exe_fft, feature_extraction, loss_MMSE, concat_2_wavs


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Load params
sp_param, dnn_param, training_param = load_config()

# Toy type (must match the dataset you created)
toy_type = 'ToyConveyor'          # 'ToyCar' or 'ToyConveyor' or 'ToyTrain'

# Paths (resolved relative to the E01_simple_AE_test directory)
obs_dir = os.path.join(_parent_dir, f'exp1_dataset_{toy_type}', 'train_normal')
dnn_dir = os.path.join(_parent_dir, 'dnn_dir')
os.makedirs(dnn_dir, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(device)}')

# Debug mode
debug_mode = False
if debug_mode:
    import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Prepare mel filter bank & window (on device)
# ══════════════════════════════════════════════════════════════════════════════

# Hann window
win = torch.from_numpy(
    scipy.signal.windows.hann(sp_param["fftl"]).astype(np.float32)
).unsqueeze(1).to(device)                                           # (fftl, 1)
sp_param["win"] = win

# Mel filter bank
BPF_np, _ = melFilterBank(sp_param["fs"], sp_param["fftl"], dnn_param["NumFB"])
BPF = torch.from_numpy(BPF_np.astype(np.float32)).to(device)        # (NumFB, fftl//2+1)

# Load training data
dev_set = wav_read_trn(obs_dir, training_param["set_size"], sp_param["wav_read_gain"])


# ══════════════════════════════════════════════════════════════════════════════
# Model & Optimizer
# ══════════════════════════════════════════════════════════════════════════════

model = FCN_AE(
    in_dim=dnn_param["NumFB"],
    hid_dim=dnn_param["hid_dim"],
    z_dim=dnn_param["z_dim"],
    num_hid=dnn_param["num_hid"],
    num_fw=dnn_param["num_fw"],
    num_bw=dnn_param["num_bw"],
).to(device)

# Adam with amsgrad + weight decay (matches original Chainer setup)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=training_param["lr_base"],
    betas=(0.9, 0.999),
    amsgrad=True,
    weight_decay=training_param["l2_weight"],
)

# Learning rate scheduler (manual linear decay, see loop below)


# ══════════════════════════════════════════════════════════════════════════════
# Sub-modules
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_wav(x: torch.Tensor):
    """
    Forward pass on a waveform chunk.
    
    Args:
        x: 1D waveform tensor on device, shape (L,)
    
    Returns:
        loss:  Scalar reconstruction loss
        score: Per-frame anomaly scores, shape (T,)
        x_feat: Frame-concatenated input features, shape (T, concat_dim)
        y:    Reconstructed features, shape (T, concat_dim)
    """
    # STFT
    Xr, Xi = exe_fft(x, sp_param)
    # Feature extraction (log-mel)
    fet = feature_extraction(Xr, Xi, BPF, sp_param["Log_reg"])
    # Forward through autoencoder
    score, x_feat, y = model(fet)
    # Loss
    loss = loss_MMSE(score)
    return loss, score, x_feat, y


def exe_one_set(X_set: list):
    """
    Train on one set of wav files.
    
    Args:
        X_set: List of waveform tensors on device
    
    Returns:
        avg_loss: Average loss over the set
        score, x, y: From the last forward pass (for debug visualisation)
    """
    total_cnt = 0
    sum_loss = 0.0
    X_set_perm = np.random.permutation(len(X_set))
    
    bp_cnt = 0
    xin = torch.zeros(sp_param["fftl"], dtype=torch.float32, device=device)
    
    for jj in range(len(X_set)):
        sample_id = X_set_perm[jj]
        x = X_set[sample_id]
        xin = concat_2_wavs(xin, x)
        bp_cnt += 1
        
        if bp_cnt == training_param["Backprop_per_file"]:
            # Forward
            loss, score, x_feat, y = evaluate_wav(xin[sp_param["fftl"]:])
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()
            total_cnt += 1
            bp_cnt = 0
            xin = torch.zeros(sp_param["fftl"], dtype=torch.float32, device=device)
    
    avg_loss = sum_loss / max(total_cnt, 1)
    return avg_loss, score, x_feat, y


def debug_draw(score, x, y):
    """Visualise the last forward pass (for debugging)."""
    score_np = score.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.imshow(np.flipud(x_np.T), aspect='auto')
    plt.title('Input (frame-concatenated features)')
    plt.subplot(3, 1, 2)
    plt.imshow(np.flipud(y_np.T), aspect='auto')
    plt.title('Reconstruction')
    plt.subplot(3, 1, 3)
    plt.plot(score_np)
    plt.xlim([0, len(score_np)])
    plt.title('Anomaly Score (per-frame MSE)')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

print('Training start...')
model.train()

for epoch in range(1, training_param["MAX_EPOCH"] + 1):
    print("-" * 64)
    
    # Shuffle sets
    set_perm = np.random.permutation(len(dev_set))
    sum_loss = 0.0
    
    for ii in tqdm(range(len(dev_set)), desc=f'Epoch {epoch}/{training_param["MAX_EPOCH"]}'):
        # Move wav files to device
        X_set = list_to_device(dev_set[set_perm[ii]], device)
        avg_loss, score, x_feat, y = exe_one_set(X_set)
        sum_loss += avg_loss
    
    print(f"      epoch: {epoch} - Development Loss = {sum_loss:.6f}")
    
    # ── Linear learning rate decay ───────────────────────────────────────
    if epoch > training_param["lr_decal_start"]:
        lr_subtract = training_param["lr_base"] / training_param["lr_decay_factor"]
        lr_subtract /= (training_param["MAX_EPOCH"] - training_param["lr_decal_start"])
        for param_group in optimizer.param_groups:
            param_group['lr'] -= lr_subtract
        print(f"      LR decayed to: {optimizer.param_groups[0]['lr']:.2e}")
    
    # ── Debug visualisation ──────────────────────────────────────────────
    if debug_mode:
        debug_draw(score, x_feat, y)


# ══════════════════════════════════════════════════════════════════════════════
# Save model
# ══════════════════════════════════════════════════════════════════════════════

save_path = os.path.join(dnn_dir, f'{toy_type}_torch.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'sp_param': sp_param,
    'dnn_param': dnn_param,
    'training_param': training_param,
    'epoch': epoch,
}, save_path)
print(f'Model saved to: {save_path}')
print('Training complete.')
