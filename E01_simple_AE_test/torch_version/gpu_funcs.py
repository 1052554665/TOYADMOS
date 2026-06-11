# -*- coding: utf-8 -*-
"""
PyTorch-compatible GPU functions for STFT, feature extraction, and loss.
Adapted from the original my_gpu_funcs.py (Chainer → PyTorch).
"""
import numpy as np
import torch
import torch.nn.functional as F


# ── Waveform concatenation ───────────────────────────────────────────────────

def concat_2_wavs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Concatenate two waveform tensors along the time axis."""
    return torch.cat((x, y), dim=0)


# ── STFT (Short-Time Fourier Transform) ──────────────────────────────────────

def exe_fft(x: torch.Tensor, sp_param: dict):
    """
    Compute STFT of a 1D waveform, matching the original manual FFT pipeline.
    
    Args:
        x:        1D waveform tensor, shape (L,)
        sp_param: Dict with keys 'fftl', 'shift', 'win'
    
    Returns:
        Xr: Real part of STFT, shape (fftl//2+1, T)
        Xi: Imaginary part of STFT, shape (fftl//2+1, T)
    """
    fftl = sp_param["fftl"]
    shift = sp_param["shift"]
    win = sp_param["win"]           # (fftl, 1) or (fftl,)
    
    L = x.shape[0]
    # Number of frames — exact match to original: floor((L - fftl) / shift)
    T = int(np.floor((L - fftl) / float(shift)))
    
    if T < 1:
        # Signal too short: pad to at least fftl
        x = F.pad(x, (0, max(0, fftl - L)))
        T = 1
    
    # Build index matrix: (fftl, T)
    shift_vec = torch.arange(0, shift * T, shift, device=x.device)       # (T,)
    idx_base = torch.arange(0, fftl, device=x.device).unsqueeze(1)       # (fftl, 1)
    indices = idx_base + shift_vec.unsqueeze(0)                           # (fftl, T)
    X = x[indices.long()]                                                 # (fftl, T)
    
    # Apply window (broadcast)
    if win.dim() == 2:
        win = win.squeeze(1)
    W = win.unsqueeze(1).expand(-1, T)                                    # (fftl, T)
    X = X * W
    
    # FFT along dim=0 (each column is a windowed frame)
    X_complex = torch.fft.rfft(X.t(), n=fftl, dim=1)                     # (T, fftl//2+1)
    Xr = X_complex.real.t()                                               # (fftl//2+1, T)
    Xi = X_complex.imag.t()                                               # (fftl//2+1, T)
    
    return Xr, Xi


# ── Feature extraction (log-mel spectrogram) ─────────────────────────────────

def feature_extraction(Xr: torch.Tensor, Xi: torch.Tensor,
                        BPF: torch.Tensor, Log_reg: float) -> torch.Tensor:
    """
    Convert complex STFT to log-mel-filterbank features.
    
    Args:
        Xr:      Real part of STFT, shape (fftl//2+1, T)
        Xi:      Imaginary part of STFT, shape (fftl//2+1, T)
        BPF:     Mel filter bank matrix, shape (NumFB, fftl//2+1)
        Log_reg: Small constant for log stability
    
    Returns:
        Log-mel spectrogram, shape (NumFB, T)
    """
    # Magnitude spectrum
    abs_X = torch.sqrt(Xr ** 2 + Xi ** 2 + 1e-10)         # (fftl//2+1, T)
    # Apply mel filter bank
    fet = torch.log(torch.matmul(BPF, abs_X) + Log_reg)   # (NumFB, T)
    return fet


# ── Loss function ────────────────────────────────────────────────────────────

def loss_MMSE(score: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss for anomaly scores.
    In the original, this is the average anomaly score across all time frames.
    
    Args:
        score: Per-frame anomaly scores, shape (T,)
    
    Returns:
        Scalar loss
    """
    return torch.mean(score)


# ── Debug ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print('debug')
    
    # Quick test
    sp_param = {
        "fs": 16000.0,
        "fftl": 512,
        "shift": 256,
        "Log_reg": 1e-8,
    }
    win = torch.hann_window(512)
    sp_param["win"] = win
    
    # Dummy signal
    x = torch.randn(16000)
    Xr, Xi = exe_fft(x, sp_param)
    print(f"Xr shape: {Xr.shape}, Xi shape: {Xi.shape}")
    
    # Dummy filter bank
    from torch_version.modules import melFilterBank
    BPF_np, _ = melFilterBank(16000, 512, 64)
    BPF = torch.from_numpy(BPF_np)
    fet = feature_extraction(Xr, Xi, BPF, 1e-8)
    print(f"Features shape: {fet.shape}")
