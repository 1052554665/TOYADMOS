# -*- coding: utf-8 -*-
"""
PyTorch-compatible data loading and mel-filterbank utilities.
Adapted from the original my_modules.py (Chainer → PyTorch).
"""
import numpy as np
import os
import scipy as scipy
from scipy.io import wavfile
import glob
from tqdm import tqdm
import torch


# ── I/O ──────────────────────────────────────────────────────────────────────

def wavread(fn):
    """Read a 16-bit WAV file, return float32 normalised to [-1, 1] and sample rate."""
    fs, data = wavfile.read(fn)
    data = data.astype(np.float32) / (2 ** 15)
    return data, fs


def wavwrite(fn, data, fs):
    """Write a float32 [-1, 1] signal to a 16-bit WAV file."""
    data = np.array(np.around(data * (2 ** 15)), dtype="int16")
    wavfile.write(fn, fs, data)


# ── Training-set loader ──────────────────────────────────────────────────────

def wav_read_trn(obs_dir, wav_per_set, wav_read_gain):
    """Load training WAV files and group them into sets (batches of wav files)."""
    obs_files = glob.glob(os.path.join(obs_dir, "*.wav"))
    Num_wav = len(obs_files)
    TrnIndex = np.random.permutation(Num_wav)
    obs_trn_all = []
    X_set = []
    cnt = 0
    print('Loading... (Training set)')
    for ii in tqdm(range(Num_wav)):
        x, org_fs = wavread(obs_files[TrnIndex[ii]])
        X_set.append(x * wav_read_gain)
        cnt += 1
        if cnt == wav_per_set:
            cnt = 0
            obs_trn_all.append(X_set)
            X_set = []
    return obs_trn_all


# ── Test-set loader ──────────────────────────────────────────────────────────

def wav_read_test(wav_dir, wav_read_gain):
    """Load all test WAV files and return signals + filenames."""
    wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
    Num_wav = len(wav_files)
    S_all = []
    fn_all = []
    print('Loading... (Test set)')
    for ii in tqdm(range(Num_wav)):
        fn = wav_files[ii]
        x, org_fs = wavread(fn)
        S_all.append(x * wav_read_gain)
        # Strip directory prefix to get relative filename
        fn_all.append(os.path.basename(fn))
    return S_all, fn_all


# ── Device helper ────────────────────────────────────────────────────────────

def list_to_device(ll, device):
    """Move a list of numpy arrays to the given torch device."""
    rr = []
    for ii in range(len(ll)):
        rr.append(torch.from_numpy(ll[ii]).float().to(device))
    return rr


# ── Mel filter bank ──────────────────────────────────────────────────────────

def hz2mel(f):
    return 1127.01048 * np.log(f / 700.0 + 1.0)


def mel2hz(m):
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)


def melFilterBank(fs, nfft, numChannels):
    """
    Design a mel-spaced filter bank.
    
    Args:
        fs:          Sampling rate [Hz]
        nfft:        FFT size
        numChannels: Number of mel filter banks
    
    Returns:
        filterbank:  (numChannels, nfft//2+1) numpy float32 array
        fcenters:    Centre frequencies of each filter [Hz]
    """
    fmax = fs / 2
    melmax = hz2mel(fmax)
    nmax = nfft / 2 + 1
    df = fs / nfft
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = fcenters // df
    if indexcenter[0] == 0:
        indexcenter[0] = 1
    for ii in range(1, len(indexcenter)):
        if indexcenter[ii - 1] >= indexcenter[ii]:
            indexcenter[ii] = indexcenter[ii - 1] + 1
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((int(numChannels), int(nmax)))
    for c in np.arange(0, numChannels):
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            filterbank[c, int(i)] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            filterbank[c, int(i)] = 1.0 - ((i - indexcenter[c]) * decrement)

    for c in np.arange(0, numChannels):
        filterbank[c, :] = filterbank[c, :] / (1e-8 + filterbank[c, :].sum())

    filterbank = filterbank.astype(np.float32)
    return filterbank, fcenters


# ── Debug ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print('debug')
