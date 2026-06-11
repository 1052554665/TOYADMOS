# -*- coding: utf-8 -*-
"""
PyTorch test/evaluation script for AE-based anomaly detection in sounds.

Migrated from 02_test.py (Chainer → PyTorch).

Computes:
  - Per-file anomaly scores (max per-frame MSE)
  - ROC curve & AUC
  - F-measure at a given false-positive rate
  - Overlooked (missed) anomaly files report

Reference:
  Y. Koizumi, et al., "ToyADMOS: A Dataset of Miniature-Machine Operating
  Sounds for Anomalous Sound Detection," WASPAA 2019.
"""
import numpy as np
import collections
import os
import sys
import torch
import scipy.signal.windows
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# ── Ensure parent directory is on sys.path ────────────────────────────────────
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.join(_parent_dir, 'Modules'))

from torch_version.config import load_config
from torch_version.model import FCN_AE
from torch_version.modules import wav_read_test, melFilterBank
from torch_version.gpu_funcs import exe_fft, feature_extraction


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

sp_param, dnn_param, training_param = load_config()

# Toy type
toy_type = 'ToyConveyor'          # 'ToyCar' or 'ToyConveyor' or 'ToyTrain'

# Paths (resolved relative to the E01_simple_AE_test directory)
dnn_dir = os.path.join(_parent_dir, 'dnn_dir')
model_fn = f'{toy_type}_torch.pth'                     # PyTorch checkpoint
dataset_dir = os.path.join(_parent_dir, f'exp1_dataset_{toy_type}')
nml_dir = os.path.join(dataset_dir, 'test_normal')
anm_dir = os.path.join(dataset_dir, 'test_anomaly')

# Results — saved under a timestamped subfolder within E01_simple_AE_test/
from datetime import datetime
_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
sav_dir = os.path.join(_parent_dir, 'test_results', f'{toy_type}_{_run_timestamp}')
os.makedirs(sav_dir, exist_ok=True)

# Analysis parameters
rho = 0.1                                    # FPR = 10%

# Anomaly conditions lookup (relative to repo root)
_repo_root = os.path.abspath(os.path.join(_parent_dir, '..'))
anomaly_cond_xlsx_dir = os.path.join(_repo_root, 'anomaly_conditions')
xlsx_fn = os.path.join(anomaly_cond_xlsx_dir, f'{toy_type}_anomay_condition.xlsx')
anm_cnd = pd.read_excel(xlsx_fn)

# Report file (saved alongside other results)
report_file = os.path.join(sav_dir, f'{toy_type}_overlook_report.txt')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(device)}')


# ══════════════════════════════════════════════════════════════════════════════
# Prepare filter bank & window
# ══════════════════════════════════════════════════════════════════════════════

win = torch.from_numpy(
    scipy.signal.windows.hann(sp_param["fftl"]).astype(np.float32)
).unsqueeze(1).to(device)
sp_param["win"] = win

BPF_np, _ = melFilterBank(sp_param["fs"], sp_param["fftl"], dnn_param["NumFB"])
BPF = torch.from_numpy(BPF_np.astype(np.float32)).to(device)

# ── Build model & load checkpoint ────────────────────────────────────────────

model = FCN_AE(
    in_dim=dnn_param["NumFB"],
    hid_dim=dnn_param["hid_dim"],
    z_dim=dnn_param["z_dim"],
    num_hid=dnn_param["num_hid"],
    num_fw=dnn_param["num_fw"],
    num_bw=dnn_param["num_bw"],
).to(device)

checkpoint_path = os.path.join(dnn_dir, model_fn)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Loaded checkpoint: {checkpoint_path} (epoch {checkpoint.get("epoch", "?")})')

# Set to eval mode (affects BN running stats)
model.eval()


# ══════════════════════════════════════════════════════════════════════════════
# Sub-module
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_wav(x: torch.Tensor):
    """
    Forward pass — returns per-frame anomaly scores.
    
    Args:
        x: 1D waveform tensor on device, shape (L,)
    
    Returns:
        score: Per-frame anomaly scores, shape (T,) on CPU as numpy
    """
    Xr, Xi = exe_fft(x, sp_param)
    fet = feature_extraction(Xr, Xi, BPF, sp_param["Log_reg"])
    score, _, _ = model(fet)
    return score.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Load test data
# ══════════════════════════════════════════════════════════════════════════════

nml_all, fn_nml = wav_read_test(nml_dir, sp_param["wav_read_gain"])
anm_all, fn_anm = wav_read_test(anm_dir, sp_param["wav_read_gain"])


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate
# ══════════════════════════════════════════════════════════════════════════════

MN = np.zeros((len(nml_all),))       # Max anomaly score per normal file
MA = np.zeros((len(anm_all),))       # Max anomaly score per anomalous file

print('Evaluating normal files...')
for jj in tqdm(range(len(nml_all))):
    x = torch.from_numpy(nml_all[jj]).float().to(device)
    score = evaluate_wav(x)
    svfn = os.path.join(sav_dir, f'nml_{fn_nml[jj]}.csv')
    np.savetxt(svfn, score, delimiter=",")
    MN[jj] = np.max(score)

print('Evaluating anomalous files...')
for jj in tqdm(range(len(anm_all))):
    x = torch.from_numpy(anm_all[jj]).float().to(device)
    score = evaluate_wav(x)
    svfn = os.path.join(sav_dir, f'anm_{fn_anm[jj]}.csv')
    np.savetxt(svfn, score, delimiter=",")
    MA[jj] = np.max(score)


# ══════════════════════════════════════════════════════════════════════════════
# ROC / AUC
# ══════════════════════════════════════════════════════════════════════════════

Thres = np.sort(MN)[::-1]           # Thresholds: sorted normal scores (descending)
TPR = np.zeros_like(MA)
for jj in range(len(anm_all)):
    TPR[jj] = np.sum(MA > Thres[jj]) / len(anm_all)

auc = np.mean(TPR)
print(f'AUC: {auc:.4f}')

# ── Plot and save ROC curve ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(np.linspace(0, 1, len(TPR)), TPR, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
ax.plot([rho, rho], [0, 1], 'r--', linewidth=1.5, label=f'FPR = {rho*100:.0f}%')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.set_title(f'ROC Curve — {toy_type} (PyTorch AE)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')

# Save in multiple formats
roc_pdf_path = os.path.join(sav_dir, 'roc_curve.pdf')
roc_png_path = os.path.join(sav_dir, 'roc_curve.png')
fig.savefig(roc_pdf_path, dpi=300, bbox_inches='tight')   # Publication-quality PDF
fig.savefig(roc_png_path, dpi=150, bbox_inches='tight')   # Quick preview PNG
plt.show()
plt.close(fig)
print(f'ROC curve saved to: {roc_pdf_path}')


# ══════════════════════════════════════════════════════════════════════════════
# F-measure @ FPR = rho
# ══════════════════════════════════════════════════════════════════════════════

rho_index = int(len(Thres) * rho)
TP = TPR[rho_index] * len(TPR)
FP = int(rho * len(Thres))
FN = len(TPR) - TP
Prec = TP / (TP + FP)
Recl = TP / (TP + FN)
Fmsr = (2 * Recl * Prec) / (Recl + Prec)
print(f'F-measure under FPR = {rho*100:.0f}% condition: {Fmsr:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# Save summary metrics to JSON
# ══════════════════════════════════════════════════════════════════════════════

import json

summary = {
    'timestamp':        datetime.now().isoformat(),
    'toy_type':         toy_type,
    'checkpoint':       os.path.basename(checkpoint_path),
    'checkpoint_epoch': checkpoint.get('epoch', '?'),
    'num_test_normal':  len(MN),
    'num_test_anomaly': len(MA),
    'rho_FPR':          rho,
    'AUC':              float(auc),
    'F_measure':        float(Fmsr),
    'Precision':        float(Prec),
    'Recall':           float(Recl),
    'TP':               float(TP),
    'FP':               float(FP),
    'FN':               float(FN),
    'threshold_at_FPR': float(Thres[rho_index]),
    'MN_stats': {
        'min':  float(np.min(MN)),
        'max':  float(np.max(MN)),
        'mean': float(np.mean(MN)),
        'std':  float(np.std(MN)),
    },
    'MA_stats': {
        'min':  float(np.min(MA)),
        'max':  float(np.max(MA)),
        'mean': float(np.mean(MA)),
        'std':  float(np.std(MA)),
    },
}

summary_json_path = os.path.join(sav_dir, 'summary_metrics.json')
with open(summary_json_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'Summary metrics saved to: {summary_json_path}')

# Also save ROC data (FPR, TPR arrays) for external plotting
roc_data_path = os.path.join(sav_dir, 'roc_data.csv')
np.savetxt(roc_data_path,
           np.column_stack((np.linspace(0, 1, len(TPR)), TPR)),
           delimiter=',', header='FPR,TPR', comments='')
print(f'ROC data saved to: {roc_data_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Overlooked files report
# ══════════════════════════════════════════════════════════════════════════════

threshold_rho = Thres[rho_index]
ovl_index = np.where(MA < threshold_rho)[0]
ovl_list = []
for jj in range(len(ovl_index)):
    ovl_fn = fn_anm[ovl_index[jj]].split('ab')
    ovl_list.append('ab' + ovl_fn[1][:2])

c = collections.Counter(ovl_list)

with open(report_file, mode='w') as f:
    f.write(f'Overlooked files under FPR = {rho*100:.0f}% condition:\n')
    for kk in list(c.keys()):
        f.write('-------------------------------------\n')
        idx = int(kk[2:]) - 1
        for ii, cc in enumerate(anm_cnd.columns):
            f.write(f'{cc}: {anm_cnd.iloc[idx, ii]}\n')
        f.write(f'Overlooked times: {c[kk]}\n')
    f.write('-------------------------------------\n')

print(f'Overlook report saved to: {report_file}')
print('Evaluation complete.')
