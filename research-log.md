## bug
```plaintext
/home/yangchen/miniconda3/envs/TOYADMOS/lib/python3.12/site-packages/chainer/backends/cuda.py:154: UserWarning: cuDNN is not enabled.
Please reinstall CuPy after you install cudnn
(see https://docs-cupy.chainer.org/en/stable/install.html#install-cudnn).
  warnings.warn(
/home/yangchen/miniconda3/envs/TOYADMOS/lib/python3.12/site-packages/chainer/_environment_check).py:72: UserWarning: 
--------------------------------------------------------------------------------
CuPy (cupy) version 14.1.1 may not be compatible with this version of Chainer.
Please consider installing the supported version by running:
  $ pip install 'cupy>=7.7.0,<8.0.0'
```

Since chainer is **officially discontinued** (development stopped in 2019), migrating the autoencoder to PyTorch.

### Fix: port to PyTorch

Full Chainer→PyTorch migration at `E01_simple_AE_test/torch_version/`

| File | Original (Chainer) | PyTorch |
|---|---|---|
| Config | `Modules/Config.py` | `torch_version/config.py` (wrapper) |
| Model | `Modules/model_definition.py` | `torch_version/model.py` |
| Data I/O | `Modules/my_modules.py` | `torch_version/modules.py` |
| DSP/GPU | `Modules/my_gpu_funcs.py` | `torch_version/gpu_funcs.py` |
| Training | `01_train.py` | `torch_version/train.py` |
| Evaluation | `02_test.py` | `torch_version/test.py` |

**Architecture preserved:**
- FCN_AE: BN → FrameConcat (±10 context frames) → Encoder (4 hidden layers×512) → Bottleneck (128-dim) → Decoder → MSE anomaly score
- Adam with amsgrad, weight decay
- Mel filterbank (64 bands) + log-mel features
- Manual STFT via `torch.fft.rfft` (matches original CuPy FFT exactly)

**Verification results (ToyConveyor, RTX 5090):**
- Training: 20 epochs, loss 339.8 → 223.4
- AUC: **0.9758**
- F-measure @ 10% FPR: **0.8953**
- GPU acceleration: ~580–675 it/s inference

**Dependencies removed:** chainer, cupy  
**Dependencies added:** torch (already in environment)

