## About

### Dataset Split (ToyConveyor)

The split is defined in `make_dataset_for_car_and_conveyor.py` and is an **unsupervised anomaly detection** setup — the model sees **only normal sounds during training**.

| Split | Count | Description |
|---|---|---|
| `train_normal/` | **1,000** | Normal sounds mixed with noise (+10 dB SNR), **no anomalies** |
| `test_normal/` | **800** | Normal sounds mixed with noise (held-out) |
| `test_anomaly/` | **400** | Anomalous sounds mixed with noise (all anomaly types) |

### Split Logic (lines 25, 104–110 of the script)

```python
num_train_samples = 1000       # ← hard-coded constant

# All normal files are shuffled, then:
#   first 1000  →  train_normal/
#   remainder   →  test_normal/
# All anomaly files → test_anomaly/  (none in training)
```

### Split Ratios

| Perspective | Ratio |
|---|---|
| Normal data (train : test) | 1000 : 800 → **55.6% / 44.4%** |
| Overall (train : test) | 1000 : 1200 → **45.5% / 54.5%** |
| Test set composition (normal : anomaly) | 800 : 400 → **2 : 1** |

### Why No Anomalies in Training?

This is by design for **unsupervised anomaly detection**: the autoencoder learns to reconstruct only normal sounds. At test time, anomalous sounds produce higher reconstruction error → anomaly score. This is the standard setup in the ToyADMOS paper (Koizumi et al., WASPAA 2019).


## Usage
### Environment
#### Option 1
The environment is packed as a tar.gz file named `TOYADMOSENV.tar.gz`, you can directorly unpack it.

**To unpack on another machine:**
```bash
mkdir -p ~/miniconda3/envs/TOYADMOS
tar -xzf TOYADMOSENV.tar.gz -C ~/miniconda3/envs/TOYADMOS
~/miniconda3/envs/TOYADMOS/bin/conda-unpack
```

#### Option 2
or Run `conda env create -f environment.yml` to recreate the environment.

then, Run `conda activate TOYADMOS`.

### make dataset
- `cd C01_create_small_INT_dataset`
- Run `make_dataset_for_car_and_conveyor.py` in `C01_create_small_INT_dataset` to make dataset.

### train and test
#### Option 1: Chainer (original code)

- `cd E01_simple_AE_test`
- Run `make_dataset_for_car_and_conveyor.py`
- Run `01_train.py` in `E01_simple_AE_test` to train a model
- Run `02_test.py` in `E01_simple_AE_test`  to evaluate a model

#### Option 2: PyTorch (ported code)
- `cd E01_simple_AE_test`
- Run `make_dataset_for_car_and_conveyor.py`
- Run `torch_version/train.py` in `E01_simple_AE_test/torch_version` to train a model
- Run `torch_version/test.py` in `E01_simple_AE_test/torch_version` to evaluate a model


