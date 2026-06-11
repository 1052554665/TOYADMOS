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


