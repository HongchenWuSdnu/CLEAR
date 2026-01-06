# CLEAR

## Running

### 1) Install Dependencies
We provide a `requirements.txt` for environment setup:

```bash
pip install -r requirements.txt
```

### 2) Training

#### Weibo19
```bash
python train_Weibo19.py
```

#### Pheme
```bash
python train_pheme.py
```

#### VRDD
```bash
python train_VRDD.py
```

### 3) Evaluation (VRDD)
r
#### Test a single checkpoint (default: uns both TEST and OOD/VAL)
```bash
python test.py Model_name.bin
```

#### Test multiple checkpoints (batch comparison)
```bash
python test.py Model_name_A.bin B.bin C.bin
```

#### OOD only
```bash
python test.py Model_name.bin --run ood
```
## Dataset

All datasets are split into training/validation/test sets with a 6:2:2 ratio; we select the best epoch based on validation accuracy and report the corresponding performance on the test set. 
The other two publicly available datasets are Weibo19 (Song et al., 2019) and Pheme (Zubiaga et al., 2017). We use the JSON-integrated version provided at <[MFAN](https://github.com/drivsaf/MFAN)>. You may construct the dataset using json_to_xlsx.py, or directly use our preprocessed version.
```bash
export OPENAI_API_KEY="your_key"
python ced_pipeline_minimal_api.py --dataset_dir dataset --output dataset.xlsx
```
