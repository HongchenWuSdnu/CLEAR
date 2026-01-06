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


