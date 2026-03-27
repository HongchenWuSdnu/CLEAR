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

### 3）Baselines
Due to repository size considerations, the additional baselines (training scripts, checkpoints, and one-click evaluation scripts) are available at [Zenodo](https://zenodo.org/records/18939502?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZhNmQ2ZDUxLWMwMmQtNGJlZi1iZDI3LTYwMjM0MDAwMWNiOCIsImRhdGEiOnt9LCJyYW5kb20iOiJkNjA4M2UxMjhiMjg4MDA2YTM0NTUwNjAwOTJmZDdmNSJ9.cr5lQXovUPtRU40Oy66_REPvoHWCabI44xJ9QdvV9Nxy9tPx0yuzqD8AD0J-98k83jNTjLlElNnLPEKBgTUbVw).

You can use the following command to test all baseline results at once:
```bash
python evaluate_all_baselines.py --xlsx_dir xlsx --ckpt_dir checkpoints
```
We used the following command to generate two dataset files incorporating white-box LLM priors:
```bash
bash prepare_white_box_priors.sh
```
You can use the following command to test the CLEAR classification results after replacing the prior with a white-box LLM:
```bash
python test_CLEAR_two_local_llm_models.py --train_script train.py
```


## Dataset

All datasets are split into training/validation/test sets with a 6:2:2 ratio; we select the best epoch based on validation accuracy and report the corresponding performance on the test set. 
The other two publicly available datasets are Weibo19 (Song et al., 2019) and Pheme (Zubiaga et al., 2017). We use the JSON-integrated version provided at [MFAN](https://github.com/drivsaf/MFAN). You may construct the dataset using json_to_xlsx.py, or directly use our preprocessed version.
```bash
export OPENAI_API_KEY="your_key"
python json_to_xlsx.py --dataset_dir dataset --output dataset.xlsx
```
