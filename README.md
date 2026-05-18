# CLEAR: Prototype-conditioned flow purification for LLM-based rumor detection with Dirichlet evidential learning
Detecting rumors on social media is challenging when posts are semantically underspecified and discussion threads are noisy or polarized, which can encourage detectors to exploit spurious correlations. We propose CLEAR (Contextual Potential Alignment Capture Network), an evidence-grounded framework that models hierarchical comment dynamics and incorporates auxiliary LLM-based veracity assessments for credibility-aware prediction. CLEAR couples prototype-conditioned flow purification with Dirichlet evidential learning to derive geometry-grounded evidence for calibrated inference. We further introduce an entropy-adaptive Hard-Shift reweighting strategy to suppress noise-driven shortcuts. Experiments on Weibo-19 (2927 samples) and PHEME (2018 samples) show that CLEAR achieves 93.16% and 91.56% accuracy, outperforming the average strong recent baselines by 3.2 and 5.5 percentage points, respectively. To stress-test generalization under distribution shift, we curate VRDD with 4020 posts (2348 non-rumors and 1672 rumors), a boundary-dense benchmark that emphasizes vague content. Results confirm CLEAR’s robustness to evolving rumor patterns and highlight the curriculum-dependent effect of reweighting.

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
Due to repository size considerations, the additional baselines (training scripts, checkpoints, and one-click evaluation scripts) are available at [Zenodo](https://zenodo.org/records/18939502?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImYwMGMxMmYwLTNiYzktNGVhMS05YTkzLWU4MWYzZGJlYTI1YSIsImRhdGEiOnt9LCJyYW5kb20iOiI4NWNlYjc5ZDFmOWQyMDk3MDAxNmZmMTVmOWRhODE0MyJ9.5Bjv1_9bheFbFQZEf-fZs-r9wMS43vY-UFuUPnJmXRmKlJ38rlegMDptQoy83l9NFBqx_FAZDoC1EF3pTrTHng).

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


## Cite
If you find our code or dataset useful for your research, please consider citing our paper:

```bibtex
@article{LIU2026104887,
  title = {CLEAR: Prototype-conditioned flow purification for LLM-based rumor detection with Dirichlet evidential learning},
  journal = {Information Processing & Management},
  volume = {63},
  number = {7, Part B},
  pages = {104887},
  year = {2026},
  issn = {0306-4573},
  doi = {10.1016/j.ipm.2026.104887},
  author = {Zihao Liu and Hongchen Wu and Xiaochang Fang and Guanlin Liu and Hongxuan Li and Zhaorong Jing and Huaxiang Zhang}
}
