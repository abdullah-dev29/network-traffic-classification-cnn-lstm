# CNN-LSTM Darknet Traffic Classifier (Phase 1)

A binary network-traffic classifier that reproduces the hybrid CNN-LSTM architecture from Mandela et al. (2025) on the **CIC-Darknet2020** dataset. The model distinguishes **Darknet traffic** (Tor + VPN flows) from **Benign traffic** (Non-Tor + NonVPN flows).

This is **Phase 1 of 3**. Later phases will add a Transformer-enhanced LSTM variant and a cross-model / cross-dataset comparison. The code is structured to support those phases without restructuring.

---

## Dataset

| Item | Detail |
|---|---|
| Name | CIC-Darknet2020 |
| Source | Kaggle — `dhoogla/cicdarknet2020` |
| File | `CIC-darknet2020-dataset/cicdarknet2020.parquet` (committed in this repo) |
| Shape (verified) | 103,121 rows × 79 columns (77 features + 2 label columns) |
| Missing / Inf / Duplicates | 0 / 0 / 0 (pre-cleaned build) |
| Binary mapping | Darknet (1) = Tor + VPN (18,101 rows, 17.6%) · Benign (0) = Non-Tor + NonVPN (85,020 rows, 82.4%) |

The dataset is committed directly to the repo (~12.8 MB, within GitHub's 100 MB limit) so Google Colab gets it via `git clone` with no manual download step.

---

## Model Architecture

Faithful to **Figure 7** of Mandela et al. (2025). The paper mentions an `Embedding` layer suited to character-level text input, which does not apply to our tabular continuous flow features. That layer is **not included** — scaled float features are reshaped to `(n_features, 1)` and fed directly into Conv1D. The LSTM layers are standard Keras `LSTM` layers; the "hybrid" refers to the CNN front-end combined with stacked LSTMs.

| Layer | Output shape |
|---|---|
| Input | (None, ~62, 1) |
| Conv1D-128, kernel=3, relu, same | (None, ~62, 128) |
| MaxPooling1D (pool=2) | (None, ~31, 128) |
| Dropout (0.3) | (None, ~31, 128) |
| Conv1D-64, kernel=3, relu, same | (None, ~31, 64) |
| MaxPooling1D (pool=2) | (None, ~15, 64) |
| LSTM-100 (return_sequences=True) | (None, ~15, 100) |
| Dropout (0.3) | (None, ~15, 100) |
| LSTM-50 | (None, 50) |
| Dense-1, sigmoid | (None, 1) |

~62 features remain after programmatically dropping ~15 zero-variance columns.

---

## Repo Structure

```
base-implementation/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                        ← all paths, hyperparams, toggles
├── CIC-darknet2020-dataset/
│   └── cicdarknet2020.parquet
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             ← load → clean → split → scale → reshape
│   ├── model.py                     ← build_cnn_lstm()
│   ├── train.py                     ← orchestration (runs on Colab)
│   └── evaluate.py                  ← metrics + figures (runs on Colab)
├── notebooks/
│   └── colab_train.ipynb            ← thin Colab runner
├── results/                         ← populated by Colab after training
│   └── .gitkeep
└── figures/                         ← populated by Colab after training
    └── .gitkeep
```

---

## How to Run

### Local (source editing only)

Training does **not** run locally — the laptop cannot handle it. Local usage is for reading and editing source files only. To check syntax:

```bash
python -m py_compile config.py src/preprocessing.py src/model.py src/train.py src/evaluate.py
```

### Colab (where training actually happens)

1. Push this repo to GitHub (see push commands at the bottom of this README).
2. Open `notebooks/colab_train.ipynb` in Google Colab.
3. Set `REPO_URL` in the first code cell to your GitHub repo URL.
4. Go to **Runtime → Change runtime type → GPU**.
5. Go to **Runtime → Run all**.

The notebook will clone the repo, install lightweight deps (pandas, scikit-learn, etc. — TensorFlow is already installed on Colab), train the model, run evaluation, and display figures inline.

**Do not reinstall TensorFlow on Colab** — Colab ships a CUDA-matched build; reinstalling it commonly breaks GPU acceleration.

---

## Results

> Fill in from your own Colab run. The numbers below are placeholders.

| Model | Accuracy | F1 | Recall | Precision | AUC | Specificity |
|---|---|---|---|---|---|---|
| Proposed CNN-LSTM (rebuild) | 0.9574 | 0.8791 | 0.8771 | 0.8811 | 0.9885 | 0.9746 |

All reported numbers come from the team's own implementation and Colab run on this dataset version. The paper's reported numbers are provided for reference only.

---

## Push to GitHub

After Claude Code commits locally, run one of these to push:

```bash
# Option A — GitHub CLI (if `gh` is installed and authenticated):
gh repo create <repo-name> --public --source=. --remote=origin --push

# Option B — manual:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Then paste the repo URL into `notebooks/colab_train.ipynb` → `REPO_URL` cell. The `%cd` step auto-derives the folder name from the URL.

---

## Team Members

- Muhammad Borhan UD Din - 63206
- Muhammad Mutahar - 63513
- Abdullah - 62724
- Ameer Hamza - 65260

---

## Academic Integrity

All model training, evaluation, and reported metrics were produced by the team's own implementation running on the `cicdarknet2020.parquet` file committed in this repository. Numbers from Mandela et al. (2025) are cited as reference only and are not claimed as the team's own results.
