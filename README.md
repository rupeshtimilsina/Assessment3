# Assessment 2: IMDb Sentiment Classification

This project compares a TF-IDF + logistic regression baseline with a fine-tuned DistilBERT model for binary sentiment classification on IMDb movie reviews.

The notebook also examines truncation windows, frozen versus fine-tuned DistilBERT, model errors, interpretability, and a gender-swap sensitivity probe.

## Clone the project

```bash
git clone https://github.com/rupeshtimilsina/Assessment3
cd https://github.com/rupeshtimilsina/Assessment3
```

The trained model is about 268 MB. If the repository uses Git LFS, install it before pulling the model:

```bash
git lfs install
git lfs pull
```

If Git LFS is unavailable, download the large model artifact from the repository storage and place it at:

```text
artifacts/advanced/model.safetensors
```

## Install dependencies

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CPU-only demo use, install the PyTorch build suitable for the local machine if the pinned CUDA build in `requirements.txt` is not available.

## Run the offline demo

1. Open `Assessment2_IMDb_Truncation_NLP.ipynb` in Jupyter, JupyterLab or VS Code.
2. Select the virtual environment as the notebook kernel.
3. Restart the kernel.
4. Run the cells under **Part B: live demo**, beginning with the setup cell.

The demo performs inference only. It does not download the IMDb dataset or train a model. It reads the cached model from `artifacts/` and the 120-review sample from `demo.csv`.

The demo includes predictions, a 128-versus-256 token toggle, visualisations, a failure case and the ethics probe. The cached model is configured to run on CPU.

## Run the full experiment

Part A downloads the IMDb dataset and trains the baseline and DistilBERT experiments. A GPU is recommended.

1. Open `Assessment2_IMDb_Truncation_NLP.ipynb`.
2. Restart the kernel.
3. Run the cells under **Part A: training and experiments** from the beginning.

Part A writes trained models, results and demo data to `artifacts/`, `backup/` and `demo.csv`.

## Project structure

```text
.
├── Assessment3_IMDb_Truncation_NLP_Timilsina_SHEA24023.ipynb
├── README.md
├── requirements.txt
├── demo.csv
├── poster.png
├── artifacts/
│   ├── advanced/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── baseline.joblib
│   ├── baseline_cda.joblib
│   ├── config.json
│   └── results.json
├── backup/
│   ├── ablation_val.png
│   ├── demo_results.png
│   ├── interpretability.png
│   ├── length_hist.png
│   └── test_results.png
└── __results___files/
    ├── __results___8_5.png
    ├── __results___13_17.png
    ├── __results___16_2.png
    ├── __results___21_2.png
    ├── __results___41_0.png
    └── __results___41_1.png
```

## Reproducibility

The notebook uses a fixed random seed of `42`. The saved results and model artifacts allow the offline demo to run without repeating training.
