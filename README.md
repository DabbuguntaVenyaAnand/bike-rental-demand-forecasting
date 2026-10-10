# Forecasting Bike Rental Demand

**Machine Learning Mini-Project — Project No. 87**

This project predicts **hourly bike rental demand** using temporal, calendar, and weather-related features. The task is formulated as a supervised **regression** problem using the UCI Bike Sharing hourly dataset.

The implementation includes exploratory data analysis, leakage-safe feature engineering, temporal validation, comparison of multiple regression models, final model selection, and a Streamlit-based live prediction demo.

---

## Team Members

| Student Name | SRN |
|---|---|
| Dabbugunta Venya Anand | PES1UG24AM074 |
| Dhruv Talavat | PES1UG24AM087 |

---

## Problem Statement

Bike-sharing demand varies strongly with factors such as:

- hour of day
- weekday / weekend patterns
- working-day status
- season
- weather
- feels-like temperature
- humidity
- windspeed

The objective is to learn these relationships from historical hourly observations and estimate the expected number of bike rentals for a future riding scenario.

**Target variable:** `cnt` — total hourly bike rentals.

---

## Dataset

We use the **hourly Bike Sharing Dataset** originally published through the UCI Machine Learning Repository.

The dataset contains **17,379 hourly observations** from **2011–2012**.

Important variables include:

- `dteday` — date
- `season` — season code
- `yr` — year indicator
- `mnth` — month
- `hr` — hour
- `holiday` — public-holiday indicator
- `weekday` — day of week
- `workingday` — working-day indicator
- `weathersit` — weather condition
- `temp` — normalized temperature
- `atemp` — normalized feels-like temperature
- `hum` — normalized humidity
- `windspeed` — normalized windspeed
- `casual` — casual-user rentals
- `registered` — registered-user rentals
- `cnt` — total rentals

`casual` and `registered` are **not used as model inputs**, because:

```text
cnt = casual + registered
```

Using them would introduce target leakage.

The raw dataset is not committed to the repository. It can be downloaded automatically using the provided script.

---

## Project Workflow

```text
Dataset
   ↓
Data Understanding & EDA
   ↓
Temporal Train/Test Split
   ↓
Feature Engineering
   ↓
Model Training + TimeSeriesSplit Validation
   ↓
Linear Regression
KNN Regression
Random Forest
Gradient Boosting
   ↓
RMSLE-based Comparison
   ↓
Final Model Selection
   ↓
Streamlit Prediction Demo
```

---

## Exploratory Data Analysis

The EDA pipeline studies demand patterns across time and weather conditions, including:

- target distribution
- daily rental trend
- hourly demand
- seasonal demand
- weather-condition demand
- feels-like temperature vs demand
- humidity vs demand
- windspeed vs demand
- correlation structure
- casual vs registered rentals
- monthly demand across years
- working-day vs non-working-day hourly profiles

Generated visualizations are stored in `reports/`.

Run:

```bash
python src/data_understanding.py
```

---

## Feature Engineering

The final modeling pipeline creates temporal and behavioral features while avoiding target leakage.

### Numerical features

- feels-like temperature (`atemp`)
- humidity
- windspeed
- cyclic hour encoding
- cyclic weekday encoding
- cyclic month encoding

### Calendar / categorical features

- season
- weather condition
- hour
- weekday
- month
- year
- holiday
- working day
- weekend indicator
- rush-hour indicator
- weekday/weekend peak-time category

The following columns are explicitly excluded from model features:

```text
cnt
casual
registered
instant
```

All final models use the **same feature set** for a fair comparison.

---

## Validation Strategy

Because the observations are chronological, the project avoids a random train/test split.

The data is sorted by date and hour and split temporally:

- **Training:** 13,915 rows
- **Holdout test:** 3,464 rows
- **Split boundary:** 7 August 2012

Hyperparameter selection is performed only on the training period using:

```text
TimeSeriesSplit(n_splits=4)
```

The primary model-selection metric is **RMSLE (Root Mean Squared Logarithmic Error)**.

Lower RMSLE is better.

The final holdout period is used only for the final generalization comparison.

---

## Models

Four regression models are compared under the same validation framework:

1. **Linear Regression** — simple baseline
2. **K-Nearest Neighbors Regression**
3. **Random Forest Regression**
4. **Gradient Boosting Regression**

KNN uses scaled numerical features because it is distance-based. Tree-based models do not require feature scaling.

Random Forest, KNN, and Gradient Boosting are tuned using temporal cross-validation with RMSLE as the scoring metric.

---

## Final Results

### Temporal holdout performance

| Model | CV RMSLE | Test RMSLE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **0.4833** | **0.4115** | **80.43** | **53.13** | **0.8667** |
| KNN Regression | 0.5976 | 0.5335 | 122.35 | 83.59 | 0.6914 |
| Gradient Boosting | 0.7257 | 0.6278 | 96.87 | 68.24 | 0.8066 |
| Linear Regression | 1.2964 | 0.9813 | 112.48 | 83.87 | 0.7392 |

### Selected production model

**Random Forest Regression**

Best cross-validation configuration:

```text
n_estimators = 400
max_depth = None
min_samples_leaf = 1
```

The production model is selected using the **lowest training-period cross-validation RMSLE**, rather than choosing a model based on the final test set.

Detailed results are saved in:

```text
models/model_metrics.json
models/model_diagnostics.json
reports/15_final_model_comparison.png
reports/16_best_model_pred_vs_actual.png
```

---

## Baseline Pipeline

A separate baseline script is retained for reproducibility of the earlier Linear Regression and Random Forest experiment.

Run:

```bash
python src/preprocess_and_baseline.py
```

Its artifacts are stored in:

```text
models/baseline_metrics.json
models/baseline_diagnostics.json
reports/13_baseline_pred_vs_actual.png
reports/14_baseline_residuals.png
```

The final model comparison in `src/train_models.py` should be treated as the authoritative model-selection pipeline.

---

## Live Demo

The project includes a Streamlit interface for interactive demand prediction.

The user selects a date and weather conditions, and the application automatically derives consistent calendar features such as:

- year
- month
- weekday
- season
- weekend status
- working-day status

The demo then reuses the same feature-engineering logic as the training pipeline before sending the feature vector to the final production model.

Run:

```bash
streamlit run app/demo_app.py
```

> `models/production_model.pkl` is generated locally by `src/train_models.py` and is not required to be stored in Git.

---

## Repository Structure

```text
bike-rental-demand-forecasting/
│
├── README.md
├── requirements.txt
│
├── app/
│   └── demo_app.py
│
├── data/
│   └── hour.csv                  # downloaded locally; gitignored
│
├── docs/
│   ├── guidelines_text.txt
│   ├── reference_paper_text.txt
│   └── team_writeup.md
│
├── models/
│   ├── baseline_diagnostics.json
│   ├── baseline_metrics.json
│   ├── model_diagnostics.json
│   ├── model_metrics.json
│   └── temp_bucket_edges.json
│
├── reports/
│   ├── 01_target_distribution.png
│   ├── ...
│   ├── 15_final_model_comparison.png
│   └── 16_best_model_pred_vs_actual.png
│
├── scripts/
│   ├── download_data.py
│   └── md2pdf.py
│
└── src/
    ├── data_understanding.py
    ├── preprocess_and_baseline.py
    ├── model_config.py
    ├── feature_engineering.py
    ├── model_training.py
    ├── evaluation.py
    └── train_models.py
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/DabbuguntaVenyaAnand/bike-rental-demand-forecasting.git
cd bike-rental-demand-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

**Windows Git Bash**

```bash
source .venv/Scripts/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python scripts/download_data.py
```

This creates:

```text
data/hour.csv
```

### 5. Run EDA

```bash
python src/data_understanding.py
```

### 6. Train and compare the final models

```bash
python src/train_models.py
```

This generates the final metrics, plots, and local production model.

### 7. Launch the demo

```bash
streamlit run app/demo_app.py
```

---

## Reproducing the Complete Pipeline

From a fresh clone:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt

python scripts/download_data.py
python src/data_understanding.py
python src/preprocess_and_baseline.py
python src/train_models.py

streamlit run app/demo_app.py
```

---

## Reference

The project problem statement is based on:

**Jimmy Du, Rolland He, Zhivko Zhechev — _Forecasting Bike Rental Demand_**, Stanford CS229 Project, 2014.

Reference paper:

https://cs229.stanford.edu/proj2014/Jimmy%20Du%2C%20Rolland%20He%2C%20Zhivko%20Zhechev%2C%20Forecasting%20Bike%20Rental%20Demand.pdf

Dataset:

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
