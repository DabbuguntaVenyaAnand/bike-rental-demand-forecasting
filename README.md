# Forecasting Bike Rental Demand

Machine Learning Mini-Project — **Project No. 87**

This project predicts **hourly bike rental demand** using temporal, calendar, and weather-related information.

The problem is formulated as a supervised **regression** task using the UCI Bike Sharing hourly dataset. The implementation covers data understanding, leakage-safe feature engineering, temporal validation, regression model comparison, final model selection, and an interactive Streamlit prediction demo.

---

## Problem Statement

Bike rental demand varies significantly depending on factors such as time of day, day of week, season, weather, humidity, windspeed, holidays, and working-day patterns.

The objective of this project is to learn these relationships from historical bike-sharing data and predict the expected number of bike rentals for a given hourly scenario.

**Target variable:** `cnt` — total number of bike rentals in an hour.

---

## Dataset

The project uses the **Bike Sharing Dataset** from the UCI Machine Learning Repository.

The hourly dataset contains **17,379 observations** from **2011 and 2012**.

Important variables include:

| Feature | Description |
|---|---|
| `dteday` | Date |
| `season` | Season |
| `yr` | Year indicator |
| `mnth` | Month |
| `hr` | Hour |
| `holiday` | Holiday indicator |
| `weekday` | Day of week |
| `workingday` | Working-day indicator |
| `weathersit` | Weather condition |
| `temp` | Normalized temperature |
| `atemp` | Normalized feels-like temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized windspeed |
| `casual` | Casual-user rentals |
| `registered` | Registered-user rentals |
| `cnt` | Total rentals |

Since:

```text
cnt = casual + registered
```

the columns `casual` and `registered` are excluded from model inputs to prevent **target leakage**.

The identifier column `instant` and the target `cnt` are also excluded from the feature matrix.

---

## Project Workflow

```text
Raw Dataset
    ↓
Data Understanding & EDA
    ↓
Chronological Train/Test Split
    ↓
Feature Engineering
    ↓
Temporal Cross-Validation
    ↓
Model Training
    ↓
Model Comparison
    ↓
Final Model Selection
    ↓
Streamlit Prediction Demo
```

---

## Exploratory Data Analysis

The data-understanding stage examines:

- target distribution
- hourly demand patterns
- daily demand trends
- seasonal demand
- weather-condition demand
- working-day and non-working-day behaviour
- temperature, humidity, and windspeed relationships
- casual and registered rental patterns
- correlations between numerical variables
- missing and duplicate observations

Run:

```bash
python src/data_understanding.py
```

EDA plots are generated locally during execution and are not stored permanently in the repository.

---

## Feature Engineering

The final modeling pipeline uses shared feature-engineering logic implemented in:

```text
src/feature_engineering.py
```

### Numerical Features

```text
atemp
hum
windspeed
hour_sin
hour_cos
dayofweek_sin
dayofweek_cos
month_sin
month_cos
```

Cyclic encodings are used for hour, weekday, and month so that periodic relationships are represented correctly.

### Calendar and Categorical Features

```text
season_label
weathersit_label
hr
weekday
mnth
yr
holiday
workingday
peak_time
is_rush_hour
is_weekend
```

Rush hour is defined only for working days during:

```text
07:00–09:00
17:00–19:00
```

The following columns are explicitly excluded from modeling:

```text
cnt
casual
registered
instant
```

All final models use the same feature set for a fair comparison.

---

## Validation Strategy

Because the dataset is chronological, a random train/test split is avoided.

The data is first sorted by date and hour and then divided temporally.

```text
Training rows : 13,915
Test rows     : 3,464
Split date    : 2012-08-07
```

Hyperparameter tuning is performed only on the training period using:

```python
TimeSeriesSplit(n_splits=4)
```

This preserves temporal ordering and prevents future observations from being used to predict the past.

### Primary Metric

The main model-selection metric is:

**RMSLE — Root Mean Squared Logarithmic Error**

Lower RMSLE indicates better performance.

The final production model is selected according to **cross-validation RMSLE**, rather than selecting a model based on performance on the final holdout set.

Additional reported metrics are:

```text
RMSE
MAE
R²
```

---

## Models Compared

Four regression algorithms are evaluated:

1. **Linear Regression**
2. **K-Nearest Neighbors Regression**
3. **Random Forest Regression**
4. **Gradient Boosting Regression**

Linear Regression serves as the basic regression baseline.

KNN uses scaled numerical features because the algorithm is distance-based.

Random Forest and Gradient Boosting are nonlinear tree-based ensemble approaches capable of capturing complex relationships between temporal, weather, and demand variables.

---

## Final Results

| Model | CV RMSLE | Test RMSLE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **0.4833** | **0.4115** | **80.43** | **53.13** | **0.8667** |
| KNN Regression | 0.5976 | 0.5335 | 122.35 | 83.59 | 0.6914 |
| Gradient Boosting | 0.7257 | 0.6278 | 96.87 | 68.24 | 0.8066 |
| Linear Regression | 1.2964 | 0.9813 | 112.48 | 83.87 | 0.7392 |

### Selected Model

The final production model is:

**Random Forest Regression**

Selected hyperparameters:

```text
n_estimators = 400
max_depth = None
min_samples_leaf = 1
```

Random Forest achieved the lowest cross-validation RMSLE and was therefore selected as the production model.

Its chronological holdout performance was:

```text
RMSLE = 0.4115
RMSE  = 80.43
MAE   = 53.13
R²    = 0.8667
```

---

## Implementation

### `src/data_understanding.py`

Performs exploratory data analysis and data-quality checks.

### `src/model_config.py`

Stores shared paths, feature definitions, mappings, and modeling configuration.

### `src/feature_engineering.py`

Implements the shared feature-engineering pipeline used by both training and inference.

### `src/model_training.py`

Defines:

```text
Linear Regression
KNN Regression
Random Forest Regression
Gradient Boosting Regression
```

and performs temporal cross-validation and hyperparameter tuning.

### `src/evaluation.py`

Contains evaluation utilities for:

```text
RMSLE
RMSE
MAE
R²
```

and generates model-comparison diagnostics.

### `src/train_models.py`

Runs the complete final modeling pipeline:

```text
Load data
→ engineer features
→ temporal split
→ train models
→ temporal cross-validation
→ evaluate holdout performance
→ select production model
→ save generated artifacts
```

### `src/preprocess_and_baseline.py`

Contains the earlier Linear Regression and Random Forest baseline experiment.

The authoritative final model-selection pipeline is:

```bash
python src/train_models.py
```

### `app/demo_app.py`

Provides the Streamlit interface for live bike-demand prediction.

The demo reuses the same feature-engineering logic as the training pipeline.

---

## Streamlit Demo

The interactive application allows a user to specify a bike-rental scenario.

The user selects a date, hour, weather condition, feels-like temperature, humidity, windspeed, and holiday status.

Calendar information such as:

```text
year
month
weekday
season
weekend status
working-day status
```

is derived consistently by the application.

The inputs are passed through the shared feature-engineering pipeline before prediction using the trained Random Forest production model.

Run:

```bash
streamlit run app/demo_app.py
```

The production model must first be generated by running:

```bash
python src/train_models.py
```

---

## Repository Structure

Only source code and code documentation are maintained in the GitHub repository.

```text
bike-rental-demand-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/
│   └── demo_app.py
│
├── scripts/
│   └── download_data.py
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

The following are generated or downloaded locally and are not required to be stored in Git:

```text
data/       → downloaded dataset
models/     → trained models and metric files
reports/    → generated visualizations and diagnostics
.venv/      → local Python environment
```

Project write-ups and presentation files are maintained separately from the source-code repository.

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DabbuguntaVenyaAnand/bike-rental-demand-forecasting.git
cd bike-rental-demand-forecasting
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

For Windows Git Bash:

```bash
source .venv/Scripts/activate
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

For Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the Dataset

```bash
python scripts/download_data.py
```

This creates:

```text
data/hour.csv
```

### 5. Run Exploratory Data Analysis

```bash
python src/data_understanding.py
```

### 6. Train and Evaluate the Models

```bash
python src/train_models.py
```

This performs temporal model comparison and generates the local production model and evaluation artifacts.

### 7. Launch the Demo

```bash
streamlit run app/demo_app.py
```

---

## Reproducing the Project

From a fresh clone:

```bash
python -m venv .venv
source .venv/Scripts/activate

python -m pip install -r requirements.txt
python scripts/download_data.py

python src/data_understanding.py
python src/train_models.py

streamlit run app/demo_app.py
```

---

## Reference

The project problem statement is based on:

**Jimmy Du, Rolland He, Zhivko Zhechev — _Forecasting Bike Rental Demand_, Stanford CS229 Project, 2014.**

Reference paper:

https://cs229.stanford.edu/proj2014/Jimmy%20Du%2C%20Rolland%20He%2C%20Zhivko%20Zhechev%2C%20Forecasting%20Bike%20Rental%20Demand.pdf

Dataset:

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
