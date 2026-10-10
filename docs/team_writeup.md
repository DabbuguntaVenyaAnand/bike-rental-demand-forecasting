# Forecasting Bike Rental Demand — Project Write-up (UE24CS352A Mini-Project)

**Team:** Dabbugunta Venya Anand (PES1UG24AM074) & Dhruv Talavat (PES1UG24AM087) — Problem No. 87

## 1. Problem Statement

Urban bike-sharing systems must match bicycle supply with hour-to-hour demand: too few bikes at rush hour means lost rides, while surplus bikes waste dock space and rebalancing effort. We formulate this as a supervised regression problem — predicting the total number of bike rentals (`cnt`) in a given hour of the Capital Bikeshare system (Washington, D.C.) from calendar information (hour, weekday, month, season, holiday/working day) and weather conditions (temperature, feels-like temperature, humidity, windspeed, weather situation). Accurate hourly forecasts let operators pre-position bikes before predictable peaks and plan dock capacity.

## 2. Dataset Details

We use the UCI Bike Sharing Dataset, hourly file (`hour.csv`): **17,379 hourly records covering 2011-01-01 to 2012-12-31**, with 17 columns. The target `cnt` (1–977 rentals/h, mean ≈ 189, right-skewed) equals `casual + registered`, so those two user-type columns are excluded from features to prevent target leakage. Predictors are calendar variables (`season`, `yr`, `mnth`, `hr`, `weekday`, `holiday`, `workingday`) and normalized weather variables (`weathersit` 1–4, `temp`, `atemp`, `hum`, `windspeed`). Checks found no missing values, no duplicate hours, and a strong `temp`–`atemp` collinearity (~0.99). EDA showed a bimodal weekday profile (peaks 7–9 AM and 5–7 PM), broad weekend daytime demand, summer > fall > spring > winter seasonality, and clear degrades under poor weather.

## 3. Approach

Featurization and evaluation follow the reference paper (Du, He & Zhechev, *Forecasting Bike Rental Demand*): **RMSLE** (log-scale, as in the Kaggle competition) is the headline metric, with RMSE/MAE/R² reported alongside. Preprocessing uses a `ColumnTransformer` (standard scaling for numerics, one-hot encoding with `handle_unknown='ignore'` for categoricals). Leakage-safe engineered features include peak-hour buckets (weekday 7–9 AM & 5–7 PM; weekend 10 AM–6 PM), quantile temperature buckets, cyclic sin/cos encodings of hour/weekday/month, and a rush-hour flag; raw `temp` is dropped in favor of `atemp`. Data is split **temporally** (train: 2011-01-01 → 2012-08-07; test: 2012-08-08 → 2012-12-31; 13,915 / 3,464 rows) so the model never trains on the future. Baselines: Linear Regression and a grid-searched Random Forest (`TimeSeriesSplit(4)`), with advanced models (GBM, SVR, conditional inference trees, stacking) from the paper explored as the modeling half.

<!-- PERSON B: add model-training/comparison paragraphs here (GBM/SVR/CTree/stacking, tuning, ensemble) -->

## 4. Implementation Overview

```
data/hour.csv                    raw dataset
src/data_understanding.py        schema/quality checks + 12 EDA figures (reports/)
src/preprocess_and_baseline.py   features, temporal split, baseline models, metrics
app/demo_app.py                  Streamlit demo: input side → model → prediction
models/                          baseline_model.pkl + metrics/diagnostics JSON
docs/, reports/, scripts/        write-up, slide deck, figures, slide generator
```

Reproduce everything with three commands (see README): `python src/data_understanding.py`, `python src/preprocess_and_baseline.py`, `streamlit run app/demo_app.py`. The demo app collects season/day/hour/weather inputs, applies the same featurization as training (including matching temperature-bucket bin edges), and displays the model's predicted hourly rentals.

<!-- PERSON B: add inference-pipeline / best-model selection paragraphs here -->

## 5. Conclusions

**Person A:** Hour of day is by far the strongest predictor, with weather variables contributing a smaller but consistent effect. The paper-aligned Random Forest baseline achieves **RMSLE 0.410, RMSE 80.0, R² 0.868** on the held-out temporal test period, substantially beating Linear Regression (RMSLE 1.045, R² 0.698) and reproducing the reference paper's finding that tree-based models dominate this data. The full EDA/preprocessing/baseline pipeline is reproducible from the README commands and provides the benchmark the advanced models must beat.

<!-- PERSON B: add final model comparison + overall conclusion here -->
