# Forecasting Bike Rental Demand

> **Machine Learning Mini-Project — Project No. 87**

A machine learning project to forecast bike rental demand using temporal, weather, and related features. Featurization and metric choice are aligned with the reference paper (Du, He, Zhechev — *Forecasting Bike Rental Demand*, Kaggle bike-sharing competition).

---

## Team Members

| Sl. No. | Student Name | SRN / ID |
| :---: | :--- | :--- |
| 1 | Dabbugunta Venya Anand | PES1UG24AM074 |
| 2 | Dhruv Talavat | PES1UG24AM087 |

---

## Task split

**Person A (Dhruv Talavat)** — dataset understanding, EDA, preprocessing, feature engineering, baseline models, problem/dataset/EDA write-up & slides, demo input side — on branch `feature/person-a-data-pipeline`.

**Person B (Dabbugunta Venya Anand)** — advanced models (GBM, SVR, stacking — cf. paper §3), model comparison & evaluation, best-model selection, inference pipeline, models/results write-up & slides, demo prediction side.

**Both** — methodology slide, README, final integration/testing, demo/conclusion, viva prep.

---

## Repo layout (Person A)

```
data/            hour.csv (UCI Bike Sharing, hourly)
src/
  data_understanding.py       dataset summary + 12 EDA figures
  preprocess_and_baseline.py  paper-aligned features, temporal split, 2 baselines
app/
  demo_app.py                 Streamlit input/interface side
docs/
  person_a_writeup.md         Person A technical write-up
  person_a_slides.pptx        Person A presentation
  reference_paper_text.txt    extracted text of the reference paper
reports/                      EDA + baseline figures (PNG)
models/                       baseline_model.pkl + metrics/diagnostics JSON
scripts/
  make_person_a_slides.py     regenerates person_a_slides.pptx
requirements.txt
```

## How to run (Person A part)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows Git Bash
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/Mac

# 0) Fetch the dataset (data/ is gitignored; one-time step)
.venv/Scripts/python scripts/download_data.py

# 1) Dataset understanding + EDA -> writes reports/*.png
.venv/Scripts/python src/data_understanding.py

# 2) Preprocessing + feature engineering + baseline models
#    -> writes models/baseline_model.pkl, models/baseline_metrics.json,
#       models/baseline_diagnostics.json, reports/13_*.png, reports/14_*.png
.venv/Scripts/python src/preprocess_and_baseline.py

# 3) Demo app (input/interface side)
.venv/Scripts/streamlit run app/demo_app.py
```

## Person A results (baselines, held-out temporal test set)

Primary metric is **RMSLE** (same as the Kaggle competition in the reference paper); RMSE/MAE/R² are reported for completeness.

| Model | RMSLE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 1.045 | 121.13 | 91.05 | 0.698 |
| Random Forest (best) | **0.410** | **79.98** | **52.84** | **0.868** |

Best model: Random Forest (`max_depth=18, n_estimators=400, min_samples_leaf=1`), selected via `TimeSeriesSplit(4)` grid search, evaluated on a held-out test period (2012-08-08 → 2012-12-31). Split boundary: 2012-08-07.

Advanced models (`src/advanced_models.py`, same features/split): Gradient Boosting RMSLE 0.502 / RMSE 65.3 / R² 0.912, RBF-SVR RMSLE 0.558, and a 50/50 GBM+RF blend at **RMSLE 0.403 / R² 0.896** (best by RMSLE; see `models/advanced_metrics.json`).

### Paper alignment summary

- **Metric:** RMSLE as headline, like the paper's Kaggle competition.
- **Features:** discretized temperature buckets, weekday/weekend peak-hour buckets (weekday 7–9 AM & 5–7 PM, weekend 10 AM–6 PM), month kept over coarse season, raw `temp` dropped in favor of `atemp` (collinearity ≈0.99).
- **Baseline pattern:** tree models ≫ linear models, reproducing the paper's finding (their RF CV-RMSLE 0.503, CTree 0.460 vs Linear 1.044).
- **Split:** paper's Kaggle split is not time-ordered; we use the temporal split the paper itself recommends as future work.

## Notes for Person B

- `models/baseline_model.pkl` is the fitted baseline pipeline; compare your advanced models against `models/baseline_metrics.json` **on RMSLE**, and reuse the temporal split (boundary 2012-08-07) for a fair comparison.
- `casual` and `registered` sum exactly to the target `cnt` — exclude them from any feature set (target leakage). The reference paper's §5.5 also found regressing on them separately is worse than modeling `cnt` directly.
- See `docs/person_a_writeup.md` §8 for other handoff notes (severe-weather extrapolation risk, RF limits, time-series extension from paper §7.3).
