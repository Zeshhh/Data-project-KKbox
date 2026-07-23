# KKBox Churn Analytics — End-to-End Retention Strategy

KKBox is a music streaming company based in Taiwan. This project takes their raw subscriber data and turns it into a working retention strategy: who is going to cancel, why, which of them are worth targeting, and what a campaign would cost and return.

**Dataset:** KKBox WSDM 2018 Churn Prediction Challenge (Kaggle) — 1,082,190 subscribers, 5 raw source files.

---

## What I Found

The main finding is counterintuitive: **churn is a billing commitment problem, not an engagement problem.** Subscribers who listen to music every day but are on short-term plans without auto-renewal churn at nearly the same rate as inactive subscribers. Plan structure outweighs listening behaviour in every model and statistical test.

The model (XGBoost, AUC 0.973) confirms this — plan commitment and auto-renewal history rank as the two strongest predictors, above all listening metrics combined.

On the business side: **30.3M TWD in monthly revenue is at risk** from high-probability churners. Of the 145,143 high-risk subscribers, 81,253 are genuinely persuadable — recently active, but structurally unprotected by their plan. A free-month retention offer breaks even between 0.31 and 1.04 months of retained subscription, depending on whether you account for cost as operating expenditure (40 TWD) or foregone revenue (149 TWD).

---

## Notebooks

| Notebook | What it does | Key output |
|---|---|---|
| 📦 [01 — Data Pipeline](https://github.com/Zeshhh/Data-project-KKbox/blob/main/notebooks/01_data_pipeline.ipynb) | Loads 5 raw source files, merges members, transactions, and user logs into one clean dataset | `master.csv` — 1,082,190 × 24 |
| 🔍 [02 — Exploratory Analysis](https://github.com/Zeshhh/Data-project-KKbox/blob/main/notebooks/02_eda.ipynb) | Compares churners vs retained subscribers across demographics, listening behaviour, and plan structure | Validated churn signal per feature |
| 📅 [03 — Cohort Retention](https://github.com/Zeshhh/Data-project-KKbox/blob/main/notebooks/03_retention_analysis.ipynb) | Tracks retention curves across 27 monthly registration cohorts (2015–2017) | Retention heatmap |
| 📉 [04 — Survival Analysis](https://github.com/Zeshhh/Data-project-KKbox/blob/main/notebooks/04_survival_analysis.ipynb) | Kaplan-Meier curves by subscriber segment, log-rank test on auto-renewal, RFM segmentation | 5 behavioural segments |
| ⚙️ [05 — Feature Engineering](https://github.com/Zeshhh/Data-project-KKbox/blob/main/notebooks/05_feature_engineering.ipynb) | Builds 9 behavioural features (rate, trajectory, composite), validates signal, removes redundant ones | `master_fe.csv` — 1,082,190 × 11 |
| 🤖 [06 — Churn Model](https://github.com/Zeshhh/Data-project-KKbox/blob/main/notebooks/06_xgboost_shap.ipynb) | XGBoost classifier with SHAP explainability — what drives each prediction | `master_scored.csv` — churn probability per subscriber |
| 💰 [07 — Financial Impact](https://github.com/Zeshhh/Data-project-KKbox/blob/main/notebooks/07_financial_impact.ipynb) | Revenue at risk, three retention scenarios, break-even chart, intervention list | 81,253 persuadable subscribers |
| 🟢 [08 — Health Scoring](https://github.com/Zeshhh/Data-project-KKbox/blob/main/notebooks/08_scoring.ipynb) | Classifies all subscribers as Red / Amber / Green based on churn probability | `master_health.csv` |

---

## Results

| Metric | Value |
|---|---|
| Model AUC-ROC | 0.973 |
| High-risk subscribers (Red) | 145,143 (13.4%) |
| Monthly revenue at risk | 30.3M TWD |
| Persuadable subscribers | 81,253 |
| Campaign break-even (operating cost model) | 0.31 months retained |
| Campaign break-even (foregone revenue model) | 1.04 months retained |

---

## Stack

Python · pandas · XGBoost · SHAP · lifelines · scikit-learn · matplotlib · seaborn · Jupyter

---

## How to Run

```bash
git clone https://github.com/Zeshhh/Data-project-KKbox.git
cd Data-project-KKbox

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download the KKBox dataset from Kaggle and place files in data/raw/
# Then run notebooks 01–08 in order
```

Dataset: [KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) — requires a free Kaggle account.
