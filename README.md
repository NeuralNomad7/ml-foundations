# ML Foundations

Three classic supervised-learning studies on tabular data: the groundwork under my later
work in robotics and physical AI. Each folder is a self-contained notebook.

| Study | Task | Data | Recorded result |
|---|---|---|---|
| [Insurance cost risk](insurance-cost-risk/) | Binary classification | [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance), 1,338 rows (included) | Random Forest 0.93 accuracy on a 20% hold-out |
| [Satellite health](satellite-health/) | Binary classification from telemetry | Raw ("unclean") telemetry CSV (included) | Random Forest 0.84 mean 5-fold CV accuracy |
| [House prices](house-prices/) | Regression | Kaggle *House Prices* competition (download separately) | Notebook compares four models; see folder README |

Every number above appears in the notebook's saved outputs.

## Insurance cost risk

Can demographics and health habits flag members likely to have above-median medical
charges? The label is `charges > median`, and `charges` is removed from the features
before training, so the models can't simply read the label back.

- Features: age, BMI and a BMI band, children, smoker, region, sex.
- Models: Logistic Regression (0.91), Random Forest (0.93), XGBoost (0.93) accuracy on
  268 held-out rows.
- Smoking status and BMI are the strongest predictors.
- Caveats: one random split, no cross-validation, and the ROC-AUC column is computed
  from hard labels rather than predicted probabilities, so it understates the curve.
  The project originally framed the label as "churn"; it is really a cost-risk label.

## Satellite health

Classifies a satellite as healthy or unhealthy from seven telemetry channels: time since
launch, orbital altitude, battery voltage, solar-panel temperature, attitude-control
error, data rate and thermal-control status. Most of the work is cleaning the raw file:
median fill for missing values, duplicate removal and capping outliers at the 99th
percentile. Logistic Regression, Random Forest, SVC and SGD are compared, and Random
Forest reaches 0.84 mean accuracy in 5-fold cross-validation.

- Caveat: the median fill and the 99th-percentile caps are computed on the full dataset
  before the split. Fitting them on the training fold only would be stricter.

## Run

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
jupyter notebook
```

## History

This repository combines three earlier repositories (Customer-Churn, House-price,
Satellite-health) into one.
