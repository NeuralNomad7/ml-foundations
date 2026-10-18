# House prices (regression)

Predicts `SalePrice` from 79 features of the Ames, Iowa housing data, as in Kaggle's
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Data

The competition's rules don't allow redistributing its files, so they're not in this repo.
Download `train.csv` and `test.csv` from the competition's Data tab and put them in this
folder, next to the notebook.

## What the notebook does

- Exploratory plots of every numeric and categorical feature against `SalePrice`.
- Outlier removal with the reasoning noted inline (for example lots over 100,000 sq ft).
- Drops sparse columns (`PoolQC`, `MiscFeature`, `Alley`, `Fence`) and redundant ones.
- A `ColumnTransformer` pipeline: mean imputation and scaling for numeric features,
  ordinal encoding for quality grades, one-hot encoding for nominal categories.
- Linear Regression, Ridge, Random Forest and XGBoost, with `GridSearchCV` tuning.
