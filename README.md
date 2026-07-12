# Household Power Forecasting Course Project

This repository implements the 2026 machine learning course assessment for
multivariate household power time-series forecasting.

The code trains and evaluates three model families with leakage-safe weather,
historical lag, rolling-statistic, ratio, difference, and exponential
moving-average features:

1. LSTM
2. Transformer encoder
3. BayesFormerUQ, an improved Transformer-style model with Bayesian dropout and
   quantile regression uncertainty outputs

Both required horizons are supported:

- short-term: use the past 90 days to predict the next 90 days
- long-term: use the past 90 days to predict the next 365 days

Each model-horizon pair is repeated over 5 random seeds. The output includes
MSE/MAE mean and standard deviation, raw per-seed metrics, and prediction-vs-ground-truth
figures.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Data

The current workspace uses the public UCI Individual household electric power
consumption dataset:

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Expected raw file:

```text
data/raw/uci_power/household_power_consumption.txt
```

Weather files are loaded from Météo-France monthly climate data when available:

```text
data/raw/weather/MENSQ_92_previous-1950-2024.csv.gz
data/raw/weather/MENSQ_75_previous-1950-2024.csv.gz
```

The public links do not provide a ready-made train/test split, so preprocessing
uses chronological splitting and keeps the last 365 days as the test set.

## Run

Generate daily data and train/test CSV files:

```powershell
python -m src.preprocess
```

Run the full 5-seed experiment:

```powershell
python -m src.train --epochs 30 --patience 30 --seeds 42 43 44 45 46 --d-model 64 --batch-size 128 --target-mode level --lr 0.001
```

Outputs:

```text
data/processed/daily_power.csv
data/processed/train.csv
data/processed/test.csv
outputs/metrics.csv
outputs/summary.csv
outputs/figures/*.png
reports/report.md
```

## Notes

The assignment asks for author contribution, research field, a GitHub link, and
screenshots in the final report. Fill those personal fields before submission.
