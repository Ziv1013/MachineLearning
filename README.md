# Household Power Forecasting Course Project

This repository implements the 2026 machine learning course assessment for
multivariate household power time-series forecasting.

The code trains and evaluates three model families with monthly weather,
historical lag, rolling-statistic, ratio, difference, and exponential
moving-average features:

1. LSTM
2. Transformer encoder
3. WeatherBayesFormerUQ, an improved model with trend-residual decomposition,
   weekly local patch convolution, weather gating, Bayesian dropout, and
   quantile regression uncertainty outputs

Both required horizons are supported:

- short-term: use the past 90 days to predict the next 90 days
- long-term: use the past 90 days to predict the next 365 days

Each model-horizon pair is repeated over 5 random seeds. The output includes
normalized MSE/MAE, inverse-standardized MSE/MAE/RMSE on the required daily
total-power scale, raw per-seed metrics, and prediction-vs-ground-truth figures.

The 90-day metrics use 276 overlapping rolling-origin test windows. Each figure
shows the pointwise mean prediction over seeds 42--46 for their common first
window, with a +/-1 sample-standard-deviation band across seeds. The report also
includes a three-model plot of all 276 rolling forecasts, where each pale line
is a five-seed mean for one forecast origin. The 365-day test split has only one
complete window, so no duplicate all-window plot is created.

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

The UCI household is located in Sceaux, so monthly weather values are selected
from the nearest station with an available observation for each field. Monthly
aggregates are shifted forward one month so every feature was available at the
forecast origin.

The public links do not provide a ready-made train/test split, so preprocessing
uses chronological splitting and keeps the last 365 days as the test set.

The required daily aggregation is applied directly:

- `global_active_power` and `global_reactive_power` are summed by day
- the sub-metering fields are summed by day
- voltage and current remain daily means in V and A

The prediction target remains the course-defined daily total
`global_active_power`; it is not converted to energy in kWh.

Empty minute measurements are causally forward-filled before aggregation.
Low-coverage days reuse the preceding daily observation, and the two partial
boundary dates are excluded because they do not contain 1440 timestamps.

## Run

Generate daily data and train/test CSV files:

```powershell
python -m src.preprocess
```

Run the full 5-seed experiment:

```powershell
python -m src.train --epochs 30 --patience 30 --seeds 42 43 44 45 46 --d-model 64 --batch-size 128 --target-mode level --lr 0.001 --mc-samples 30
```

Outputs:

```text
data/processed/daily_power.csv
data/processed/train.csv
data/processed/test.csv
outputs/metrics.csv
outputs/summary.csv
outputs/figures/*.png
outputs/predictions/*.npz
outputs/checkpoints/*.pt
reports/report.md
```

## Notes

Models are trained on a target standardized using the fit segment only. The
reported normalized errors are convenient for model comparison, while the
inverse-standardized metrics and plots use the assignment's daily total-power
scale. Normalized values from a different split or scaler are not directly
comparable.

The assignment asks for author contribution, research field, a GitHub link, and
screenshots in the final report. Fill those personal fields before submission.
