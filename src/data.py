from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


TARGET_COLUMN = "global_active_power"
DEFAULT_FEATURE_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
    "reactive_active_ratio",
    "intensity_power_ratio",
    "metering_total",
    "power_lag_1",
    "power_lag_7",
    "power_lag_30",
    "power_roll_mean_7",
    "power_roll_std_7",
    "power_roll_mean_30",
    "power_roll_std_30",
    "power_diff_1",
    "power_diff_7",
    "power_ema_7",
    "power_ema_30",
    "power_vs_roll7",
    "power_vs_roll30",
]
FUTURE_FEATURE_COLUMNS = ["dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]


class WindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        future_features: np.ndarray,
        target: np.ndarray,
        baseline: np.ndarray,
        input_len: int,
        horizon: int,
        starts: np.ndarray,
        target_mode: str,
        baseline_mode: str,
    ) -> None:
        self.features = features
        self.future_features = future_features
        self.target = target
        self.baseline = baseline
        self.input_len = input_len
        self.horizon = horizon
        self.starts = starts.astype(np.int64)
        self.target_mode = target_mode
        self.baseline_mode = baseline_mode

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        input_end = start + self.input_len
        target_end = input_end + self.horizon

        x = self.features[start:input_end]
        future_x = self.future_features[input_end:target_end]
        y_level = self.target[input_end:target_end, 0]

        if self.baseline_mode == "seasonal":
            base = self.baseline[input_end:target_end, 0]
        elif self.baseline_mode == "last":
            base = np.repeat(self.target[input_end - 1, 0], self.horizon)
        elif self.baseline_mode == "mean":
            base = np.repeat(self.target[start:input_end, 0].mean(), self.horizon)
        elif self.baseline_mode == "zero":
            base = np.zeros(self.horizon, dtype=np.float32)
        else:
            raise ValueError(f"Unknown baseline mode: {self.baseline_mode}")

        if self.target_mode == "residual":
            y = y_level - base
        elif self.target_mode == "level":
            y = y_level
        else:
            raise ValueError(f"Unknown target mode: {self.target_mode}")

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(future_x, dtype=torch.float32),
            torch.tensor(base, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


@dataclass
class PreparedData:
    train_ds: WindowDataset
    val_ds: WindowDataset
    test_ds: WindowDataset
    feature_columns: list[str]
    future_columns: list[str]
    target_scaler: StandardScaler
    test_dates: list[str]
    input_dim: int
    future_dim: int
    target_mode: str
    baseline_mode: str


def load_daily_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Daily data not found: {path}. Run python -m src.preprocess first.")
    df = pd.read_csv(path, parse_dates=["date"], engine="python")
    return df.sort_values("date").reset_index(drop=True)


def _build_windows(
    features: np.ndarray,
    future_features: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    input_len: int,
    horizon: int,
    starts: np.ndarray,
    target_mode: str,
    baseline_mode: str,
) -> WindowDataset:
    return WindowDataset(
        features=features,
        future_features=future_features,
        target=target,
        baseline=baseline,
        input_len=input_len,
        horizon=horizon,
        starts=starts,
        target_mode=target_mode,
        baseline_mode=baseline_mode,
    )


def prepare_data(
    daily_path: Path,
    input_len: int,
    horizon: int,
    test_days: int,
    val_fraction: float = 0.15,
    target_mode: str = "residual",
    baseline_mode: str = "seasonal",
) -> PreparedData:
    df = load_daily_data(daily_path)
    feature_columns = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    future_columns = [c for c in FUTURE_FEATURE_COLUMNS if c in df.columns]
    train_end = len(df) - test_days
    if train_end <= input_len + horizon:
        raise ValueError("Training split is too short for the requested input and horizon.")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(df.loc[: train_end - 1, feature_columns])
    target_scaler.fit(df.loc[: train_end - 1, [TARGET_COLUMN]])

    features = feature_scaler.transform(df[feature_columns]).astype(np.float32)
    future_features = feature_scaler.transform(df[feature_columns])[
        :, [feature_columns.index(c) for c in future_columns]
    ].astype(np.float32)
    target = target_scaler.transform(df[[TARGET_COLUMN]]).astype(np.float32)
    baseline = build_seasonal_baseline(df, train_end, target_scaler).astype(np.float32)

    train_starts = np.arange(0, train_end - input_len - horizon + 1)
    split = max(1, int(len(train_starts) * (1.0 - val_fraction)))
    fit_starts = train_starts[:split]
    val_starts = train_starts[split:]
    if len(val_starts) == 0:
        val_starts = train_starts[-1:]
        fit_starts = train_starts[:-1]

    first_test_start = train_end - input_len
    last_test_start = len(df) - input_len - horizon
    test_starts = np.arange(first_test_start, last_test_start + 1)
    if len(test_starts) == 0:
        raise ValueError("Test split is too short for requested horizon.")

    train_ds = _build_windows(
        features,
        future_features,
        target,
        baseline,
        input_len,
        horizon,
        fit_starts,
        target_mode,
        baseline_mode,
    )
    val_ds = _build_windows(
        features,
        future_features,
        target,
        baseline,
        input_len,
        horizon,
        val_starts,
        target_mode,
        baseline_mode,
    )
    test_ds = _build_windows(
        features,
        future_features,
        target,
        baseline,
        input_len,
        horizon,
        test_starts,
        target_mode,
        baseline_mode,
    )

    first_target_start = int(test_starts[0] + input_len)
    test_dates = (
        df.loc[first_target_start : first_target_start + horizon - 1, "date"]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    return PreparedData(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        feature_columns=feature_columns,
        future_columns=future_columns,
        target_scaler=target_scaler,
        test_dates=test_dates,
        input_dim=len(feature_columns),
        future_dim=len(future_columns),
        target_mode=target_mode,
        baseline_mode=baseline_mode,
    )


def inverse_target(target_scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, 1)
    restored = target_scaler.inverse_transform(flat).reshape(values.shape)
    return restored


def build_seasonal_baseline(
    df: pd.DataFrame,
    train_end: int,
    target_scaler: StandardScaler,
) -> np.ndarray:
    train = df.iloc[:train_end].copy()
    global_mean = float(train[TARGET_COLUMN].mean())
    by_doy = train.groupby(train["date"].dt.dayofyear)[TARGET_COLUMN].mean()
    baseline = np.array(
        [by_doy.get(day, global_mean) for day in df["date"].dt.dayofyear],
        dtype=np.float32,
    ).reshape(-1, 1)
    baseline_df = pd.DataFrame(baseline, columns=[TARGET_COLUMN])
    return target_scaler.transform(baseline_df)
