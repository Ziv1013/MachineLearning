from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DEFAULT = Path("data/raw/uci_power/household_power_consumption.txt")
OUT_DIR_DEFAULT = Path("data/processed")
WEATHER_DEFAULTS = [
    Path("data/raw/weather/MENSQ_92_previous-1950-2024.csv.gz"),
    Path("data/raw/weather/MENSQ_75_previous-1950-2024.csv.gz"),
]


SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
]
MEAN_COLUMNS = ["voltage", "global_intensity"]
WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
EPSILON = 1e-6


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_minute_data(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_path}. Download and extract the UCI dataset first."
        )

    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
    )
    df = _normalize_columns(df)
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    numeric_cols = [
        "global_active_power",
        "global_reactive_power",
        "voltage",
        "global_intensity",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Formula given in the assessment. It is computed per minute before daily
    # aggregation so the feature remains physically meaningful.
    df["sub_metering_remainder"] = (
        df["global_active_power"] * 1000.0 / 60.0
        - (df["sub_metering_1"] + df["sub_metering_2"] + df["sub_metering_3"])
    )

    return df.dropna(subset=["datetime"]).sort_values("datetime")


def load_monthly_weather(weather_paths: list[Path]) -> pd.DataFrame | None:
    frames = []
    usecols = ["AAAAMM", *WEATHER_COLUMNS]
    for path in weather_paths:
        if not path.exists():
            continue
        weather = pd.read_csv(path, sep=";", usecols=lambda col: col in usecols)
        if "AAAAMM" not in weather.columns:
            continue
        for col in WEATHER_COLUMNS:
            if col not in weather.columns:
                weather[col] = np.nan
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
        weather["AAAAMM"] = pd.to_numeric(weather["AAAAMM"], errors="coerce")
        frames.append(weather[usecols])

    if not frames:
        return None

    monthly = pd.concat(frames, ignore_index=True)
    monthly = monthly.dropna(subset=["AAAAMM"])
    monthly["year_month"] = monthly["AAAAMM"].astype(int).astype(str)
    monthly = monthly.groupby("year_month", as_index=False)[WEATHER_COLUMNS].median()
    monthly[WEATHER_COLUMNS] = monthly[WEATHER_COLUMNS].interpolate().ffill().bfill()
    return monthly


def add_weather_features(daily: pd.DataFrame, monthly_weather: pd.DataFrame | None) -> None:
    if monthly_weather is None:
        return

    daily["year_month"] = daily.index.strftime("%Y%m")
    merged = daily[["year_month"]].merge(monthly_weather, on="year_month", how="left")
    for col in WEATHER_COLUMNS:
        values = pd.to_numeric(merged[col], errors="coerce")
        daily[col] = values.interpolate().ffill().bfill().to_numpy()
    daily.drop(columns=["year_month"], inplace=True)


def build_daily_table(
    df: pd.DataFrame,
    min_daily_points: int = 720,
    monthly_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["datetime"].dt.floor("D")

    grouped = df.groupby("date")
    daily_sum = grouped[SUM_COLUMNS].sum(min_count=1)
    daily_mean = grouped[MEAN_COLUMNS].mean()
    valid_minutes = grouped["global_active_power"].count().rename("valid_minutes")
    daily = pd.concat([daily_sum, daily_mean], axis=1).reset_index()
    daily = daily.merge(valid_minutes.reset_index(), on="date", how="left")

    value_cols = SUM_COLUMNS + MEAN_COLUMNS
    low_coverage = daily["valid_minutes"] < min_daily_points
    daily.loc[low_coverage, value_cols] = np.nan

    daily = daily.sort_values("date").reset_index(drop=True)
    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(all_days)
    daily.index.name = "date"

    daily[value_cols] = daily[value_cols].interpolate(method="time").ffill().bfill()
    daily["valid_minutes"] = daily["valid_minutes"].fillna(0).astype(int)

    idx = daily.index
    daily["day_of_week"] = idx.dayofweek
    daily["month"] = idx.month
    daily["day_of_year"] = idx.dayofyear
    daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7.0)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7.0)
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12.0)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12.0)
    daily["doy_sin"] = np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(float)
    daily["quarter"] = idx.quarter
    daily["quarter_sin"] = np.sin(2 * np.pi * daily["quarter"] / 4.0)
    daily["quarter_cos"] = np.cos(2 * np.pi * daily["quarter"] / 4.0)

    add_weather_features(daily, monthly_weather)
    add_engineered_features(daily)

    daily = daily.reset_index()
    return daily


def add_engineered_features(daily: pd.DataFrame) -> None:
    power = daily["global_active_power"]
    reactive = daily["global_reactive_power"]
    intensity = daily["global_intensity"]
    sub_total = (
        daily["sub_metering_1"]
        + daily["sub_metering_2"]
        + daily["sub_metering_3"]
        + daily["sub_metering_remainder"]
    )

    daily["reactive_active_ratio"] = reactive / (power.abs() + EPSILON)
    daily["intensity_power_ratio"] = intensity / (power.abs() + EPSILON)
    daily["metering_total"] = sub_total
    daily["metering_1_share"] = daily["sub_metering_1"] / (sub_total.abs() + EPSILON)
    daily["metering_2_share"] = daily["sub_metering_2"] / (sub_total.abs() + EPSILON)
    daily["metering_3_share"] = daily["sub_metering_3"] / (sub_total.abs() + EPSILON)
    daily["metering_remainder_share"] = daily["sub_metering_remainder"] / (
        sub_total.abs() + EPSILON
    )

    for lag in [1, 2, 7, 14, 30]:
        daily[f"power_lag_{lag}"] = power.shift(lag).fillna(power)
        daily[f"reactive_lag_{lag}"] = reactive.shift(lag).fillna(reactive)
        daily[f"intensity_lag_{lag}"] = intensity.shift(lag).fillna(intensity)

    for window in [3, 7, 14, 30]:
        daily[f"power_roll_mean_{window}"] = power.rolling(window, min_periods=1).mean()
        daily[f"power_roll_std_{window}"] = (
            power.rolling(window, min_periods=2).std().fillna(0.0)
        )
        daily[f"power_roll_min_{window}"] = power.rolling(window, min_periods=1).min()
        daily[f"power_roll_max_{window}"] = power.rolling(window, min_periods=1).max()
        daily[f"reactive_roll_mean_{window}"] = reactive.rolling(
            window, min_periods=1
        ).mean()
        daily[f"intensity_roll_mean_{window}"] = intensity.rolling(
            window, min_periods=1
        ).mean()

    daily["power_diff_1"] = power.diff(1).fillna(0.0)
    daily["power_diff_7"] = (power - power.shift(7)).fillna(0.0)
    daily["power_ema_7"] = power.ewm(span=7, adjust=False).mean()
    daily["power_ema_30"] = power.ewm(span=30, adjust=False).mean()
    daily["power_vs_roll7"] = power - daily["power_roll_mean_7"]
    daily["power_vs_roll30"] = power - daily["power_roll_mean_30"]


def write_split_files(daily: pd.DataFrame, out_dir: Path, test_days: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(daily) <= test_days + 90:
        raise ValueError("Not enough daily rows for a 90-day input plus requested test span.")

    train = daily.iloc[: -test_days].copy()
    test = daily.iloc[-test_days:].copy()

    daily.to_csv(out_dir / "daily_power.csv", index=False)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate UCI minute data to daily data.")
    parser.add_argument("--raw-path", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--weather-paths", type=Path, nargs="*", default=WEATHER_DEFAULTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--min-daily-points", type=int, default=720)
    args = parser.parse_args()

    minute = load_minute_data(args.raw_path)
    monthly_weather = load_monthly_weather(args.weather_paths)
    daily = build_daily_table(
        minute,
        min_daily_points=args.min_daily_points,
        monthly_weather=monthly_weather,
    )
    write_split_files(daily, args.out_dir, args.test_days)

    print(f"Wrote daily data: {args.out_dir / 'daily_power.csv'} ({len(daily)} rows)")
    print(f"Wrote train/test split with last {args.test_days} days as test.")
    if monthly_weather is None:
        print("Weather files were not found; generated data without weather columns.")
    else:
        print(f"Added weather columns: {', '.join(WEATHER_COLUMNS)}")


if __name__ == "__main__":
    main()
