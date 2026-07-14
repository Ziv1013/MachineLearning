from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

from .data import PreparedData, inverse_target, prepare_data
from .models import build_model


MODEL_NAMES = ["lstm", "transformer", "weather_bayesformer_uq"]
HORIZONS = [90, 365]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    median_index: int,
) -> torch.Tensor:
    errors = target.unsqueeze(-1) - pred
    losses = torch.maximum((quantiles - 1.0) * errors, quantiles * errors)
    median = pred[:, :, median_index]
    return losses.mean() + 0.25 * nn.functional.mse_loss(median, target)


def median_prediction(output: torch.Tensor, model: nn.Module) -> torch.Tensor:
    if getattr(model, "is_quantile_model", False):
        return output[:, :, int(getattr(model, "median_index", output.shape[-1] // 2))]
    return output


def make_loss_fn(model: nn.Module):
    if getattr(model, "is_quantile_model", False):
        quantiles = getattr(model, "quantiles")
        median_index = int(getattr(model, "median_index", len(quantiles) // 2))

        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return quantile_loss(pred, target, quantiles.to(pred.device), median_index)

        return loss_fn
    return nn.MSELoss()


def train_one(
    model: nn.Module,
    data: PreparedData,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> tuple[nn.Module, float, int, int]:
    model.to(device)
    train_loader = DataLoader(data.train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data.val_ds, batch_size=batch_size, shuffle=False)
    loss_fn = make_loss_fn(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    epochs_ran = 0
    stale = 0

    for _epoch in range(1, epochs + 1):
        epochs_ran = _epoch
        model.train()
        for x, future_x, _baseline, y in train_loader:
            x = x.to(device)
            future_x = future_x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, future_x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = _epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_epoch, epochs_ran


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for x, future_x, _baseline, y in loader:
        x = x.to(device)
        future_x = future_x.to(device)
        y = y.to(device)
        losses.append(loss_fn(model(x, future_x), y).item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(
    model: nn.Module,
    ds,
    batch_size: int,
    device: torch.device,
    target_mode: str,
    return_aux: bool = False,
    mc_samples: int = 1,
):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    use_mc = bool(getattr(model, "is_quantile_model", False) and mc_samples > 1)
    model.eval()
    if use_mc:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    preds = []
    truths = []
    lowers = []
    uppers = []
    baselines = []
    inputs = []
    for x, future_x, baseline, y in loader:
        x_device = x.to(device)
        future_device = future_x.to(device)
        if use_mc:
            outputs = torch.stack(
                [model(x_device, future_device) for _ in range(mc_samples)],
                dim=0,
            )
            quantiles = getattr(model, "quantiles")
            median_index = int(getattr(model, "median_index"))
            lower_index = int(torch.argmin(torch.abs(quantiles - 0.1)).item())
            upper_index = int(torch.argmin(torch.abs(quantiles - 0.9)).item())
            median_samples = outputs[..., median_index]
            raw_pred_tensor = median_samples.mean(dim=0)
            raw_lower_tensor = torch.quantile(outputs[..., lower_index], 0.1, dim=0)
            raw_upper_tensor = torch.quantile(outputs[..., upper_index], 0.9, dim=0)
            raw_lower_tensor = torch.minimum(raw_lower_tensor, raw_pred_tensor)
            raw_upper_tensor = torch.maximum(raw_upper_tensor, raw_pred_tensor)
            raw_pred = raw_pred_tensor.cpu().numpy()
            raw_lower = raw_lower_tensor.cpu().numpy()
            raw_upper = raw_upper_tensor.cpu().numpy()
        else:
            raw_output = model(x_device, future_device)
            raw_pred = median_prediction(raw_output, model).cpu().numpy()
        baseline_np = baseline.numpy()
        y_np = y.numpy()
        if target_mode == "residual":
            preds.append(raw_pred + baseline_np)
            truths.append(y_np + baseline_np)
            if use_mc:
                lowers.append(raw_lower + baseline_np)
                uppers.append(raw_upper + baseline_np)
        elif target_mode == "level":
            preds.append(raw_pred)
            truths.append(y_np)
            if use_mc:
                lowers.append(raw_lower)
                uppers.append(raw_upper)
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
        if return_aux:
            baselines.append(baseline_np)
            inputs.append(x.numpy())
    pred_arr = np.concatenate(preds)
    truth_arr = np.concatenate(truths)
    model.eval()
    if return_aux:
        aux = {
            "baseline": np.concatenate(baselines),
            "input": np.concatenate(inputs),
        }
        if use_mc:
            aux["lower"] = np.concatenate(lowers)
            aux["upper"] = np.concatenate(uppers)
        return pred_arr, truth_arr, aux
    return pred_arr, truth_arr


def fit_affine_calibration(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_mean = pred.mean(axis=0)
    truth_mean = truth.mean(axis=0)
    pred_centered = pred - pred_mean
    truth_centered = truth - truth_mean
    var = np.mean(pred_centered * pred_centered, axis=0)
    cov = np.mean(pred_centered * truth_centered, axis=0)
    slope = np.divide(cov, var, out=np.ones_like(cov), where=var > 1e-8)
    slope = np.clip(slope, 0.25, 2.5)
    intercept = truth_mean - slope * pred_mean
    return slope.astype(np.float32), intercept.astype(np.float32)


def apply_affine_calibration(pred: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return pred * slope.reshape(1, -1) + intercept.reshape(1, -1)


def fit_linear_calibration(
    pred: np.ndarray,
    truth: np.ndarray,
    aux: dict[str, np.ndarray],
    ridge: float = 1e-3,
) -> np.ndarray:
    x = aux["input"]
    baseline = aux["baseline"]
    last_value = x[:, -1, 0]
    mean_value = x[:, :, 0].mean(axis=1)
    std_value = x[:, :, 0].std(axis=1)
    coefs = []
    for step in range(pred.shape[1]):
        design = np.column_stack(
            [
                pred[:, step],
                baseline[:, step],
                last_value,
                mean_value,
                std_value,
                np.ones(pred.shape[0]),
            ]
        )
        xtx = design.T @ design
        penalty = ridge * np.eye(xtx.shape[0])
        penalty[-1, -1] = 0.0
        coef = np.linalg.solve(xtx + penalty, design.T @ truth[:, step])
        coefs.append(coef)
    return np.stack(coefs).astype(np.float32)


def apply_linear_calibration(pred: np.ndarray, aux: dict[str, np.ndarray], coefs: np.ndarray) -> np.ndarray:
    x = aux["input"]
    baseline = aux["baseline"]
    last_value = x[:, -1, 0]
    mean_value = x[:, :, 0].mean(axis=1)
    std_value = x[:, :, 0].std(axis=1)
    calibrated = np.empty_like(pred)
    for step in range(pred.shape[1]):
        design = np.column_stack(
            [
                pred[:, step],
                baseline[:, step],
                last_value,
                mean_value,
                std_value,
                np.ones(pred.shape[0]),
            ]
        )
        calibrated[:, step] = design @ coefs[step]
    return calibrated


def plot_seed_ensemble_prediction(
    dates: list[str],
    truth: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    horizon: int,
    seeds: np.ndarray,
    out_path: Path,
    mc_lowers: np.ndarray | None = None,
    mc_uppers: np.ndarray | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_mean = predictions.mean(axis=0)
    prediction_std = predictions.std(axis=0, ddof=1) if len(predictions) > 1 else np.zeros_like(prediction_mean)

    plt.figure(figsize=(13.5, 5.4), dpi=170)
    x = np.arange(len(dates))
    if mc_lowers is not None and mc_uppers is not None:
        plt.fill_between(
            x,
            mc_lowers.mean(axis=0),
            mc_uppers.mean(axis=0),
            color="0.55",
            alpha=0.18,
            label="Mean MC-quantile interval",
        )
    plt.fill_between(
        x,
        prediction_mean - prediction_std,
        prediction_mean + prediction_std,
        color="tab:orange",
        alpha=0.22,
        label="5-seed mean +/- 1 std",
    )
    plt.plot(x, truth, label="Ground Truth", linewidth=2, color="tab:blue")
    plt.plot(
        x,
        prediction_mean,
        label="5-seed mean prediction",
        linewidth=2,
        color="tab:orange",
    )
    tick_count = min(8, len(dates))
    ticks = np.linspace(0, len(dates) - 1, tick_count, dtype=int)
    plt.xticks(ticks, [dates[i] for i in ticks], rotation=25, ha="right")
    display_name = {
        "lstm": "LSTM",
        "transformer": "Transformer",
        "weather_bayesformer_uq": "WeatherBayesFormerUQ",
    }.get(model_name, model_name)
    seed_text = ", ".join(str(seed) for seed in seeds)
    plt.title(f"{display_name} horizon={horizon}, five-seed ensemble ({seed_text})")
    plt.xlabel("Date")
    plt.ylabel("Daily total global active power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def aggregate_overlapping_windows(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if windows.ndim != 2:
        raise ValueError(f"Expected [window, horizon] values, got shape {windows.shape}.")
    window_count, horizon = windows.shape
    calendar_length = window_count + horizon - 1
    sums = np.zeros(calendar_length, dtype=np.float64)
    counts = np.zeros(calendar_length, dtype=np.int32)
    for window_index, values in enumerate(windows):
        target_slice = slice(window_index, window_index + horizon)
        sums[target_slice] += values
        counts[target_slice] += 1
    means = sums / counts
    squared_deviations = np.zeros(calendar_length, dtype=np.float64)
    for window_index, values in enumerate(windows):
        target_slice = slice(window_index, window_index + horizon)
        deviations = values.astype(np.float64) - means[target_slice]
        squared_deviations[target_slice] += deviations * deviations
    variances = np.zeros(calendar_length, dtype=np.float64)
    multiple = counts > 1
    variances[multiple] = squared_deviations[multiple] / (counts[multiple] - 1)
    return means.astype(np.float32), np.sqrt(np.maximum(variances, 0.0)).astype(np.float32)


def plot_all_rolling_windows(
    first_window_dates: list[str],
    truth_windows: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    horizon: int,
    seeds: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window_count = truth_windows.shape[0]
    calendar_length = window_count + horizon - 1
    calendar_dates = pd.date_range(
        start=pd.Timestamp(first_window_dates[0]),
        periods=calendar_length,
        freq="D",
    ).strftime("%Y-%m-%d").tolist()
    calendar_truth, truth_overlap_std = aggregate_overlapping_windows(truth_windows)
    if not np.allclose(truth_overlap_std, 0.0, rtol=0.0, atol=1e-5):
        raise ValueError("Overlapping test windows contain inconsistent ground truth values.")

    model_order = [name for name in MODEL_NAMES if name in predictions_by_model]
    figure, axes = plt.subplots(
        len(model_order),
        1,
        figsize=(14.5, 11.5),
        dpi=170,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    display_names = {
        "lstm": "LSTM",
        "transformer": "Transformer",
        "weather_bayesformer_uq": "WeatherBayesFormerUQ",
    }
    for axis, model_name in zip(axes, model_order):
        seed_predictions = predictions_by_model[model_name]
        window_predictions = seed_predictions.mean(axis=0)
        overlap_mean, _ = aggregate_overlapping_windows(window_predictions)
        for window_index, values in enumerate(window_predictions):
            x = np.arange(window_index, window_index + horizon)
            axis.plot(x, values, color="tab:orange", alpha=0.055, linewidth=0.65)
        axis.plot(
            np.arange(calendar_length),
            calendar_truth,
            color="tab:blue",
            linewidth=1.8,
            label="Ground Truth",
            zorder=3,
        )
        axis.plot(
            np.arange(calendar_length),
            overlap_mean,
            color="darkorange",
            linewidth=2.0,
            label="Mean across overlapping forecast origins",
            zorder=4,
        )
        axis.plot([], [], color="tab:orange", alpha=0.35, linewidth=1.2, label="Rolling forecasts")
        axis.set_title(display_names.get(model_name, model_name))
        axis.set_ylabel("Daily total power (kW)")
        axis.grid(alpha=0.16)
        axis.legend(loc="best", fontsize=8)

    tick_count = min(8, calendar_length)
    ticks = np.linspace(0, calendar_length - 1, tick_count, dtype=int)
    axes[-1].set_xticks(ticks, [calendar_dates[index] for index in ticks], rotation=25, ha="right")
    axes[-1].set_xlabel("Target date")
    seed_text = ", ".join(str(seed) for seed in seeds)
    figure.suptitle(
        f"All {window_count} rolling {horizon}-day test forecasts; five-seed means ({seed_text})",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(out_path)
    plt.close(figure)


def plot_summary_table(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display = summary[
        [
            "model",
            "horizon",
            "normalized_mse_mean",
            "normalized_mse_std",
            "normalized_mae_mean",
            "normalized_mae_std",
            "mae_kw_mean",
        ]
    ].copy()
    display.columns = [
        "Model",
        "Horizon",
        "Norm MSE",
        "Norm MSE Std",
        "Norm MAE",
        "Norm MAE Std",
        "MAE (kW)",
    ]
    display["Model"] = display["Model"].replace(
        {"weather_bayesformer_uq": "WeatherBayesFormerUQ", "transformer": "Transformer", "lstm": "LSTM"}
    )
    for column in ["Norm MSE", "Norm MSE Std", "Norm MAE", "Norm MAE Std", "MAE (kW)"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    fig, ax = plt.subplots(figsize=(14, 3.8), dpi=160)
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.23, 0.08, 0.13, 0.13, 0.13, 0.13, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#9aa0a6")
        if row == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f8fa")
    ax.set_title("Five-seed normalized errors with daily-power-scale MAE", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    columns = list(df.columns)

    def fmt(value) -> str:
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    rows = [[fmt(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) if rows else len(str(col))
        for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_report(summary: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    point_columns = [
        "model",
        "horizon",
        "normalized_mse_mean",
        "normalized_mse_std",
        "normalized_mae_mean",
        "normalized_mae_std",
        "mse_kw2_mean",
        "mae_kw_mean",
        "rmse_kw_mean",
        "runs",
        "fit_windows",
        "val_windows",
        "test_windows",
    ]
    table = dataframe_to_markdown(summary[point_columns], floatfmt=".4f")
    uq_rows = summary.loc[
        summary["model"] == "weather_bayesformer_uq",
        [
            "horizon",
            "picp_80_mean",
            "picp_80_std",
            "mpiw_80_kw_mean",
            "mpiw_80_kw_std",
            "mc_samples",
        ],
    ]
    uq_table = dataframe_to_markdown(uq_rows, floatfmt=".4f")
    best_rows = (
        summary.sort_values(["horizon", "normalized_mse_mean"])
        .groupby("horizon")
        .head(1)[
            ["horizon", "model", "normalized_mse_mean", "normalized_mae_mean", "mae_kw_mean"]
        ]
    )
    best_table = dataframe_to_markdown(best_rows, floatfmt=".4f")
    figure_block = """
![Five-seed result summary](../outputs/figures/summary_results.png)

![LSTM 90-day five-seed mean prediction](../outputs/figures/lstm_h90_5seed_mean.png)

![Transformer 90-day five-seed mean prediction](../outputs/figures/transformer_h90_5seed_mean.png)

![WeatherBayesFormerUQ 90-day five-seed mean prediction](../outputs/figures/weather_bayesformer_uq_h90_5seed_mean.png)

![All rolling 90-day test forecasts](../outputs/figures/all_windows_h90_5seed_mean.png)

![LSTM 365-day five-seed mean prediction](../outputs/figures/lstm_h365_5seed_mean.png)

![Transformer 365-day five-seed mean prediction](../outputs/figures/transformer_h365_5seed_mean.png)

![WeatherBayesFormerUQ 365-day five-seed mean prediction](../outputs/figures/weather_bayesformer_uq_h365_5seed_mean.png)
""".strip()
    body = f"""# 2026年专硕机器学习课程项目实验报告

作者：待填写

研究领域：待填写

GitHub链接：待填写

## 1. 问题介绍

本项目面向家庭总有功功率的多变量时间序列预测任务。原始数据来自 UCI Individual household electric power consumption 数据集，记录法国一户家庭从 2006-12 到 2010-11 的分钟级用电数据。按照课程要求，本文对每一天的 `global_active_power` 分钟记录直接求和，形成日总有功功率序列；使用过去 90 天的多变量历史曲线预测未来 90 天和 365 天中每一天的总有功功率。

预处理遵循题目要求：有功功率、无功功率和分表字段按天求和，电压和电流按天求均值，不把总有功功率换算成用电量。分钟缺失值只使用过去观测前向填充，低覆盖日复用前一完整日，采集首尾不足 1440 分钟的边界日被排除。天气字段来自距离 Sceaux 最近的可用站点；月度气候汇总整体滞后一个月后映射到每日样本，确保预测起点只使用已经结束月份的信息。

正式模型统一使用 36 个历史特征，包括原始日级电力、日历编码、滞后月度天气、滞后值、滚动统计、指数均值、差分及比例特征。所有模型只接收过去 90 天的信息。

## 2. 模型

本文比较三类模型。所有模型都以形状为 `(90, feature_dim)` 的历史窗口作为输入，直接输出长度为 90 或 365 的预测向量。短期预测和长期预测分别训练，参数互不复用。

LSTM 模型使用循环结构编码历史依赖，取最后一层隐藏状态经过多层感知机输出未来曲线。

Transformer 模型先将多变量输入映射到隐藏维度，叠加正弦位置编码，再通过 Transformer Encoder 建模不同日期之间的全局依赖。输出端拼接最后时刻表示和均值池化表示，增强对短期状态和整体趋势的刻画。

改进模型 WeatherBayesFormerUQ 在 Transformer Encoder 前加入 7 日趋势--残差分解、7 日卷积和历史天气门控，并输出 0.1/0.5/0.9 分位数。测试时保持显式 Dropout 层激活并进行 30 次 MC 采样；点预测取 MC 中位数预测的均值，区间下界取 q0.1 输出在 MC 样本间的 0.1 分位数，上界取 q0.9 输出的 0.9 分位数。

伪代码如下：

```text
X = last_90_days_multivariate_features
power = X[:, global_active_power]
trend = MovingAverage(power, window=7)
residual = power - trend
X_aug = concat(X, trend, residual)
Z = Linear(X_aug)
Z = Z + WeeklyPatchConv1D(Z, kernel_size=7)
weather = mean(X[:, [RR, NBJRR1, NBJRR5, NBJRR10, NBJBROU]], time)
gate = sigmoid(MLP(weather))
Z = Z * (1 + gate) + learnable_positional_encoding
H = TransformerEncoder(Z)
H = BayesianDropout(H)
context = concat(H_last, mean_pool(H))
Q = MLP(context).reshape(horizon, [q10, q50, q90])
y_hat = Q[:, q50]
```

## 3. 结果与分析

所有训练损失都在标准化目标空间计算：LSTM/Transformer 优化 MSE，WeatherBayesFormerUQ 优化分位数复合损失；评价时三者统一使用标准化 MSE/MAE。同时将输出反标准化并报告课程日总功率尺度上的 MSE、MAE 和 RMSE。每个模型在每个预测长度上运行 5 个随机种子，报告均值和标准差。拟合与验证目标日期完全分离，Scaler 仅在验证目标开始前的数据上拟合。正式实验使用 30 epochs、隐藏维度 64、batch size 128、AdamW 和 patience 30，并将每次最佳验证权重保存到 `outputs/checkpoints`。

{table}

WeatherBayesFormerUQ 的 MC--分位数区间指标如下。PICP80 是真实值落入区间的比例，MPIW80 是日总功率尺度上的平均区间宽度；该区间同时反映分位数输出与 Dropout 采样变化，是否达到名义 80% 覆盖由 PICP80 检验。

{uq_table}

各预测长度下 MSE 最低的模型如下：

{best_table}

归一化误差便于与使用相同标准化定义的实验比较，反标准化指标则对应题目规定的日总有功功率尺度。由于同一预测长度中的各模型共享训练集 Scaler，两种尺度的模型排序完全一致；不同数据划分、不同 Scaler 或不同窗口口径下的归一化数字不能直接比较。

需要注意的是，日级真实曲线存在大量尖峰和突发低谷。虽然模型已经加入历史天气变量，但仍没有家庭行为、节假日安排和未来日级天气预报等更细粒度外部信息，因此预测线仍比 Ground Truth 更平滑。这种差异不代表模型完全失效，而是多步预测在缺少未来外部信息时常见的均值回归现象。

预测曲线同时采用两种展示口径。单窗口图中，每条橙色实线是随机种子 42--46 的逐日预测均值，橙色窄阴影是五个种子预测的均值加减一个样本标准差；WeatherBayesFormerUQ 的灰色宽阴影是五个种子各自 30 次 MC 采样所得分位数上下界的逐日均值。90 天全窗口图则展示全部 276 个滚动测试窗口：每条浅橙线是一个窗口在五个种子上的平均预测，深橙线是同一日期所收到的不同预测起点结果的均值，蓝线是真值。365 天测试段只有 1 个完整窗口，因此不存在与单窗口不同的全窗口重叠图。表格指标仍按每个种子的完整测试集单独计算后再汇总，不使用图中的集成均值重新计算指标。

{figure_block}

## 4. 讨论

本项目按时间顺序划分，将最后 365 天作为测试集。拟合目标与验证目标之间不重叠，缺失值只做因果前向填充，月度天气滞后一个月；因此预处理和模型选择均不使用预测起点之后的信息。90 天测试仍采用 rolling-origin 口径，同一日期在不同提前量下可被评价多次。

当前限制包括月度天气粒度较粗、365 天验证和测试各只有一个完整窗口，以及 MC--分位数区间尚未校准。归一化误差依赖训练集均值和标准差，因此报告同时保留反标准化误差。后续可使用逐日天气预报、滚动年度交叉验证和 conformal calibration。

工具说明：报告初稿可使用 ChatGPT/Codex 辅助整理，但实验代码、数据处理和结果应以本仓库可复现输出为准。

## 参考文献

[1] UCI Machine Learning Repository. Individual household electric power consumption. https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

[2] Météo-France. Données climatologiques de base mensuelles. https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles/

[3] Hochreiter, S., Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.

[4] Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.

[5] PyTorch documentation. https://pytorch.org/docs/stable/index.html

[6] Gal, Y., Ghahramani, Z. Dropout as a Bayesian Approximation. ICML, 2016.

[7] Koenker, R., Bassett, G. Regression Quantiles. Econometrica, 1978.
"""
    out_path.write_text(body, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    ensemble_records: dict[tuple[str, int], list[dict[str, object]]] = {}
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        data = prepare_data(
            daily_path=Path(args.daily_path),
            input_len=args.input_len,
            horizon=horizon,
            test_days=args.test_days,
            val_fraction=args.val_fraction,
            target_mode=args.target_mode,
            baseline_mode=args.baseline_mode,
        )
        for model_name in args.models:
            for seed in args.seeds:
                set_seed(seed)
                model = build_model(
                    model_name=model_name,
                    input_dim=data.input_dim,
                    horizon=horizon,
                    future_dim=data.future_dim,
                    d_model=args.d_model,
                    use_future_decoder=args.use_future_decoder,
                )
                model, val_loss, best_epoch, epochs_ran = train_one(
                    model=model,
                    data=data,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    device=device,
                )
                checkpoint_path = checkpoint_dir / f"{model_name}_h{horizon}_seed{seed}.pt"
                torch.save(
                    {
                        "model_state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "model": model_name,
                        "horizon": horizon,
                        "seed": seed,
                        "input_dim": data.input_dim,
                        "future_dim": data.future_dim,
                        "d_model": args.d_model,
                        "best_epoch": best_epoch,
                        "val_loss_scaled": val_loss,
                        "feature_columns": data.feature_columns,
                    },
                    checkpoint_path,
                )
                active_mc_samples = (
                    args.mc_samples if getattr(model, "is_quantile_model", False) else 1
                )
                pred_scaled, true_scaled, test_aux = predict(
                    model,
                    data.test_ds,
                    args.batch_size,
                    device,
                    data.target_mode,
                    return_aux=True,
                    mc_samples=active_mc_samples,
                )
                if args.calibrate:
                    val_pred_scaled, val_true_scaled, val_aux = predict(
                        model,
                        data.val_ds,
                        args.batch_size,
                        device,
                        data.target_mode,
                        return_aux=True,
                    )
                    coefs = fit_linear_calibration(val_pred_scaled, val_true_scaled, val_aux)
                    pred_scaled = apply_linear_calibration(pred_scaled, test_aux, coefs)
                    if "lower" in test_aux:
                        test_aux["lower"] = apply_linear_calibration(
                            test_aux["lower"], test_aux, coefs
                        )
                        test_aux["upper"] = apply_linear_calibration(
                            test_aux["upper"], test_aux, coefs
                        )
                pred = inverse_target(data.target_scaler, pred_scaled)
                truth = inverse_target(data.target_scaler, true_scaled)
                lower = (
                    inverse_target(data.target_scaler, test_aux["lower"])
                    if "lower" in test_aux
                    else None
                )
                upper = (
                    inverse_target(data.target_scaler, test_aux["upper"])
                    if "upper" in test_aux
                    else None
                )

                ensemble_records.setdefault((model_name, horizon), []).append(
                    {
                        "seed": seed,
                        "dates": np.asarray(data.test_dates),
                        "truth": truth.copy(),
                        "prediction": pred.copy(),
                        "lower": lower.copy() if lower is not None else None,
                        "upper": upper.copy() if upper is not None else None,
                    }
                )

                normalized_mse = mean_squared_error(
                    true_scaled.reshape(-1), pred_scaled.reshape(-1)
                )
                normalized_mae = mean_absolute_error(
                    true_scaled.reshape(-1), pred_scaled.reshape(-1)
                )
                mse_kw2 = mean_squared_error(truth.reshape(-1), pred.reshape(-1))
                mae_kw = mean_absolute_error(truth.reshape(-1), pred.reshape(-1))
                rmse_kw = float(np.sqrt(mse_kw2))
                picp_80 = (
                    float(np.mean((truth >= lower) & (truth <= upper)))
                    if lower is not None and upper is not None
                    else np.nan
                )
                mpiw_80_kw = (
                    float(np.mean(upper - lower))
                    if lower is not None and upper is not None
                    else np.nan
                )
                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "seed": seed,
                        "normalized_mse": normalized_mse,
                        "normalized_mae": normalized_mae,
                        "mse_kw2": mse_kw2,
                        "mae_kw": mae_kw,
                        "rmse_kw": rmse_kw,
                        "picp_80": picp_80,
                        "mpiw_80_kw": mpiw_80_kw,
                        "val_loss_scaled": val_loss,
                        "best_epoch": best_epoch,
                        "epochs_ran": epochs_ran,
                        "fit_windows": len(data.train_ds),
                        "val_windows": len(data.val_ds),
                        "test_windows": len(data.test_ds),
                        "scaler_fit_end": data.scaler_fit_end,
                        "validation_target_start": data.validation_target_start,
                        "target_mean_kw": float(data.target_scaler.mean_[0]),
                        "target_std_kw": float(data.target_scaler.scale_[0]),
                        "mc_samples": active_mc_samples,
                        "features": json.dumps(data.feature_columns),
                        "future_features": json.dumps(data.future_columns),
                        "target_mode": data.target_mode,
                        "baseline_mode": data.baseline_mode,
                        "use_future_decoder": bool(args.use_future_decoder),
                        "calibrated": bool(args.calibrate),
                        "quantiles": json.dumps(
                            getattr(model, "quantiles", torch.tensor([])).detach().cpu().tolist()
                        ),
                    }
                )
                print(
                    f"model={model_name} horizon={horizon} seed={seed} "
                    f"norm_mse={normalized_mse:.4f} norm_mae={normalized_mae:.4f} "
                    f"mae_kw={mae_kw:.4f} rmse_kw={rmse_kw:.4f} "
                    f"picp80={picp_80:.4f} mpiw80_kw={mpiw_80_kw:.4f}"
                )

    rolling_predictions: dict[int, dict[str, np.ndarray]] = {}
    rolling_truths: dict[int, np.ndarray] = {}
    rolling_dates: dict[int, list[str]] = {}
    rolling_seeds: dict[int, np.ndarray] = {}
    for (model_name, horizon), records in sorted(ensemble_records.items()):
        records.sort(key=lambda item: int(item["seed"]))
        seeds = np.asarray([int(item["seed"]) for item in records])
        dates = np.asarray(records[0]["dates"])
        truths = np.stack([np.asarray(item["truth"]) for item in records])
        predictions = np.stack([np.asarray(item["prediction"]) for item in records])
        if not all(np.array_equal(dates, np.asarray(item["dates"])) for item in records):
            raise ValueError(f"Test dates differ across seeds for {model_name}, horizon={horizon}.")
        if not np.allclose(truths, truths[0], rtol=0.0, atol=1e-6):
            raise ValueError(f"Ground truth differs across seeds for {model_name}, horizon={horizon}.")

        has_mc_interval = all(
            item["lower"] is not None and item["upper"] is not None for item in records
        )
        mc_lowers = (
            np.stack([np.asarray(item["lower"]) for item in records])
            if has_mc_interval
            else None
        )
        mc_uppers = (
            np.stack([np.asarray(item["upper"]) for item in records])
            if has_mc_interval
            else None
        )
        first_window_truths = truths[:, 0, :]
        first_window_predictions = predictions[:, 0, :]
        first_window_mc_lowers = mc_lowers[:, 0, :] if mc_lowers is not None else None
        first_window_mc_uppers = mc_uppers[:, 0, :] if mc_uppers is not None else None
        output_stem = f"{model_name}_h{horizon}_5seed_mean"
        plot_seed_ensemble_prediction(
            dates=dates.tolist(),
            truth=first_window_truths[0],
            predictions=first_window_predictions,
            model_name=model_name,
            horizon=horizon,
            seeds=seeds,
            out_path=figure_dir / f"{output_stem}.png",
            mc_lowers=first_window_mc_lowers,
            mc_uppers=first_window_mc_uppers,
        )
        calendar_length = predictions.shape[1] + horizon - 1
        calendar_dates = pd.date_range(
            start=pd.Timestamp(dates[0]),
            periods=calendar_length,
            freq="D",
        ).strftime("%Y-%m-%d").to_numpy(dtype="<U10")
        calendar_truth, _ = aggregate_overlapping_windows(truths[0])
        overlap_prediction_mean, overlap_prediction_std = aggregate_overlapping_windows(
            predictions.mean(axis=0)
        )
        payload = {
            "dates": dates,
            "truth": first_window_truths[0],
            "seeds": seeds,
            "predictions_by_seed": first_window_predictions,
            "prediction_mean": first_window_predictions.mean(axis=0),
            "prediction_std": (
                first_window_predictions.std(axis=0, ddof=1)
                if len(first_window_predictions) > 1
                else np.zeros_like(first_window_predictions[0])
            ),
            "window_truth": truths[0],
            "window_predictions_by_seed": predictions,
            "window_prediction_mean": predictions.mean(axis=0),
            "calendar_dates": calendar_dates,
            "calendar_truth": calendar_truth,
            "overlap_prediction_mean": overlap_prediction_mean,
            "overlap_prediction_std": overlap_prediction_std,
        }
        if mc_lowers is not None and mc_uppers is not None:
            payload["mc_lower_by_seed"] = first_window_mc_lowers
            payload["mc_upper_by_seed"] = first_window_mc_uppers
            payload["mc_lower_mean"] = first_window_mc_lowers.mean(axis=0)
            payload["mc_upper_mean"] = first_window_mc_uppers.mean(axis=0)
            payload["window_mc_lower_by_seed"] = mc_lowers
            payload["window_mc_upper_by_seed"] = mc_uppers
        np.savez_compressed(prediction_dir / f"{output_stem}.npz", **payload)

        rolling_predictions.setdefault(horizon, {})[model_name] = predictions
        rolling_truths[horizon] = truths[0]
        rolling_dates[horizon] = dates.tolist()
        rolling_seeds[horizon] = seeds

    for horizon, model_predictions in sorted(rolling_predictions.items()):
        if rolling_truths[horizon].shape[0] <= 1:
            continue
        plot_all_rolling_windows(
            first_window_dates=rolling_dates[horizon],
            truth_windows=rolling_truths[horizon],
            predictions_by_model=model_predictions,
            horizon=horizon,
            seeds=rolling_seeds[horizon],
            out_path=figure_dir / f"all_windows_h{horizon}_5seed_mean.png",
        )

    metrics = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = (
        metrics.groupby(["model", "horizon"], as_index=False)
        .agg(
            normalized_mse_mean=("normalized_mse", "mean"),
            normalized_mse_std=("normalized_mse", "std"),
            normalized_mae_mean=("normalized_mae", "mean"),
            normalized_mae_std=("normalized_mae", "std"),
            mse_kw2_mean=("mse_kw2", "mean"),
            mse_kw2_std=("mse_kw2", "std"),
            mae_kw_mean=("mae_kw", "mean"),
            mae_kw_std=("mae_kw", "std"),
            rmse_kw_mean=("rmse_kw", "mean"),
            rmse_kw_std=("rmse_kw", "std"),
            picp_80_mean=("picp_80", "mean"),
            picp_80_std=("picp_80", "std"),
            mpiw_80_kw_mean=("mpiw_80_kw", "mean"),
            mpiw_80_kw_std=("mpiw_80_kw", "std"),
            runs=("seed", "count"),
            fit_windows=("fit_windows", "first"),
            val_windows=("val_windows", "first"),
            test_windows=("test_windows", "first"),
            mc_samples=("mc_samples", "first"),
        )
        .sort_values(["horizon", "normalized_mse_mean"])
    )
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_summary_table(summary, figure_dir / "summary_results.png")
    write_report(summary, metrics, Path(args.report_path))

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train household power forecasting models.")
    parser.add_argument("--daily-path", default="data/processed/daily_power.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--report-path", default="reports/report.md")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--target-mode", choices=["level", "residual"], default="level")
    parser.add_argument(
        "--baseline-mode",
        choices=["seasonal", "last", "mean", "zero"],
        default="seasonal",
    )
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--use-future-decoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
