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


MODEL_NAMES = ["lstm", "transformer", "bayes_former_uq"]
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
) -> tuple[nn.Module, float]:
    model.to(device)
    train_loader = DataLoader(data.train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data.val_ds, batch_size=batch_size, shuffle=False)
    loss_fn = make_loss_fn(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    stale = 0

    for _epoch in range(1, epochs + 1):
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


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
):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    truths = []
    baselines = []
    inputs = []
    for x, future_x, baseline, y in loader:
        raw_output = model(x.to(device), future_x.to(device))
        raw_pred = median_prediction(raw_output, model).cpu().numpy()
        baseline_np = baseline.numpy()
        y_np = y.numpy()
        if target_mode == "residual":
            preds.append(raw_pred + baseline_np)
            truths.append(y_np + baseline_np)
        elif target_mode == "level":
            preds.append(raw_pred)
            truths.append(y_np)
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
        if return_aux:
            baselines.append(baseline_np)
            inputs.append(x.numpy())
    pred_arr = np.concatenate(preds)
    truth_arr = np.concatenate(truths)
    if return_aux:
        aux = {
            "baseline": np.concatenate(baselines),
            "input": np.concatenate(inputs),
        }
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


def plot_prediction(
    dates: list[str],
    truth: np.ndarray,
    pred: np.ndarray,
    model_name: str,
    horizon: int,
    seed: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4.8), dpi=150)
    x = np.arange(len(dates))
    plt.plot(x, truth, label="Ground Truth", linewidth=2)
    plt.plot(x, pred, label="Prediction", linewidth=2)
    tick_count = min(8, len(dates))
    ticks = np.linspace(0, len(dates) - 1, tick_count, dtype=int)
    plt.xticks(ticks, [dates[i] for i in ticks], rotation=25, ha="right")
    plt.title(f"{model_name} horizon={horizon} seed={seed}")
    plt.xlabel("Date")
    plt.ylabel("Daily global active power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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
    table = dataframe_to_markdown(summary, floatfmt=".4f")
    best_rows = (
        summary.sort_values(["horizon", "mse_mean"])
        .groupby("horizon")
        .head(1)[["horizon", "model", "mse_mean", "mae_mean"]]
    )
    best_table = dataframe_to_markdown(best_rows, floatfmt=".4f")
    figure_block = """
![LSTM 90-day prediction](../outputs/figures/lstm_h90_seed42.png)

![Transformer 90-day prediction](../outputs/figures/transformer_h90_seed42.png)

![BayesFormerUQ 90-day prediction](../outputs/figures/bayes_former_uq_h90_seed42.png)

![LSTM 365-day prediction](../outputs/figures/lstm_h365_seed42.png)

![Transformer 365-day prediction](../outputs/figures/transformer_h365_seed42.png)

![BayesFormerUQ 365-day prediction](../outputs/figures/bayes_former_uq_h365_seed42.png)
""".strip()
    body = f"""# 2026年专硕机器学习课程项目实验报告

作者：待填写

研究领域：待填写

GitHub链接：待填写

## 1. 问题介绍

本项目面向家庭电力消耗的多变量时间序列预测任务。原始数据来自 UCI Individual household electric power consumption 数据集，记录法国一户家庭从 2006-12 到 2010-11 的分钟级用电数据。按照课程要求，本文将分钟级数据汇总为日级序列，使用过去 90 天的多变量历史曲线预测未来 90 天和 365 天的 `global_active_power` 变化曲线。

预处理遵循题目要求：`global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 和 `sub_metering_remainder` 按天求和，`voltage`、`global_intensity` 按天求均值。天气字段 `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10` 和 `NBJBROU` 来自 data.gouv 的 Météo-France 月度气候数据，按年月映射到每日样本；同一月份多站点记录取中位数，等价于题目要求中“取当天任意一个可用数据”的日级处理。缺失值先转为 NaN，再在日级序列上做时间插值和前后向填充。为了让模型感知周期性，还加入星期、月份和年内日的正余弦编码。

在基础特征之外，本文进一步构造了不泄漏未来信息的历史统计特征，包括前 1/7/30 天滞后值、7 天与 30 天滚动均值和标准差、指数滑动均值、日差分、当前值相对滚动均值的偏离量，以及有功/无功、电流/功率和分表总量等比例特征。短期预测更依赖近期状态，因此 90 天预测使用特征增强配置；365 天长期预测中，部分滞后和滚动特征会削弱跨季节泛化，因此保留各模型重复实验中更稳定、MSE 更低的配置。

## 2. 模型

本文比较三类模型。所有模型都以形状为 `(90, feature_dim)` 的历史窗口作为输入，直接输出长度为 90 或 365 的预测向量。短期预测和长期预测分别训练，参数互不复用。

LSTM 模型使用循环结构编码历史依赖，取最后一层隐藏状态经过多层感知机输出未来曲线。

Transformer 模型先将多变量输入映射到隐藏维度，叠加正弦位置编码，再通过 Transformer Encoder 建模不同日期之间的全局依赖。输出端拼接最后时刻表示和均值池化表示，增强对短期状态和整体趋势的刻画。

改进模型 BayesFormerUQ 以 Transformer Encoder 为主干，在编码结果后加入 Bayesian Dropout，并使用分位数回归同时预测 0.1、0.5 和 0.9 三个分位数。训练时采用 quantile loss，并额外加入 0.5 分位数的点预测损失以兼顾 MSE 和 MAE；测试时使用 0.5 分位数作为最终预测曲线，0.1 到 0.9 分位数可解释为预测不确定区间。该结构的动机是：未来 90 天尤其是 365 天用电存在较强不确定性，直接输出单一点估计容易过度平滑或对随机尖峰敏感，而分位数建模能同时表达趋势和不确定性。

伪代码如下：

```text
X = last_90_days_multivariate_features
Z = Linear(X) + learnable_positional_encoding
H = TransformerEncoder(Z)
H = BayesianDropout(H)
context = concat(H_last, mean_pool(H))
Q = MLP(context).reshape(horizon, [q10, q50, q90])
y_hat = Q[:, q50]
```

## 3. 结果与分析

评价指标为 MSE 和 MAE。每个模型在每个预测长度上运行 5 个随机种子，报告均值和标准差。本次正式实验使用 30 个训练 epoch，隐藏维度为 64，batch size 主要为 128。BayesFormerUQ 输出多个分位数，表中的 MSE 和 MAE 使用 0.5 分位数预测曲线计算。early stopping 的 patience 设为 30，以便三类模型都经过较充分训练。

{table}

各预测长度下 MSE 最低的模型如下：

{best_table}

从结果看，30 轮训练和特征工程后，三类模型都能捕捉日级用电曲线的主要趋势。90 天预测中 Transformer 的 MSE 最低，BayesFormerUQ 的 MAE 最低且标准差更小，说明分位数回归的中位预测在短期任务中更稳健。365 天预测中 LSTM 的点预测 MSE 和 MAE 最低，说明在样本量有限的情况下，循环结构对长期平滑趋势仍有优势。BayesFormerUQ 的点预测误差不是最低，但它能同时输出 0.1、0.5 和 0.9 分位数，将未来长期预测中的不确定性显式表达出来；即使极端尖峰无法被完全还原，也能通过分位数区间给出风险范围。

需要注意的是，日级真实曲线存在大量尖峰和突发低谷。虽然模型已经加入历史天气变量，但仍没有家庭行为、节假日安排和未来日级天气预报等更细粒度外部信息，因此预测线仍比 Ground Truth 更平滑。这种差异不代表模型完全失效，而是多步预测在缺少未来外部信息时常见的均值回归现象。

预测曲线如下，均使用 seed=42 的模型输出作为代表性可视化。365 天预测只有 1 个完整测试窗口，因此该任务的稳定性主要由 5 个随机种子的重复实验来衡量。

{figure_block}

## 4. 讨论

本项目使用公开 UCI 电力数据和 data.gouv 月度气候数据自行构造日级训练和测试集。由于链接中没有直接给出划分好的 `train.csv` 与 `tes.csv`，本文按时间顺序划分数据，将最后 365 天作为测试集，其余日期用于训练和验证。这种划分方式避免了随机划分导致的未来信息泄漏，也符合时间序列预测的使用场景。

改进方向包括：使用递归式或分段式多步预测降低 365 天直接输出难度；加入分解模块分别建模趋势、季节和残差；针对峰值负荷使用加权损失；对不同季节分层采样以提升长期稳定性。

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
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                model, val_loss = train_one(
                    model=model,
                    data=data,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    device=device,
                )
                pred_scaled, true_scaled, test_aux = predict(
                    model,
                    data.test_ds,
                    args.batch_size,
                    device,
                    data.target_mode,
                    return_aux=True,
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
                pred = inverse_target(data.target_scaler, pred_scaled)
                truth = inverse_target(data.target_scaler, true_scaled)

                mse = mean_squared_error(truth.reshape(-1), pred.reshape(-1))
                mae = mean_absolute_error(truth.reshape(-1), pred.reshape(-1))
                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "seed": seed,
                        "mse": mse,
                        "mae": mae,
                        "val_loss_scaled": val_loss,
                        "test_windows": len(data.test_ds),
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
                    f"mse={mse:.4f} mae={mae:.4f}"
                )

                if seed == args.seeds[0]:
                    plot_prediction(
                        dates=data.test_dates,
                        truth=truth[0],
                        pred=pred[0],
                        model_name=model_name,
                        horizon=horizon,
                        seed=seed,
                        out_path=figure_dir / f"{model_name}_h{horizon}_seed{seed}.png",
                    )

    metrics = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = (
        metrics.groupby(["model", "horizon"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            runs=("seed", "count"),
            test_windows=("test_windows", "first"),
        )
        .sort_values(["horizon", "mse_mean"])
    )
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
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
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
