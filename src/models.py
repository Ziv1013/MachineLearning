from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FutureDecoder(nn.Module):
    def __init__(self, context_dim: int, future_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(context_dim + future_dim),
            nn.Linear(context_dim + future_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor, future_x: torch.Tensor) -> torch.Tensor:
        repeated = context.unsqueeze(1).expand(-1, future_x.size(1), -1)
        z = torch.cat([repeated, future_x], dim=-1)
        return self.head(z).squeeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        future_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_future_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.use_future_decoder = use_future_decoder
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        if use_future_decoder:
            self.decoder = FutureDecoder(hidden_dim, future_dim, hidden_dim, dropout)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, horizon),
            )

    def forward(self, x: torch.Tensor, future_x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        if self.use_future_decoder:
            return self.decoder(hidden[-1], future_x)
        return self.head(hidden[-1])


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        future_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_future_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.use_future_decoder = use_future_decoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        if use_future_decoder:
            self.decoder = FutureDecoder(d_model * 2, future_dim, d_model, dropout)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, horizon),
            )

    def forward(self, x: torch.Tensor, future_x: torch.Tensor) -> torch.Tensor:
        z = self.positional(self.input_proj(x))
        encoded = self.encoder(z)
        pooled = torch.cat([encoded[:, -1], encoded.mean(dim=1)], dim=1)
        if self.use_future_decoder:
            return self.decoder(pooled, future_x)
        return self.head(pooled)


class BayesFormerUQForecaster(nn.Module):
    """Transformer with Bayesian dropout and quantile outputs."""

    is_quantile_model = True

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        future_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_quantiles = len(quantiles)
        self.median_index = min(range(self.num_quantiles), key=lambda i: abs(quantiles[i] - 0.5))
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32), persistent=False)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.bayesian_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * self.num_quantiles),
        )

    def forward(self, x: torch.Tensor, future_x: torch.Tensor) -> torch.Tensor:
        del future_x
        z = self.input_proj(x) + self.position[:, : x.size(1)]
        encoded = self.encoder(z)
        encoded = self.bayesian_dropout(encoded)
        pooled = torch.cat([encoded[:, -1], encoded.mean(dim=1)], dim=1)
        out = self.head(pooled)
        return out.view(x.size(0), self.horizon, self.num_quantiles)


def build_model(
    model_name: str,
    input_dim: int,
    horizon: int,
    future_dim: int,
    d_model: int,
    use_future_decoder: bool = False,
) -> nn.Module:
    name = model_name.lower()
    if name == "lstm":
        return LSTMForecaster(
            input_dim=input_dim,
            horizon=horizon,
            future_dim=future_dim,
            hidden_dim=d_model,
            use_future_decoder=use_future_decoder,
        )
    if name == "transformer":
        return TransformerForecaster(
            input_dim=input_dim,
            horizon=horizon,
            future_dim=future_dim,
            d_model=d_model,
            use_future_decoder=use_future_decoder,
        )
    if name in {"bayes_former_uq", "bayesformer_uq", "bayesian_transformer_uq", "bayesformer"}:
        return BayesFormerUQForecaster(
            input_dim=input_dim,
            horizon=horizon,
            future_dim=future_dim,
            d_model=d_model,
        )
    raise ValueError(f"Unknown model: {model_name}")
