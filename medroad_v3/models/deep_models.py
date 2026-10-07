"""
MedROAD V3 — Deep Models (LSTM + Transformer)
Both models accept sequences of shape (batch, seq_len, n_features)
and return scalar risk probabilities in [0, 1].
"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn

from medroad_v3 import config

logger = logging.getLogger(__name__)


# ── LSTM ─────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    2-layer bidirectional LSTM with dropout.
    Architecture matches paper: hidden=128, layers=2, dropout=0.3.
    """

    def __init__(
        self,
        n_features: int  = config.N_FEATURES,
        hidden_size: int = config.LSTM_HIDDEN,
        num_layers: int  = config.LSTM_LAYERS,
        dropout: float   = config.LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.bn   = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Unsquashed scores, for use with BCEWithLogitsLoss."""
        out, _ = self.lstm(x)              # (batch, seq_len, hidden)
        last = out[:, -1, :]               # take last timestep
        last = self.bn(last)
        last = self.drop(last)
        return self.head(last).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


# ── Transformer ───────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    """
    Transformer encoder for clinical time-series.
    Architecture: d_model=64, nhead=4, 2 encoder layers, sinusoidal PE.
    """

    def __init__(
        self,
        n_features:  int   = config.N_FEATURES,
        d_model:     int   = config.TF_D_MODEL,
        nhead:       int   = config.TF_NHEAD,
        num_layers:  int   = config.TF_NUM_LAYERS,
        dim_ff:      int   = config.TF_DIM_FF,
        dropout:     float = config.TF_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            # Pre-norm. PyTorch defaults to post-norm, which diverges in the
            # first optimisation steps unless the learning rate is warmed up;
            # on a small, heavily imbalanced cohort it produces a non-finite
            # loss at epoch 1. Pre-norm is stable without warmup.
            norm_first=True,
        )
        # A pre-norm stack needs a final LayerNorm. Without it nothing bounds
        # the residual stream, activations grow with depth and training steps,
        # and the loss diverges to the thousands even though it never becomes
        # NaN. PyTorch does not add this automatically.
        self.encoder    = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.pool       = nn.AdaptiveAvgPool1d(1)  # global average pooling over seq
        self.head       = nn.Linear(d_model, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Unsquashed scores, for use with BCEWithLogitsLoss."""
        x = self.input_proj(x)             # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)                # (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)             # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)       # (batch, d_model)
        return self.head(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


# ── Temperature scaling calibration ──────────────────────────────────────────

class TemperatureScaledModel(nn.Module):
    """
    Post-hoc calibration for LSTM/Transformer via temperature scaling.
    Learns a single scalar T such that p_cal = sigmoid(logit / T).
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.model = base_model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled logits."""
        with torch.no_grad():
            # Take logits from the base model directly. Recovering them by
            # inverting the sigmoid loses precision and returns infinities once
            # probabilities saturate, which is exactly the regime a large class
            # weight drives the model into.
            if hasattr(self.model, "forward_logits"):
                logits = self.model.forward_logits(x)
            else:
                probs = self.model(x).clamp(1e-6, 1 - 1e-6)
                logits = torch.log(probs / (1 - probs))
        return logits / self.temperature.clamp(min=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))

    def calibrate(
        self,
        val_seq: np.ndarray,     # (N, seq_len, n_features)
        val_labels: np.ndarray,  # (N,)
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        """Fit temperature on a held-out validation set using NLL."""
        # The temperature scalar is created on CPU while the wrapped model may
        # already be on GPU, so next(self.parameters()) can report the wrong
        # device. Take it from the wrapped model and move the scalar to match.
        device = next(self.model.parameters()).device
        self.temperature.data = self.temperature.data.to(device)
        x = torch.tensor(val_seq, dtype=torch.float32).to(device)
        y = torch.tensor(val_labels, dtype=torch.float32).to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward_logits(x), y)
            loss.backward()
            return loss

        optimizer.step(closure)
        logger.info("Temperature calibrated: T=%.4f", self.temperature.item())


# ── Inference utilities ───────────────────────────────────────────────────────

def predict_sequence(
    model: nn.Module,
    sequence: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Run a single sample through LSTM or Transformer.
    sequence: (seq_len, n_features) → returns scalar probability.
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        prob = model(x).item()
    return float(prob)


def predict_batched(
    model: nn.Module,
    sequences: np.ndarray,
    device: str = "cpu",
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Score a sequence tensor in batches.

    A single forward pass over tens of thousands of sequences exceeds CUDA's
    grid limits inside the fused transformer kernel and fails with "invalid
    configuration argument". Batching also keeps peak memory bounded, which
    matters once a cohort reaches hundreds of thousands of windows.
    """
    model.eval()
    model.to(device)
    out = np.empty(len(sequences), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i:i + batch_size]
            t = torch.tensor(chunk, dtype=torch.float32).to(device)
            out[i:i + batch_size] = model(t).cpu().numpy()
    return out


def load_model(model_class: type, path: str, device: str = "cpu") -> nn.Module:
    """
    Load a saved model and place it on the requested device.

    map_location governs where the stored tensors are read to, not where the
    module lives. Without the explicit .to(device) the parameters stay on CPU
    while callers send inputs to GPU, and every forward pass fails with a
    device mismatch.
    """
    model = model_class()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded %s from %s on %s", model_class.__name__, path, device)
    return model


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
    logger.info("Saved %s to %s", type(model).__name__, path)
