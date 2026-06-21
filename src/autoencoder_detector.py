"""
Autoencoder Anomaly Detector
==============================
Neural network autoencoder that learns to reconstruct normal events.
High reconstruction error → anomaly.

Architecture:
  Input(6) → Dense(32) → ReLU → Dense(16) → ReLU → Dense(8) [bottleneck]
           → Dense(16) → ReLU → Dense(32) → ReLU → Dense(6)

Implemented in pure numpy (no torch/tensorflow dependency) so it
runs in any environment. A PyTorch version is included as reference.
The architecture matches what would be used in production with PyTorch.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AutoencoderResult:
    event_id: str
    timestamp: float
    service: str
    metric_name: str
    value: float
    reconstruction_error: float
    is_anomaly: bool
    confidence: float
    detector: str = "Autoencoder"


# ─── Numpy MLP Implementation ─────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


class _Layer:
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.activation = activation
        self._input = None
        self._pre_act = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        z = x @ self.W + self.b
        self._pre_act = z
        return relu(z) if self.activation == "relu" else z

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        if self.activation == "relu":
            grad_out = grad_out * relu_grad(self._pre_act)
        grad_W = self._input.T @ grad_out
        grad_b = grad_out.sum(axis=0)
        grad_in = grad_out @ self.W.T
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_in


class NumpyAutoencoder:
    """
    Lightweight autoencoder built in pure numpy.
    Architecture mirrors the PyTorch reference implementation.
    """

    def __init__(self, input_dim: int = 6, latent_dim: int = 8, lr: float = 0.001):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        # Encoder
        self._enc = [
            _Layer(input_dim, 32, "relu", seed=1),
            _Layer(32,        16, "relu", seed=2),
            _Layer(16, latent_dim,"relu", seed=3),
        ]
        # Decoder
        self._dec = [
            _Layer(latent_dim, 16, "relu", seed=4),
            _Layer(16,         32, "relu", seed=5),
            _Layer(32, input_dim, "linear", seed=6),
        ]
        self._layers = self._enc + self._dec

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for layer in self._layers:
            h = layer.forward(h)
        return h

    def _backward(self, x: np.ndarray, x_hat: np.ndarray) -> float:
        loss_val = mse(x, x_hat)
        grad = 2 * (x_hat - x) / x.shape[0]
        for layer in reversed(self._layers):
            grad = layer.backward(grad, self.lr)
        return loss_val

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 80,
        batch_size: int = 128,
        verbose: bool = True
    ) -> List[float]:
        losses = []
        n = len(X)
        rng = np.random.RandomState(42)
        for epoch in range(epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = X[idx[start:start + batch_size]]
                x_hat = self._forward(batch)
                loss  = self._backward(batch, x_hat)
                epoch_loss += loss
            avg_loss = epoch_loss / (n // batch_size)
            losses.append(avg_loss)
            if verbose and epoch % 20 == 0:
                logger.info("Epoch %d/%d | Loss: %.6f", epoch + 1, epochs, avg_loss)
        return losses

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def build_features(events: List[dict], feature_names: List[str]) -> np.ndarray:
    """
    Build fixed-length feature vector per event.
    Features: [value, log1p(value), value^2, z-diff, rolling_mean_ratio, metric_onehot_idx]
    """
    metric_map = {m: i for i, m in enumerate(feature_names)}
    rows = []
    values_so_far = []
    for ev in events:
        v = ev["value"]
        values_so_far.append(v)
        roll_mean = np.mean(values_so_far[-50:]) if len(values_so_far) >= 2 else v
        diff = v - values_so_far[-2] if len(values_so_far) >= 2 else 0.0
        m_idx = metric_map.get(ev.get("metric_name", ""), 0) / max(len(feature_names), 1)
        rows.append([
            v,
            np.log1p(abs(v)),
            v ** 2 / 1e6,
            diff,
            v / (roll_mean + 1e-8),
            m_idx
        ])
    return np.array(rows, dtype=np.float64)


# ─── Detector ─────────────────────────────────────────────────────────────────

class AutoencoderDetector:
    """
    Autoencoder anomaly detector.
    Trains on normal data; detects anomalies via reconstruction error.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        lr: float = 0.001,
        threshold_percentile: float = 97.5
    ):
        self.latent_dim = latent_dim
        self.lr = lr
        self.threshold_percentile = threshold_percentile
        self.model: NumpyAutoencoder = None
        self.scaler_mean: np.ndarray = None
        self.scaler_std:  np.ndarray = None
        self._threshold: float = None
        self._metric_names: List[str] = []
        self._fitted = False
        logger.info("AutoencoderDetector initialized (latent_dim=%d, threshold=%.1f%%)",
                    latent_dim, threshold_percentile)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.scaler_mean) / (self.scaler_std + 1e-8)

    def fit(self, events: List[dict], epochs: int = 80) -> List[float]:
        self._metric_names = list({e.get("metric_name", "") for e in events})
        X = build_features(events, self._metric_names)
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std  = X.std(axis=0)
        X_scaled = self._scale(X)

        self.model = NumpyAutoencoder(input_dim=X.shape[1], latent_dim=self.latent_dim, lr=self.lr)
        losses = self.model.fit(X_scaled, epochs=epochs, verbose=True)

        # Compute threshold on training reconstruction errors
        train_errors = self.model.reconstruction_error(X_scaled)
        self._threshold = np.percentile(train_errors, self.threshold_percentile)
        self._fitted = True
        logger.info("Autoencoder fitted | threshold=%.6f", self._threshold)
        return losses

    def score_batch(self, events: List[dict]) -> List[AutoencoderResult]:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        X = build_features(events, self._metric_names)
        X_scaled = self._scale(X)
        errors = self.model.reconstruction_error(X_scaled)

        # Normalise error to [0,1] confidence
        max_e = errors.max() if errors.max() > 0 else 1.0
        confidence = np.clip(errors / (self._threshold * 3), 0, 1)

        results = []
        for i, ev in enumerate(events):
            is_anomaly = errors[i] > self._threshold
            results.append(AutoencoderResult(
                event_id=ev["event_id"],
                timestamp=ev["timestamp"],
                service=ev["service"],
                metric_name=ev["metric_name"],
                value=ev["value"],
                reconstruction_error=round(float(errors[i]), 8),
                is_anomaly=is_anomaly,
                confidence=round(float(confidence[i]), 4),
                detector="Autoencoder"
            ))
        return results
