"""Pure-Python / NumPy / Pandas technical and volatility indicators (no C-compiler TA-Lib dependency)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_atr_14(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """Compute Wilder's 14-day Average True Range (ATR) in pure Python/Pandas."""
    if len(closes) < 2 or len(highs) != len(closes) or len(lows) != len(closes):
        return 0.0
    h = pd.Series(highs, dtype="float64")
    l = pd.Series(lows, dtype="float64")
    c = pd.Series(closes, dtype="float64")
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    window = min(period, len(tr) - 1)
    atr = tr.iloc[1:].ewm(alpha=1.0 / max(window, 1), adjust=False).mean().iloc[-1]
    return float(max(0.0, atr))


def compute_rsi_14(closes: Sequence[float], period: int = 14) -> float:
    """Compute Wilder's 14-day Relative Strength Index (RSI) in [0.0, 100.0]."""
    if len(closes) < 3:
        return 50.0
    c = pd.Series(closes, dtype="float64")
    delta = c.diff().dropna()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    window = min(period, len(delta))
    avg_gain = gains.ewm(alpha=1.0 / max(window, 1), adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1.0 / max(window, 1), adjust=False).mean().iloc[-1]

    if avg_loss <= 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0.0, 100.0))


def compute_realized_vol_20d(closes: Sequence[float], window: int = 20, trading_days: int = 252) -> float:
    """Compute annualized 20-day realized log-return volatility."""
    if len(closes) < 3:
        return 0.0
    arr = np.asarray(closes, dtype=np.float64)
    if np.any(arr <= 0):
        return 0.0
    log_rets = np.diff(np.log(arr))
    tail = log_rets[-min(window, len(log_rets)) :]
    if len(tail) < 2:
        return 0.0
    vol = float(np.std(tail, ddof=1) * np.sqrt(trading_days))
    return max(0.0, vol)


def compute_bid_ask_spread_pct(bid: float, ask: float) -> float:
    """Return bid-ask spread width as a fraction of the mid price: (ask - bid) / mid."""
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return 1.0
    mid = 0.5 * (bid + ask)
    if mid <= 0.0:
        return 1.0
    return float((ask - bid) / mid)
