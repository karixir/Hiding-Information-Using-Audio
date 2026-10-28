from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Niezgodne wymiary: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in decibels."""
    err = mse(a, b)
    if err == 0:
        return float("inf")
    max_val = float(np.max(np.abs(a)))
    if max_val == 0:
        return 0.0
    return 10.0 * np.log10((max_val ** 2) / err)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """A simple SSIM approximation for grayscale images."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Niezgodne wymiary: {a.shape} vs {b.shape}")

    mu_x = np.mean(a)
    mu_y = np.mean(b)
    sigma_x = np.std(a)
    sigma_y = np.std(b)
    sigma_xy = np.mean((a - mu_x) * (b - mu_y))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)
