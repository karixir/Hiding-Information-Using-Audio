from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio(file_path: str | Path) -> Tuple[np.ndarray, int]:
    """Load audio as normalized float samples with sample rate."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku audio: {path}")

    data, sr = sf.read(path, dtype="int16", always_2d=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    return data.astype(np.float64) / 32767.0, int(sr)


def save_audio(file_path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Save audio to WAV format in 16-bit PCM while preserving the sample values."""
    out = Path(file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples)
    if np.issubdtype(arr.dtype, np.integer):
        pcm = np.clip(arr.astype(np.int32), -32768, 32767).astype(np.int16)
    else:
        arr = np.asarray(samples, dtype=np.float64)
        pcm = np.clip(np.rint(arr * 32767.0), -32768, 32767).astype(np.int16)
    sf.write(out, pcm, sample_rate)


def make_test_cover(length_samples: int = 44100, sample_rate: int = 44100) -> np.ndarray:
    """Generate a simple synthetic cover audio signal for tests."""
    t = np.linspace(0, length_samples / sample_rate, length_samples, endpoint=False)
    tone = 0.45 * np.sin(2 * np.pi * 440 * t)
    modulated = 0.35 * np.sin(2 * np.pi * 220 * t + 0.5)
    return tone + modulated


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """Normalize samples to [-1, 1]."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return arr
    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        return arr
    return arr / max_abs
