from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load_image_as_grayscale(image_path: str | Path) -> np.ndarray:
    """Load an image and convert it to a 2D grayscale matrix."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku obrazu: {path}")

    with Image.open(path) as img:
        grayscale = img.convert("L")
        return np.asarray(grayscale, dtype=np.float64)


def pad_image_to_square(matrix: np.ndarray, target_size: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Pad an image matrix to the nearest square size if it is not already square."""
    if matrix.ndim == 3:
        matrix = np.mean(matrix, axis=2)

    h, w = matrix.shape
    n = max(h, w)
    if target_size is not None:
        n = max(target_size, n)

    nearest = int(2 ** np.ceil(np.log2(n))) if n > 0 else 1
    pad_h = nearest - h
    pad_w = nearest - w

    padded = np.pad(matrix, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return padded, nearest


def ensure_square_image(matrix: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
    """Ensure the input matrix is square by padding with zeros."""
    if matrix.ndim != 2:
        raise ValueError(f"Obraz musi mieć 2 wymiary, otrzymano {matrix.ndim}.")
    padded, _ = pad_image_to_square(matrix, target_size)
    return padded


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Scale matrix values into [0, 255] range."""
    arr = np.asarray(matrix, dtype=np.float64)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.float64)
    return ((arr - arr_min) / (arr_max - arr_min)) * 255.0


def matrix_to_vector(matrix: np.ndarray, pair_vertical: bool = True) -> np.ndarray:
    """Convert a square 2D matrix into a 1D vector for embedding."""
    arr = np.asarray(matrix, dtype=np.float64).reshape(-1)
    if pair_vertical:
        paired = arr.reshape(-1, 2)
        if paired.shape[0] % 2 == 1:
            paired = np.pad(paired, ((0, 1), (0, 0)), mode="constant", constant_values=0)
        left = paired[:, 0]
        right = paired[:, 1]
        return np.concatenate([left, right[::-1]])
    return arr


def vector_to_matrix(vector: np.ndarray, size: int) -> np.ndarray:
    """Recover a matrix from a flattened vector."""
    arr = np.asarray(vector, dtype=np.float64).reshape(size, size)
    return arr


def save_image(matrix: np.ndarray, output_path: str | Path) -> None:
    """Save a matrix as an image in PNG format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.asarray(np.clip(matrix, 0, 255), dtype=np.uint8), mode="L")
    img.save(out)
