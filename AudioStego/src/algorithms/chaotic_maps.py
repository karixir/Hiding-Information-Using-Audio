from __future__ import annotations

import numpy as np


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Macierz musi być 2D, otrzymano {arr.ndim}D.")
    return arr


def arnold_cat_map(matrix: np.ndarray, p: int = 1, q: int = 1, iterations: int = 10) -> np.ndarray:
    """Apply Arnold Cat Map to a square matrix."""
    arr = _validate_matrix(matrix)
    n = arr.shape[0]
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Mapa Arnolda działa tylko na macierzy kwadratowej.")

    transformed = arr.copy()

    for _ in range(iterations):
        x, y = np.indices((n, n))
        new_x = (x + p * y) % n
        new_y = (q * x + (p * q + 1) * y) % n
        new_matrix = np.zeros_like(transformed, dtype=np.float64)
        new_matrix[new_x, new_y] = transformed[x, y]
        transformed = new_matrix

    return transformed


def inverse_arnold_cat_map(matrix: np.ndarray, p: int = 1, q: int = 1, iterations: int = 10) -> np.ndarray:
    """Reverse Arnold Cat Map transform using the true modular inverse of the forward matrix."""
    arr = _validate_matrix(matrix)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Mapa Arnolda działa tylko na macierzy kwadratowej.")

    n = arr.shape[0]
    transformed = arr.copy()

    for _ in range(iterations):
        x, y = np.indices((n, n))
        new_x = ((p * q + 1) * x - p * y) % n
        new_y = ((-q) * x + y) % n
        new_matrix = np.zeros_like(transformed, dtype=np.float64)
        new_matrix[new_x, new_y] = transformed[x, y]
        transformed = new_matrix

    return transformed


def normalize_baker_key(key_vector: list[int] | np.ndarray, size: int) -> list[int]:
    """Normalize a Baker-map key into a valid partition whose elements sum to the matrix size."""
    if size <= 0:
        raise ValueError("Rozmiar macierzy musi być dodatni.")

    if key_vector is None:
        keys = [size]
    else:
        keys = [int(v) for v in key_vector]

    if not keys or all(v <= 0 for v in keys):
        raise ValueError("Wektor klucza dla mapy Bakera musi zawierać dodatnie wartości.")

    total = sum(keys)
    if total == size:
        return keys

    normalized = np.asarray(keys, dtype=np.float64)
    scaled = np.round((normalized / total) * size).astype(int)
    diff = size - int(np.sum(scaled))
    if diff != 0:
        scaled[-1] += diff
    if np.sum(scaled) != size or np.any(scaled <= 0):
        raise ValueError(f"Nie można znormalizować klucza dla rozmiaru {size}: {keys}")
    return [int(v) for v in scaled]


def baker_map(matrix: np.ndarray, key_vector: list[int] | np.ndarray) -> np.ndarray:
    """Apply a reversible Baker-style block permutation to a square matrix."""
    arr = _validate_matrix(matrix)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Mapa Bakera działa tylko na macierzy kwadratowej.")

    n = arr.shape[0]
    key = normalize_baker_key(key_vector, n)

    permutation = []
    start = 0
    for block in key:
        end = start + block
        permutation.extend(np.arange(start, end)[::-1])
        start = end

    perm = np.asarray(permutation, dtype=int)
    return arr[np.ix_(perm, perm)]


def inverse_baker_map(matrix: np.ndarray, key_vector: list[int] | np.ndarray) -> np.ndarray:
    """Inverse of the Baker permutation using the same key."""
    arr = _validate_matrix(matrix)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Mapa Bakera działa tylko na macierzy kwadratowej.")

    n = arr.shape[0]
    key = normalize_baker_key(key_vector, n)

    permutation = []
    start = 0
    for block in key:
        end = start + block
        permutation.extend(np.arange(start, end)[::-1])
        start = end

    perm = np.asarray(permutation, dtype=int)
    return arr[np.ix_(perm, perm)]


def hybrid_chaotic_encryption(
    matrix: np.ndarray,
    p: int = 1,
    q: int = 1,
    iterations: int = 10,
    key_vector: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Sequential Arnold + Baker chaotic encryption."""
    arr = _validate_matrix(matrix)
    encrypted = arnold_cat_map(arr, p=p, q=q, iterations=iterations)
    if key_vector is not None:
        encrypted = baker_map(encrypted, key_vector)
    return encrypted


def inverse_hybrid_chaotic_encryption(
    matrix: np.ndarray,
    p: int = 1,
    q: int = 1,
    iterations: int = 10,
    key_vector: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Reverse hybrid encryption in the reverse pipeline order."""
    arr = _validate_matrix(matrix)
    if key_vector is not None:
        arr = inverse_baker_map(arr, key_vector)
    return inverse_arnold_cat_map(arr, p=p, q=q, iterations=iterations)
