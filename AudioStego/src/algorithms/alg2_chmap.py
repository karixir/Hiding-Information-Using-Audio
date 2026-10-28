from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pywt

if __package__ in (None, ""):
    file_path = Path(__file__).resolve()
    project_root = file_path.parents[3]
    src_root = file_path.parents[1]
    for candidate in (project_root, src_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

try:
    from AudioStego.src.algorithms.chaotic_maps import (
        arnold_cat_map,
        baker_map,
        hybrid_chaotic_encryption,
        inverse_arnold_cat_map,
        inverse_baker_map,
        inverse_hybrid_chaotic_encryption,
    )
    from AudioStego.src.core.audio_utils import load_audio, save_audio
    from AudioStego.src.core.image_utils import ensure_square_image, load_image_as_grayscale, matrix_to_vector, save_image
    from AudioStego.src.core.metrics import mse, psnr, ssim
except ImportError:  # pragma: no cover
    from algorithms.chaotic_maps import (
        arnold_cat_map,
        baker_map,
        hybrid_chaotic_encryption,
        inverse_arnold_cat_map,
        inverse_baker_map,
        inverse_hybrid_chaotic_encryption,
    )
    from core.audio_utils import load_audio, save_audio
    from core.image_utils import ensure_square_image, load_image_as_grayscale, matrix_to_vector, save_image
    from core.metrics import mse, psnr, ssim


class ChaoticAudioStego:
    """Image encryption and audio steganography pipeline powered by chaotic maps."""

    @staticmethod
    def prepare_secret_image(secret_image_path: str | Path, target_size: Optional[int] = None) -> np.ndarray:
        """Load and normalize a secret image into a square grayscale matrix."""
        matrix = load_image_as_grayscale(secret_image_path)
        square_matrix = ensure_square_image(matrix, target_size=target_size)
        if square_matrix.shape[0] != square_matrix.shape[1]:
            raise ValueError(f"Obraz po normalizacji nie jest kwadratem: {square_matrix.shape}")
        return square_matrix.astype(np.float64)

    @staticmethod
    def encrypt_image(matrix: np.ndarray, map_type: str, p: int = 1, q: int = 1, iterations: int = 10, key_vector: Optional[list[int] | np.ndarray] = None) -> np.ndarray:
        """Apply the selected chaotic map encryption to a square grayscale matrix."""
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Obraz tajny musi być macierzą kwadratową NxN.")

        map_type = (map_type or "arnold").lower()
        if map_type == "arnold":
            return arnold_cat_map(arr, p=p, q=q, iterations=iterations)
        if map_type == "baker":
            if key_vector is None:
                raise ValueError("Dla mapy Baker wymagany jest klucz key_vector.")
            return baker_map(arr, key_vector)
        if map_type == "hybrid":
            if key_vector is None:
                raise ValueError("Dla mapy hybrydowej wymagany jest klucz key_vector.")
            return hybrid_chaotic_encryption(arr, p=p, q=q, iterations=iterations, key_vector=key_vector)
        raise ValueError(f"Nieznany typ mapy: {map_type}. Dostępne: arnold, baker, hybrid")

    @staticmethod
    def decrypt_image(matrix: np.ndarray, map_type: str, p: int = 1, q: int = 1, iterations: int = 10, key_vector: Optional[list[int] | np.ndarray] = None) -> np.ndarray:
        """Reverse the selected chaotic map transformation."""
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Obraz zaszyfrowany musi być macierzą kwadratową NxN.")

        map_type = (map_type or "arnold").lower()
        if map_type == "arnold":
            return inverse_arnold_cat_map(arr, p=p, q=q, iterations=iterations)
        if map_type == "baker":
            if key_vector is None:
                raise ValueError("Dla mapy Baker wymagany jest klucz key_vector.")
            return inverse_baker_map(arr, key_vector)
        if map_type == "hybrid":
            if key_vector is None:
                raise ValueError("Dla mapy hybrydowej wymagany jest klucz key_vector.")
            return inverse_hybrid_chaotic_encryption(arr, p=p, q=q, iterations=iterations, key_vector=key_vector)
        raise ValueError(f"Nieznany typ mapy: {map_type}. Dostępne: arnold, baker, hybrid")

    @staticmethod
    def image_to_audio_like_signal(matrix: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """Convert the encrypted matrix into a binary payload bitstream for embedding."""
        arr = np.asarray(matrix, dtype=np.uint8)
        flat = arr.reshape(-1)
        bits = np.unpackbits(flat.astype(np.uint8))
        return bits.astype(np.float64)

    @staticmethod
    def _encode_length_header(length: int) -> np.ndarray:
        """Encode a 32-bit unsigned integer length in big-endian bit order."""
        if length < 0:
            raise ValueError("Długość payloadu nie może być ujemna.")
        if length > (1 << 32) - 1:
            raise ValueError(f"Długość payloadu jest zbyt duża dla nagłówka 32-bit: {length}")
        return np.asarray([(length >> shift) & 1 for shift in range(31, -1, -1)], dtype=np.float64)

    @staticmethod
    def _decode_length_header(bits: np.ndarray) -> int:
        """Decode a 32-bit unsigned integer length from the first 32 bits."""
        bit_values = np.asarray(bits, dtype=np.float64).reshape(-1)
        if bit_values.size < 32:
            raise ValueError("Nagłówek długości jest zbyt krótki.")
        value = 0
        for bit in bit_values[:32]:
            value = (value << 1) | int(np.round(float(bit)))
        return int(value)

    @staticmethod
    def embed_payload_in_audio(cover_audio: np.ndarray, payload_signal: np.ndarray, alpha: float = 0.08, wavelet: str = "db4") -> np.ndarray:
        """Embed the payload in the cover audio using a stable LSB bitstream layout."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha musi być w zakresie (0, 1].")

        if payload_signal.size == 0:
            raise ValueError("Sygnał payload nie może być pusty.")

        cover = np.asarray(cover_audio, dtype=np.float64).reshape(-1)
        payload = np.asarray(payload_signal, dtype=np.float64).reshape(-1)
        header = ChaoticAudioStego._encode_length_header(payload.size)
        data = np.concatenate([header, payload], axis=0)

        cover_pcm = np.rint(np.clip(cover * 32767.0, -32768, 32767)).astype(np.int32)
        capacity = cover_pcm.size
        if data.size > capacity:
            raise ValueError(f"Pojemność audio za mała. Potrzebne: {data.size}, dostępne: {capacity}.")

        stego_pcm = cover_pcm.copy()
        for i, bit in enumerate(data):
            bit_val = int(np.round(float(bit)))
            stego_pcm[i] = (stego_pcm[i] & ~1) | bit_val

        return stego_pcm.astype(np.int16)

    @staticmethod
    def extract_payload_from_audio(stego_audio: np.ndarray, wavelet: str = "db4") -> np.ndarray:
        """Extract the binary payload from the stego audio using the same bitstream layout."""
        samples = np.asarray(stego_audio, dtype=np.float64).reshape(-1)
        if samples.size < 32:
            raise ValueError("Nagranie stego jest zbyt krótkie, aby zawierać nagłówek długości payloadu.")

        if np.asarray(stego_audio).dtype.kind in {"i", "u"}:
            pcm = np.asarray(stego_audio, dtype=np.int32).reshape(-1)
        else:
            pcm = np.rint(np.clip(samples * 32767.0, -32768, 32767)).astype(np.int32)

        header_bits = (pcm[:32] & 1).astype(np.float64)
        length = ChaoticAudioStego._decode_length_header(header_bits)
        if length <= 0:
            return np.empty(0, dtype=np.float64)

        payload_end = min(32 + length, pcm.size)
        payload_bits = (pcm[32:payload_end] & 1).astype(np.float64)
        if payload_bits.size < length:
            raise ValueError(f"Payload nie został w pełni odzyskany: oczekiwano {length}, odzyskano {payload_bits.size}.")
        full_bits = np.concatenate([header_bits, payload_bits[:length]], axis=0)
        return full_bits.astype(np.float64)

    @staticmethod
    def embed_image_into_audio(cover_audio_path: str | Path, secret_image_path: str | Path, output_path: str | Path, map_type: str = "hybrid", p: int = 1, q: int = 1, iterations: int = 10, key_vector: Optional[list[int] | np.ndarray] = None, alpha: float = 0.08) -> dict:
        """Encrypt an image, transform it into an audio-like signal, and embed it into a cover audio file."""
        print("[1/6] Ładowanie obrazu tajnego...")
        secret_matrix = ChaoticAudioStego.prepare_secret_image(secret_image_path)

        print("[2/6] Szyfrowanie obrazu za pomocą mapy chaotycznej...")
        encrypted = ChaoticAudioStego.encrypt_image(secret_matrix, map_type=map_type, p=p, q=q, iterations=iterations, key_vector=key_vector)

        print("[3/6] Przygotowanie obrazu do sygnału audio-podobnego...")
        payload_signal = ChaoticAudioStego.image_to_audio_like_signal(encrypted)
        payload_signal = payload_signal / max(np.max(np.abs(payload_signal)), 1.0)

        print("[4/6] Wczytanie pliku cover audio...")
        cover_audio, sample_rate = load_audio(cover_audio_path)

        print("[5/6] Osadzanie danych w wysokich częstotliwościach DWT...")
        stego_audio = ChaoticAudioStego.embed_payload_in_audio(cover_audio, payload_signal, alpha=alpha)
        save_audio(output_path, stego_audio, sample_rate)

        print("[6/6] Zakończono pipeline osadzania.")
        return {
            "cover_info": cover_audio.shape,
            "sample_rate": sample_rate,
            "secret_shape": secret_matrix.shape,
            "encrypted_shape": encrypted.shape,
            "output": str(output_path),
        }

    @staticmethod
    def extract_image_from_audio(stego_audio_path: str | Path, restored_output_path: str | Path, map_type: str = "hybrid", p: int = 1, q: int = 1, iterations: int = 10, key_vector: Optional[list[int] | np.ndarray] = None, wavelet: str = "db4") -> np.ndarray:
        """Extract the embedded image signal from the stego audio and reverse the encryption."""
        print("[1/4] Wczytanie stego-audio...")
        stego_audio, sample_rate = load_audio(stego_audio_path)

        print("[2/4] Ekstrakcja payloadu z cech DWT...")
        payload_bits = ChaoticAudioStego.extract_payload_from_audio(stego_audio, wavelet=wavelet)
        if payload_bits.size == 0:
            raise ValueError("Nie znaleziono ukrytej informacji w pliku stego-audio.")

        print("[3/4] Rekonstrukcja sygnału obrazu i odwrócenie szyfrowania...")
        if payload_bits.size < 32:
            raise ValueError("Payload jest zbyt krótki, aby odtworzyć obraz.")

        length = ChaoticAudioStego._decode_length_header(payload_bits[:32])
        if length <= 0:
            raise ValueError("Nagłówek długości payloadu jest niepoprawny.")

        data_bits = payload_bits[32 : 32 + length]
        if data_bits.size != length:
            raise ValueError(f"Niepełny payload: oczekiwano {length}, odzyskano {data_bits.size}.")
        if data_bits.size % 8 != 0:
            data_bits = data_bits[: data_bits.size - (data_bits.size % 8)]
        if data_bits.size == 0:
            raise ValueError("Payload po odszyfrowaniu jest pusty.")

        values = np.packbits(data_bits.astype(np.uint8), bitorder="big")
        size = int(math.sqrt(values.size))
        if size * size != values.size:
            raise ValueError(f"Wielkość odzyskanego obrazu jest niepoprawna: {values.size}")

        matrix = values.reshape(size, size)
        recovered = ChaoticAudioStego.decrypt_image(matrix, map_type=map_type, p=p, q=q, iterations=iterations, key_vector=key_vector)
        recovered = np.clip(recovered, 0, 255)

        print("[4/4] Zapis odzyskanego obrazu...")
        save_image(recovered, restored_output_path)
        print(f"[OK] Odtworzony obraz zapisano do: {restored_output_path}")
        print(f"[INFO] Współczynnik próbkowania: {sample_rate} Hz")
        return recovered

    @staticmethod
    def evaluate_recovery(original_image_path: str | Path, recovered_image_path: str | Path) -> dict:
        """Compute image quality metrics between original and recovered images."""
        original = np.asarray(load_image_as_grayscale(original_image_path), dtype=np.float64)
        recovered = np.asarray(load_image_as_grayscale(recovered_image_path), dtype=np.float64)
        if original.shape != recovered.shape:
            raise ValueError(f"Kształt obrazu oryginalnego i odzyskanego różni się: {original.shape} vs {recovered.shape}")
        return {
            "MSE": mse(original, recovered),
            "PSNR": psnr(original, recovered),
            "SSIM": ssim(original, recovered),
        }


def _parse_key_vector(raw_key: str | list[int] | tuple[int, ...] | np.ndarray | None) -> list[int] | None:
    """Parse a comma-separated key vector into a list of integers."""
    if raw_key is None:
        return None
    if isinstance(raw_key, (list, tuple, np.ndarray)):
        values = [int(value) for value in raw_key]
    else:
        values = [int(item.strip()) for item in str(raw_key).split(",") if item.strip()]
    if not values:
        return None
    return values


def _default_encrypted_output_path(cover_audio_path: str | Path) -> Path:
    """Build the default stego output path next to the original cover audio."""
    cover_path = Path(cover_audio_path)
    suffix = cover_path.suffix or ".wav"
    return cover_path.with_name(f"{cover_path.stem}_encrypted{suffix}")


def _resolve_encrypt_output_path(cover_audio_path: str | Path, output_path: str | Path | None) -> Path:
    """Resolve the encrypt output target: full file path, directory, or default naming."""
    if output_path is None:
        return _default_encrypted_output_path(cover_audio_path)

    candidate = Path(output_path)
    if candidate.exists() and candidate.is_dir():
        return candidate / _default_encrypted_output_path(cover_audio_path).name

    if candidate.suffix == "":
        return candidate / _default_encrypted_output_path(cover_audio_path).name

    return candidate


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for encrypting and decrypting the image payload."""
    parser = argparse.ArgumentParser(
        description="Standalone chaotic image encryption + audio steganography pipeline for alg2_chmap.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Available actions: encrypt, decrypt")

    encrypt_parser = subparsers.add_parser(
        "encrypt",
        help="Encrypt an image into a cover audio file.",
        description="Encrypt a secret image with a chaotic map and embed it inside a WAV cover file.",
        add_help=True,
    )
    encrypt_parser.add_argument("-s", "--secret-image", required=True, help="Path to the secret image to hide.")
    encrypt_parser.add_argument("-c", "--cover-audio", required=True, help="Path to the cover audio WAV file.")
    encrypt_parser.add_argument("-o", "--output-path", default=None, help="Optional stego output path. If omitted, a sibling file named <cover>_encrypted.<ext> is created. If it points to a directory or has no extension, the default file name is created inside that directory.")
    encrypt_parser.add_argument("-m", "--map-type", default="hybrid", choices=["arnold", "baker", "hybrid"], help="Chaotic map type.")
    encrypt_parser.add_argument("-p", "--p", type=int, default=1, help="Arnold map parameter p.")
    encrypt_parser.add_argument("-q", "--q", type=int, default=1, help="Arnold map parameter q.")
    encrypt_parser.add_argument("-i", "--iterations", type=int, default=8, help="Number of chaotic iterations.")
    encrypt_parser.add_argument("-k", "--key-vector", default="8,8,8,8,8,8,8,8", help="Comma-separated key vector for Baker or hybrid maps.")
    encrypt_parser.add_argument("-a", "--alpha", type=float, default=0.08, help="Embedding strength factor in audio domain.")

    decrypt_parser = subparsers.add_parser(
        "decrypt",
        help="Extract and decrypt an embedded image from stego audio.",
        description="Extract the embedded payload from a stego audio file and recover the original secret image.",
        add_help=True,
    )
    decrypt_parser.add_argument("-s", "--stego-audio", required=True, help="Path to the embedded stego audio file.")
    decrypt_parser.add_argument("-o", "--output-path", required=True, help="Path to save the recovered image.")
    decrypt_parser.add_argument("-m", "--map-type", default="hybrid", choices=["arnold", "baker", "hybrid"], help="Chaotic map type used during encryption.")
    decrypt_parser.add_argument("-p", "--p", type=int, default=1, help="Arnold map parameter p.")
    decrypt_parser.add_argument("-q", "--q", type=int, default=1, help="Arnold map parameter q.")
    decrypt_parser.add_argument("-i", "--iterations", type=int, default=8, help="Number of chaotic iterations.")
    decrypt_parser.add_argument("-k", "--key-vector", default="8,8,8,8,8,8,8,8", help="Comma-separated key vector for Baker or hybrid maps.")

    return parser


def _run_encrypt(args: argparse.Namespace) -> int:
    """Execute the encrypt pipeline from the CLI arguments."""
    stego_output = _resolve_encrypt_output_path(args.cover_audio, args.output_path)
    stego_output.parent.mkdir(parents=True, exist_ok=True)
    key_vector = _parse_key_vector(args.key_vector)

    result = ChaoticAudioStego.embed_image_into_audio(
        cover_audio_path=args.cover_audio,
        secret_image_path=args.secret_image,
        output_path=stego_output,
        map_type=args.map_type,
        p=args.p,
        q=args.q,
        iterations=args.iterations,
        key_vector=key_vector,
        alpha=args.alpha,
    )

    print("\n[OK] Encryption finished.")
    print(f"Stego audio saved to: {stego_output}")
    for key, value in result.items():
        if key == "output":
            continue
        print(f"{key}: {value}")
    return 0


def _run_decrypt(args: argparse.Namespace) -> int:
    """Execute the decrypt pipeline from the CLI arguments."""
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_vector = _parse_key_vector(args.key_vector)

    recovered = ChaoticAudioStego.extract_image_from_audio(
        stego_audio_path=args.stego_audio,
        restored_output_path=output_path,
        map_type=args.map_type,
        p=args.p,
        q=args.q,
        iterations=args.iterations,
        key_vector=key_vector,
    )

    print("\n[OK] Decryption finished.")
    print(f"Recovered image saved to: {output_path}")
    print(f"Recovered shape: {recovered.shape}")
    return 0


def main() -> int:
    """Entry point for running the algorithm as a standalone script."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.mode == "encrypt":
            return _run_encrypt(args)
        if args.mode == "decrypt":
            return _run_decrypt(args)
        parser.error(f"Unsupported mode: {args.mode}")
        return 2
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
