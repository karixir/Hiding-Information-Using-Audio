from __future__ import annotations

import math
import operator
import struct
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pywt
import soundfile as sf
from PIL import Image
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

RsaKey: TypeAlias = RSAPublicKey | RSAPrivateKey | bytes | bytearray | str | Path
_HEADER_BITS = 32
_IMAGE_HEADER = struct.Struct(">4sII")
_IMAGE_MAGIC = b"DWT1"
_OAEP_HASH = hashes.SHA256


class DwtRsaAudioStego:
    """Hide an RSA-encrypted grayscale image in Haar-DWT audio detail coefficients."""

    @staticmethod
    def generate_rsa_key_pair(key_size: int = 2048) -> tuple[bytes, bytes]:
        """Return a PEM-encoded (private, public) RSA key pair."""
        if key_size < 2048:
            raise ValueError("Rozmiar klucza RSA musi wynosić co najmniej 2048 bitów.")

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return private_pem, public_pem

    @staticmethod
    def _load_public_key(key: RsaKey) -> RSAPublicKey:
        if isinstance(key, RSAPublicKey):
            return key
        if isinstance(key, RSAPrivateKey):
            return key.public_key()
        key_data = DwtRsaAudioStego._read_key(key)
        loaded = serialization.load_pem_public_key(key_data)
        if not isinstance(loaded, RSAPublicKey):
            raise TypeError("Wymagany jest publiczny klucz RSA w formacie PEM.")
        return loaded

    @staticmethod
    def _load_private_key(key: RsaKey) -> RSAPrivateKey:
        if isinstance(key, RSAPrivateKey):
            return key
        key_data = DwtRsaAudioStego._read_key(key)
        loaded = serialization.load_pem_private_key(key_data, password=None)
        if not isinstance(loaded, RSAPrivateKey):
            raise TypeError("Wymagany jest prywatny klucz RSA w formacie PEM.")
        return loaded

    @staticmethod
    def _read_key(key: RsaKey) -> bytes:
        if isinstance(key, (str, Path)):
            return Path(key).read_bytes()
        if isinstance(key, (bytes, bytearray)):
            return bytes(key)
        raise TypeError("Klucz RSA musi być obiektem RSA, ścieżką lub danymi PEM.")

    @staticmethod
    def _oaep_padding() -> padding.OAEP:
        digest = _OAEP_HASH()
        return padding.OAEP(
            mgf=padding.MGF1(algorithm=digest),
            algorithm=digest,
            label=None,
        )

    @staticmethod
    def _encrypt_rsa(data: bytes, public_key: RSAPublicKey) -> bytes:
        key_bytes = (public_key.key_size + 7) // 8
        hash_bytes = _OAEP_HASH().digest_size
        max_chunk_size = key_bytes - 2 * hash_bytes - 2
        if max_chunk_size <= 0:
            raise ValueError("Klucz RSA jest zbyt krótki dla szyfrowania OAEP-SHA256.")
        return b"".join(
            public_key.encrypt(data[offset : offset + max_chunk_size], DwtRsaAudioStego._oaep_padding())
            for offset in range(0, len(data), max_chunk_size)
        )

    @staticmethod
    def _decrypt_rsa(ciphertext: bytes, private_key: RSAPrivateKey) -> bytes:
        key_bytes = (private_key.key_size + 7) // 8
        if not ciphertext or len(ciphertext) % key_bytes:
            raise ValueError("Długość szyfrogramu nie pasuje do rozmiaru klucza RSA.")
        return b"".join(
            private_key.decrypt(ciphertext[offset : offset + key_bytes], DwtRsaAudioStego._oaep_padding())
            for offset in range(0, len(ciphertext), key_bytes)
        )

    @staticmethod
    def _read_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Nie znaleziono pliku audio: {path}")
        samples, sample_rate = sf.read(path, dtype="float64", always_2d=False)
        if samples.size == 0:
            raise ValueError("Plik audio nie zawiera próbek.")
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        if samples.ndim != 1 or not np.all(np.isfinite(samples)):
            raise ValueError("Audio musi zawierać skończone próbki mono lub stereo.")
        return np.asarray(samples, dtype=np.float64), int(sample_rate)

    @staticmethod
    def _seed_value(seed: int) -> int:
        if isinstance(seed, bool):
            raise ValueError("Seed PN musi być nieujemną liczbą całkowitą.")
        try:
            value = operator.index(seed)
        except TypeError as exc:
            raise ValueError("Seed PN musi być nieujemną liczbą całkowitą.") from exc
        if value < 0:
            raise ValueError("Seed PN musi być nieujemną liczbą całkowitą.")
        return value

    @staticmethod
    def _encode_bits(
        bits: np.ndarray,
        details: np.ndarray,
        indices: np.ndarray,
        threshold: float,
        margin: float,
    ) -> None:
        magnitudes = np.abs(details[indices])
        zeros_to_change = indices[(bits < 0.5) & (magnitudes >= threshold)]
        ones_to_change = indices[(bits >= 0.5) & (magnitudes < threshold)]
        details[zeros_to_change] = (
            np.where(details[zeros_to_change] < 0.0, -1.0, 1.0)
            * max(0.0, threshold - margin)
        )
        details[ones_to_change] = (
            np.where(details[ones_to_change] < 0.0, -1.0, 1.0)
            * (threshold + margin)
        )

    @staticmethod
    def _read_bits(details: np.ndarray, indices: np.ndarray, threshold: float) -> np.ndarray:
        return (np.abs(details[indices]) >= threshold).astype(np.uint8)

    @staticmethod
    def _validate_threshold_ratio(threshold_ratio: float) -> float:
        ratio = float(threshold_ratio)
        if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError("threshold_ratio musi należeć do przedziału (0, 1).")
        return ratio

    @staticmethod
    def embed_image_into_audio(
        cover_audio_path: str | Path,
        secret_image_path: str | Path,
        output_path: str | Path,
        public_key: RsaKey,
        *,
        seed: int = 0,
        threshold_ratio: float = 0.01,
    ) -> dict[str, object]:
        """Encrypt a grayscale image with RSA-OAEP and embed its bits using the paper's DWT method."""
        seed_value = DwtRsaAudioStego._seed_value(seed)
        ratio = DwtRsaAudioStego._validate_threshold_ratio(threshold_ratio)
        rsa_public_key = DwtRsaAudioStego._load_public_key(public_key)

        with Image.open(secret_image_path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.uint8)
        height, width = pixels.shape
        plaintext = _IMAGE_HEADER.pack(_IMAGE_MAGIC, height, width) + pixels.tobytes()
        ciphertext = DwtRsaAudioStego._encrypt_rsa(plaintext, rsa_public_key)
        ciphertext_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8), bitorder="big")
        if ciphertext_bits.size > (1 << _HEADER_BITS) - 1:
            raise ValueError("Zaszyfrowany obraz jest zbyt duży dla nagłówka długości 32-bit.")
        header_bits = np.fromiter(
            ((ciphertext_bits.size >> shift) & 1 for shift in range(_HEADER_BITS - 1, -1, -1)),
            dtype=np.uint8,
            count=_HEADER_BITS,
        )

        cover, sample_rate = DwtRsaAudioStego._read_audio(cover_audio_path)
        approximation, detail = pywt.dwt(cover, "haar", mode="symmetric")
        detail = np.asarray(detail, dtype=np.float64)
        if detail.size <= _HEADER_BITS:
            raise ValueError("Audio jest zbyt krótkie na nagłówek DWT.")

        detail_peak = float(np.max(np.abs(detail)))
        if detail_peak == 0.0:
            raise ValueError("Nie można osadzać danych w cichym sygnale audio.")
        threshold = ratio * detail_peak
        margin = max(detail_peak * 1e-5, np.finfo(np.float64).eps)

        available_indices = np.arange(_HEADER_BITS, detail.size)
        if ciphertext_bits.size > available_indices.size:
            raise ValueError(
                f"Za mała pojemność audio: potrzeba {ciphertext_bits.size + _HEADER_BITS} "
                f"współczynników DWT, dostępne {detail.size}."
            )
        permutation = np.random.default_rng(seed_value).permutation(available_indices)
        payload_indices = permutation[: ciphertext_bits.size]

        header_indices = np.arange(_HEADER_BITS)
        DwtRsaAudioStego._encode_bits(
            header_bits, detail, header_indices, threshold, margin
        )
        DwtRsaAudioStego._encode_bits(
            ciphertext_bits, detail, payload_indices, threshold, margin
        )

        stego = pywt.idwt(approximation, detail, "haar", mode="symmetric")[: cover.size]
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, stego.astype(np.float32), sample_rate, subtype="FLOAT")
        return {
            "output": str(out_path),
            "sample_rate": sample_rate,
            "image_shape": pixels.shape,
            "ciphertext_bytes": len(ciphertext),
            "embedded_bits": int(ciphertext_bits.size + _HEADER_BITS),
            "threshold": threshold,
        }

    @staticmethod
    def extract_image_from_audio(
        stego_audio_path: str | Path,
        output_image_path: str | Path,
        private_key: RsaKey,
        *,
        seed: int = 0,
        threshold_ratio: float = 0.01,
    ) -> np.ndarray:
        """Extract, RSA-decrypt, and save the hidden grayscale image."""
        seed_value = DwtRsaAudioStego._seed_value(seed)
        ratio = DwtRsaAudioStego._validate_threshold_ratio(threshold_ratio)
        rsa_private_key = DwtRsaAudioStego._load_private_key(private_key)
        stego, _ = DwtRsaAudioStego._read_audio(stego_audio_path)
        _, detail = pywt.dwt(stego, "haar", mode="symmetric")
        detail = np.asarray(detail, dtype=np.float64)
        if detail.size <= _HEADER_BITS:
            raise ValueError("Audio jest zbyt krótkie, aby odczytać nagłówek DWT.")

        detail_peak = float(np.max(np.abs(detail)))
        if detail_peak == 0.0:
            raise ValueError("Nie znaleziono danych w cichym sygnale audio.")
        threshold = ratio * detail_peak
        header_bits = DwtRsaAudioStego._read_bits(
            detail, np.arange(_HEADER_BITS), threshold
        )
        ciphertext_bit_count = 0
        for bit in header_bits:
            ciphertext_bit_count = (ciphertext_bit_count << 1) | int(bit)

        key_bytes = (rsa_private_key.key_size + 7) // 8
        if (
            ciphertext_bit_count == 0
            or ciphertext_bit_count % 8 != 0
            or ciphertext_bit_count % (key_bytes * 8) != 0
        ):
            raise ValueError("Nagłówek długości ukrytego szyfrogramu jest niepoprawny.")
        ciphertext_bytes = ciphertext_bit_count // 8
        payload_count = ciphertext_bit_count
        if payload_count > detail.size - _HEADER_BITS:
            raise ValueError("Nagłówek wskazuje szyfrogram większy niż pojemność audio.")

        available_indices = np.arange(_HEADER_BITS, detail.size)
        payload_indices = np.random.default_rng(seed_value).permutation(available_indices)[
            :payload_count
        ]
        ciphertext_bits = DwtRsaAudioStego._read_bits(
            detail, payload_indices, threshold
        )
        ciphertext = np.packbits(ciphertext_bits, bitorder="big").tobytes()
        if len(ciphertext) != ciphertext_bytes:
            raise ValueError("Nie udało się odczytać pełnego szyfrogramu z audio.")

        plaintext = DwtRsaAudioStego._decrypt_rsa(ciphertext, rsa_private_key)
        if len(plaintext) < _IMAGE_HEADER.size:
            raise ValueError("Odszyfrowane dane są zbyt krótkie dla nagłówka obrazu.")
        magic, height, width = _IMAGE_HEADER.unpack_from(plaintext)
        if magic != _IMAGE_MAGIC or height == 0 or width == 0:
            raise ValueError("Niepoprawny nagłówek odszyfrowanego obrazu.")
        pixels = plaintext[_IMAGE_HEADER.size :]
        if len(pixels) != height * width:
            raise ValueError(
                f"Rozmiar obrazu nie zgadza się z nagłówkiem: "
                f"oczekiwano {height * width} bajtów, otrzymano {len(pixels)}."
            )

        recovered = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)
        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(recovered, mode="L").save(out_path)
        return recovered
