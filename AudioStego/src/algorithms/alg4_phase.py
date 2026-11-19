from __future__ import annotations

import hashlib
import argparse
import struct
import sys
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import medfilt, resample_poly, wiener

if __package__ in (None, ""):
    file_path = Path(__file__).resolve()
    project_root = file_path.parents[3]
    src_root = file_path.parents[1]
    for candidate in (project_root, src_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

try:
    from AudioStego.src.core.metrics import bit_error_rate, snr_db
except ImportError:  # pragma: no cover
    from core.metrics import bit_error_rate, snr_db


MAGIC = b"MBCP"
HEADER_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sBIHIHIHHHHI")
HEADER_SIZE = HEADER_STRUCT.size


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku WAV: {path}")
    rate, samples = wavfile.read(str(path))
    if samples.dtype != np.int16:
        raise ValueError(
            f"Oczekiwano 16-bitowego PCM (int16), otrzymano {samples.dtype}."
        )
    if samples.ndim not in (1, 2) or samples.size == 0:
        raise ValueError("Niepoprawny lub pusty plik WAV.")
    return int(rate), np.asarray(samples, dtype=np.int16)


def write_wav(path: str | Path, rate: int, samples: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples)
    if data.dtype != np.int16:
        data = np.clip(np.rint(data), -32768, 32767).astype(np.int16)
    wavfile.write(str(out), int(rate), data)


def to_mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples)
    if data.ndim == 1:
        return data.astype(np.int16, copy=False)
    if data.ndim == 2:
        if data.shape[1] == 0:
            raise ValueError("Plik audio nie ma kanałów.")
        return np.rint(data.astype(np.float64).mean(axis=1)).clip(-32768, 32767).astype(np.int16)
    raise ValueError("Niepoprawny kształt danych audio.")


def prepare_secret(
    samples: np.ndarray,
    rate: int,
    target_rate: int = 8000,
    target_bits: int = 16,
) -> tuple[np.ndarray, int]:
    mono = to_mono(samples)
    if isinstance(target_rate, bool) or not isinstance(target_rate, int) or target_rate <= 0:
        raise ValueError("target_rate musi być dodatnią liczbą całkowitą.")
    if target_bits not in (8, 16):
        raise ValueError("target_bits musi być 8 lub 16.")

    if target_rate != rate:
        g = np.gcd(int(rate), int(target_rate))
        mono = resample_poly(mono.astype(np.float64), target_rate // g, rate // g)

    mono = np.asarray(np.rint(mono), dtype=np.float64).clip(-32768, 32767)
    if target_bits == 8:
        values = np.rint(((mono + 32768.0) / 65536.0) * 255.0).clip(0, 255)
        mono = np.rint(values * 256.0 - 32768.0).clip(-32768, 32767).astype(np.int16)
    else:
        mono = mono.astype(np.int16)
    return mono, int(target_rate)


def bytes_to_bits(data: bytes | bytearray | np.ndarray) -> np.ndarray:
    if isinstance(data, (bytes, bytearray)):
        raw = np.frombuffer(data, dtype=np.uint8)
    else:
        raw = np.asarray(data, dtype=np.uint8).reshape(-1)
    return np.unpackbits(raw, bitorder="big").astype(np.uint8)


def bits_to_bytes(bits: Iterable[int] | np.ndarray) -> bytes:
    values = np.asarray(list(bits) if not isinstance(bits, np.ndarray) else bits, dtype=np.uint8).reshape(-1)
    if values.size == 0:
        return b""
    if np.any((values != 0) & (values != 1)):
        raise ValueError("Tablica bitów musi zawierać tylko 0 i 1.")
    padded = values.size % 8
    if padded:
        values = np.pad(values, (0, 8 - padded), constant_values=0)
    return np.packbits(values, bitorder="big").tobytes()


def samples_to_bits(samples: np.ndarray, bit_depth: int | None = None) -> np.ndarray:
    data = np.asarray(samples)
    if data.ndim != 1:
        data = to_mono(data)
    if bit_depth is None:
        bit_depth = 16 if data.dtype == np.int16 else 8
    if bit_depth == 16:
        raw = data.astype(np.int16).view(np.uint16).astype(np.uint16).view(np.uint8)
        return np.unpackbits(raw, bitorder="big").astype(np.uint8)
    if bit_depth == 8:
        values = np.rint(((data.astype(np.float64) + 32768.0) / 65536.0) * 255.0).clip(0, 255).astype(np.uint8)
        return np.unpackbits(values, bitorder="big").astype(np.uint8)
    raise ValueError("bit_depth musi być 8 lub 16.")


def bits_to_samples(bits: Iterable[int] | np.ndarray, bit_depth: int) -> np.ndarray:
    raw = np.asarray(list(bits) if not isinstance(bits, np.ndarray) else bits, dtype=np.uint8).reshape(-1)
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("Tablica bitów musi zawierać tylko 0 i 1.")
    if raw.size % 8:
        raw = np.pad(raw, (0, 8 - raw.size % 8), constant_values=0)
    values = np.packbits(raw, bitorder="big")
    if bit_depth == 16:
        if values.size % 2:
            values = np.pad(values, (0, 1), constant_values=0)
        return values.astype(np.uint8).view(np.uint16).astype(np.int32).astype(np.int16)
    if bit_depth == 8:
        unsigned = values.astype(np.int32)
        return np.rint(unsigned * 256.0 - 32768.0).clip(-32768, 32767).astype(np.int16)
    raise ValueError("bit_depth musi być 8 lub 16.")


def build_header(
    secret_rate: int,
    secret_channels: int,
    secret_length: int,
    secret_bit_depth: int,
    segment_size: int,
    bins_per_segment: int,
    bits_per_bin: int,
    start_bin: int,
    bin_stride: int,
) -> bytes:
    if secret_rate <= 0 or secret_length < 0:
        raise ValueError("Niepoprawne metadane tajnego audio.")
    if not 1 <= secret_channels <= 65535:
        raise ValueError("Niepoprawna liczba kanałów tajnego audio.")
    if secret_bit_depth not in (8, 16):
        raise ValueError("secret_bit_depth musi być 8 lub 16.")
    if segment_size <= 0 or bins_per_segment <= 0 or bits_per_bin <= 0:
        raise ValueError("Niepoprawna konfiguracja kodowania.")
    fields = (
        MAGIC,
        HEADER_VERSION,
        int(secret_rate),
        int(secret_channels),
        int(secret_length),
        int(secret_bit_depth),
        int(segment_size),
        int(bins_per_segment),
        int(bits_per_bin),
        int(start_bin),
        int(bin_stride),
    )
    if not all(isinstance(v, (int, np.integer)) for v in fields[1:]):
        raise ValueError("Wszystkie pola nagłówka muszą być liczbami całkowitymi.")
    crc = zlib.crc32(HEADER_STRUCT.pack(*fields, 0)[: -4]) & 0xFFFFFFFF
    return HEADER_STRUCT.pack(*fields, crc)


def parse_header(data: bytes) -> dict:
    if len(data) < HEADER_SIZE:
        raise ValueError("Nie wykryto ukrytej wiadomości: zbyt krótki nagłówek.")
    (
        magic,
        version,
        secret_rate,
        secret_channels,
        secret_length,
        secret_bit_depth,
        segment_size,
        bins_per_segment,
        bits_per_bin,
        start_bin,
        bin_stride,
        crc,
    ) = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    if magic != MAGIC or version != HEADER_VERSION:
        raise ValueError("Nie wykryto ukrytej wiadomości: niepoprawny magic lub wersja.")
    stored = HEADER_STRUCT.pack(
        magic, version, secret_rate, secret_channels, secret_length,
        secret_bit_depth, segment_size, bins_per_segment, bits_per_bin,
        start_bin, bin_stride, 0,
    )
    if zlib.crc32(stored[:-4]) & 0xFFFFFFFF != crc:
        raise ValueError("Nie wykryto ukrytej wiadomości: niepoprawny CRC.")
    if secret_rate <= 0 or secret_length <= 0 or secret_channels <= 0:
        raise ValueError("Nie wykryto ukrytej wiadomości: niepoprawne metadane.")
    if secret_bit_depth not in (8, 16):
        raise ValueError("Nie wykryto ukrytej wiadomości: niepoprawna głębia bitowa.")
    if segment_size <= 0 or bins_per_segment <= 0 or bits_per_bin <= 0:
        raise ValueError("Nie wykryto ukrytej wiadomości: niepoprawna konfiguracja.")
    return {
        "magic": magic,
        "version": version,
        "secret_rate": int(secret_rate),
        "secret_channels": int(secret_channels),
        "secret_length": int(secret_length),
        "secret_bit_depth": int(secret_bit_depth),
        "segment_size": int(segment_size),
        "bins_per_segment": int(bins_per_segment),
        "bits_per_bin": int(bits_per_bin),
        "start_bin": int(start_bin),
        "bin_stride": int(bin_stride),
        "crc": int(crc),
    }


def protect_header(header: bytes, password: str | None = None) -> bytes:
    if not password:
        return bytes(header)
    key = hashlib.sha256(password.encode("utf-8")).digest()
    return bytes(byte ^ key[i % len(key)] for i, byte in enumerate(header))


def unprotect_header(header: bytes, password: str | None = None) -> bytes:
    return protect_header(header, password)


def bootstrap_config() -> dict:
    return {
        "segment_size": 2048,
        "bins_per_segment": 4,
        "bits_per_bin": 1,
        "start_bin": 1,
        "bin_stride": 1,
        "min_magnitude": 1000.0,
        "password": None,
    }


def header_segment_count(config: dict | None = None) -> int:
    cfg = config or bootstrap_config()
    bits_per_segment = int(cfg["bins_per_segment"]) * int(cfg["bits_per_bin"])
    if bits_per_segment <= 0:
        raise ValueError("bits_per_segment musi być dodatnie.")
    return int(np.ceil(HEADER_SIZE * 8 / bits_per_segment))


def phase_levels(bits_per_bin: int) -> np.ndarray:
    if isinstance(bits_per_bin, bool) or not isinstance(bits_per_bin, int) or bits_per_bin <= 0:
        raise ValueError("bits_per_bin musi być dodatnią liczbą całkowitą.")
    if bits_per_bin > 12:
        raise ValueError("bits_per_bin nie może przekraczać 12.")
    return np.arange(1 << bits_per_bin, dtype=np.float64) * (2 * np.pi / (1 << bits_per_bin))


def bits_to_phase(bits: Iterable[int] | np.ndarray) -> float:
    values = np.asarray(list(bits) if not isinstance(bits, np.ndarray) else bits, dtype=np.uint8).reshape(-1)
    if values.size == 0:
        raise ValueError("Grupa bitów nie może być pusta.")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("Bity muszą być 0 lub 1.")
    levels_count = 1 << values.size
    index = 0
    for bit in values:
        index = (index << 1) | int(bit)
    return float(index * (2 * np.pi / levels_count))


def wrap_phase_difference(a: float, b: float) -> float:
    result = (np.asarray(a, dtype=np.float64) - float(b) + np.pi) % (2 * np.pi) - np.pi
    if result.ndim == 0:
        return float(result)
    return result


def phase_to_bits(phase: float, bits_per_bin: int) -> np.ndarray:
    levels = phase_levels(bits_per_bin)
    distances = np.abs(wrap_phase_difference(levels, float(phase)))
    index = int(np.argmin(distances))
    width = 1 << bits_per_bin
    return np.asarray([(index >> shift) & 1 for shift in range(bits_per_bin - 1, -1, -1)], dtype=np.uint8)


def select_bins(config: dict, spectrum_magnitudes: np.ndarray) -> np.ndarray:
    segment_size = int(config["segment_size"])
    count = int(config["bins_per_segment"])
    start = int(config["start_bin"])
    stride = int(config["bin_stride"])
    if segment_size <= 0 or count <= 0 or start <= 0 or stride <= 0:
        raise ValueError("Niepoprawna konfiguracja selekcji binów.")
    nyquist = segment_size // 2
    bins = np.asarray([start + i * stride for i in range(count)], dtype=np.int64)
    if bins.size == 0 or bins[0] < 1 or np.max(bins) >= nyquist:
        raise ValueError(f"Wybrane biny muszą mieścić się w przedziale 1..{nyquist - 1}.")

    magnitudes = np.asarray(spectrum_magnitudes).reshape(-1)
    if magnitudes.size < nyquist + 1:
        raise ValueError("Spektrum jest zbyt krótkie dla konfiguracji segmentu.")
    min_magnitude = float(config.get("min_magnitude", 0.0))
    if np.any(magnitudes[bins] < min_magnitude):
        raise ValueError("Co najmniej jeden wybrany bin ma zbyt małą amplitudę.")
    return bins


def segment_signal(samples: np.ndarray, segment_size: int) -> tuple[list[np.ndarray], np.ndarray]:
    data = np.asarray(samples).reshape(-1)
    if segment_size <= 0:
        raise ValueError("segment_size musi być dodatnie.")
    length = (data.size // segment_size) * segment_size
    segments = [data[i:i + segment_size] for i in range(0, length, segment_size)]
    tail = data[length:]
    return segments, tail


def find_header_segments(samples: np.ndarray, config: dict, count: int) -> list[int]:
    data = np.asarray(samples).reshape(-1)
    segment_size = int(config["segment_size"])
    if count <= 0:
        raise ValueError("Liczba segmentów nagłówka musi być dodatnia.")

    bins_config = dict(config)
    bins_config["min_magnitude"] = 0.0
    threshold = float(config.get("min_magnitude", 0.0))
    usable: list[int] = []
    for index in range(data.size // segment_size):
        segment = data[index * segment_size:(index + 1) * segment_size]
        magnitudes = np.abs(np.fft.rfft(segment.astype(np.float64)))
        bins = select_bins(bins_config, magnitudes)
        if np.all(magnitudes[bins] >= threshold) and np.any(magnitudes[bins] > 0):
            usable.append(index)
            if len(usable) == count:
                break
    return usable


def embed_segment(segment: np.ndarray, bit_chunk: Iterable[int] | np.ndarray, config: dict) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float64).reshape(-1)
    if data.size != int(config["segment_size"]):
        raise ValueError("Segment ma niezgodny rozmiar.")
    bits = np.asarray(list(bit_chunk) if not isinstance(bit_chunk, np.ndarray) else bit_chunk, dtype=np.uint8).reshape(-1)
    expected = int(config["bins_per_segment"]) * int(config["bits_per_bin"])
    if bits.size != expected:
        raise ValueError(f"Oczekiwano {expected} bitów na segment, otrzymano {bits.size}.")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Bity segmentu muszą być 0 lub 1.")

    spectrum = np.fft.rfft(data)
    bins = select_bins(config, np.abs(spectrum))
    bpb = int(config["bits_per_bin"])
    for i, bin_index in enumerate(bins):
        group = bits[i * bpb:(i + 1) * bpb]
        spectrum[bin_index] = np.abs(spectrum[bin_index]) * np.exp(1j * bits_to_phase(group))
    rebuilt = np.fft.irfft(spectrum, n=data.size)
    peak = float(np.max(np.abs(rebuilt)))
    if peak > 32767:
        rebuilt *= 32767.0 / peak
    return np.rint(np.clip(rebuilt, -32768, 32767)).astype(np.int16)


def extract_segment(segment: np.ndarray, config: dict) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float64).reshape(-1)
    if data.size != int(config["segment_size"]):
        raise ValueError("Segment ma niezgodny rozmiar.")
    spectrum = np.fft.rfft(data)
    bins = select_bins(config, np.abs(spectrum))
    bpb = int(config["bits_per_bin"])
    return np.concatenate([phase_to_bits(np.angle(spectrum[int(b)]), bpb) for b in bins])


def compute_capacity(
    host_length: int,
    segment_size: int,
    bits_per_segment: int,
    header_segments: int,
    sample_rate: int | None = None,
) -> tuple[int, int, float]:
    if host_length < 0 or segment_size <= 0 or bits_per_segment <= 0 or header_segments < 0:
        raise ValueError("Niepoprawne parametry pojemności.")
    segments = int(host_length // segment_size)
    payload_capacity = max(0, segments - header_segments) * bits_per_segment
    bps = (bits_per_segment * sample_rate / segment_size) if sample_rate is not None else 0.0
    return segments, payload_capacity, float(bps)


def repeat_bits(bits: np.ndarray, repetition: int) -> np.ndarray:
    if repetition == 1:
        return np.asarray(bits, dtype=np.uint8).copy()
    if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 1:
        raise ValueError("repetition musi być dodatnią liczbą całkowitą.")
    return np.repeat(np.asarray(bits, dtype=np.uint8).reshape(-1), repetition)


def majority_vote(bits: np.ndarray, repetition: int) -> np.ndarray:
    if repetition == 1:
        return np.asarray(bits, dtype=np.uint8).copy()
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if values.size % repetition:
        raise ValueError("Liczba bitów nie jest podzielna przez współczynnik powtórzeń.")
    blocks = values.reshape(-1, repetition)
    return (blocks.sum(axis=1) * 2 >= repetition).astype(np.uint8)


def median_filter(x: np.ndarray, kernel: int = 3) -> np.ndarray:
    data = np.asarray(x, dtype=np.float64).reshape(-1)
    if kernel < 1:
        raise ValueError("kernel musi być dodatni.")
    if kernel % 2 == 0:
        kernel += 1
    return medfilt(data, kernel_size=kernel)


def median_passes(x: np.ndarray, kernel: int = 3, passes: int = 2) -> np.ndarray:
    data = np.asarray(x, dtype=np.float64).reshape(-1)
    if passes < 0:
        raise ValueError("passes nie może być ujemne.")
    result = data.copy()
    for _ in range(passes):
        result = median_filter(result, kernel)
    return result


def wiener_filter(x: np.ndarray, window: int = 1000) -> np.ndarray:
    data = np.asarray(x, dtype=np.float64).reshape(-1)
    if window <= 1:
        raise ValueError("window musi być większe od 1.")
    return wiener(data, mysize=window)


class MultiBinPhaseCoding:
    @staticmethod
    def embed(
        cover_audio_path: str | Path,
        secret_audio_path: str | Path,
        output_path: str | Path,
        password: str | None = None,
        segment_size: int = 2048,
        bins_per_segment: int = 4,
        bits_per_bin: int = 2,
        start_bin: int = 1,
        bin_stride: int = 1,
        secret_rate: int = 8000,
        secret_bits: int = 16,
        repetition: int = 1,
    ) -> dict:
        host_rate, host = read_wav(cover_audio_path)
        secret_source_rate, secret = read_wav(secret_audio_path)

        secret_mono, prepared_rate = prepare_secret(
            secret,
            secret_source_rate,
            target_rate=secret_rate,
            target_bits=secret_bits,
        )
        if secret_mono.size == 0:
            raise ValueError("Plik tajnego audio jest pusty.")

        payload_bits = samples_to_bits(secret_mono, secret_bits)
        payload_bits = repeat_bits(payload_bits, repetition)

        header = build_header(
            prepared_rate,
            1,
            int(secret_mono.size),
            int(secret_bits),
            int(segment_size),
            int(bins_per_segment),
            int(bits_per_bin),
            int(start_bin),
            int(bin_stride),
        )
        protected_header = protect_header(header, password)
        header_bits = bytes_to_bits(protected_header)

        bootstrap = bootstrap_config()
        if host.ndim == 2:
            host_channel = host[:, 0].copy()
            stego = host.copy()
        else:
            host_channel = host.copy()
            stego = host.copy()

        header_segment_count_value = header_segment_count(bootstrap)
        header_locations = find_header_segments(
            host_channel, bootstrap, header_segment_count_value
        )
        if len(header_locations) != header_segment_count_value:
            raise ValueError(
                "Nie znaleziono wystarczającej liczby aktywnych segmentów do "
                "zapisania nagłówka. Użyj innego pliku cover."
            )
        header_offset = (header_locations[-1] + 1) * bootstrap["segment_size"]
        bits_per_segment = int(bins_per_segment) * int(bits_per_bin)
        payload_host_length = max(0, host_channel.size - header_offset)
        payload_segments_count = payload_host_length // int(segment_size)
        payload_capacity = payload_segments_count * bits_per_segment
        capacity_bps = bits_per_segment * host_rate / int(segment_size)
        if payload_bits.size > payload_capacity:
            shortfall = payload_bits.size - payload_capacity
            raise ValueError(
                "Plik audio cover jest zbyt krótki. "
                f"Brakuje {shortfall} bitów; dostępne {payload_capacity} bitów payloadu "
                f"(pojemność: {capacity_bps:.1f} bitów/s)."
            )

        header_area = host_channel[:header_offset].copy()
        for ordinal, index in enumerate(header_locations):
            start_sample = index * bootstrap["segment_size"]
            segment = header_area[
                start_sample:start_sample + bootstrap["segment_size"]
            ]
            start = ordinal * bootstrap["bins_per_segment"]
            chunk = np.zeros(bootstrap["bins_per_segment"], dtype=np.uint8)
            available = header_bits[start:start + bootstrap["bins_per_segment"]]
            chunk[:available.size] = available
            header_area[start_sample:start_sample + bootstrap["segment_size"]] = (
                embed_segment(segment, chunk, bootstrap)
            )

        config = {
            "segment_size": int(segment_size),
            "bins_per_segment": int(bins_per_segment),
            "bits_per_bin": int(bits_per_bin),
            "start_bin": int(start_bin),
            "bin_stride": int(bin_stride),
            "min_magnitude": 0.0,
            "password": password,
        }
        payload_index = 0
        payload_segments, payload_tail = segment_signal(
            host_channel[header_offset:], int(segment_size)
        )
        output_payload_segments: list[np.ndarray] = []
        for segment in payload_segments:
            start = payload_index
            chunk = payload_bits[start:start + bits_per_segment]
            if chunk.size == 0:
                output_payload_segments.append(segment.astype(np.int16))
            else:
                padded = np.zeros(bits_per_segment, dtype=np.uint8)
                padded[:chunk.size] = chunk
                output_payload_segments.append(embed_segment(segment, padded, config))
                payload_index += chunk.size

        stego_channel_parts = [
            header_area,
            np.concatenate(output_payload_segments) if output_payload_segments else np.asarray([], dtype=np.int16),
            payload_tail,
        ]
        stego_channel = np.concatenate([part for part in stego_channel_parts if part.size])

        if host.ndim == 2:
            stego[:, 0] = stego_channel
        else:
            stego = stego_channel

        write_wav(output_path, host_rate, stego)
        return {
            "output": str(output_path),
            "sample_rate": host_rate,
            "secret_rate": prepared_rate,
            "secret_bits": secret_bits,
            "capacity_bits": int(payload_capacity),
            "capacity_bits_per_second": float(capacity_bps),
            "payload_bits": int(payload_bits.size),
            "host_snr_db": snr_db(host[:, 0] if host.ndim == 2 else host, stego[:, 0] if stego.ndim == 2 else stego),
        }

    @staticmethod
    def extract(
        stego_audio_path: str | Path,
        output_path: str | Path,
        password: str | None = None,
        repetition: int = 1,
    ) -> dict:
        _, stego = read_wav(stego_audio_path)
        channel = stego[:, 0] if stego.ndim == 2 else stego
        bootstrap = bootstrap_config()
        segments, _ = segment_signal(channel, bootstrap["segment_size"])
        header_segment_count_value = header_segment_count(bootstrap)
        metadata = None
        header_offset = 0
        legacy_error: ValueError | None = None
        if len(segments) >= header_segment_count_value:
            legacy_config = dict(bootstrap, min_magnitude=0.0)
            legacy_bits = np.concatenate([
                extract_segment(segment, legacy_config)
                for segment in segments[:header_segment_count_value]
            ])[:HEADER_SIZE * 8]
            try:
                metadata = parse_header(
                    unprotect_header(bits_to_bytes(legacy_bits), password)
                )
                header_offset = header_segment_count_value * bootstrap["segment_size"]
            except ValueError as exc:
                legacy_error = exc

        if metadata is None:
            header_locations = find_header_segments(
                channel, bootstrap, header_segment_count_value
            )
            if len(header_locations) != header_segment_count_value:
                if legacy_error is not None:
                    raise ValueError(
                        f"{legacy_error} Nagłówek z aktywnych segmentów nie został znaleziony."
                    ) from legacy_error
                raise ValueError("Nie znaleziono aktywnych segmentów z nagłówkiem.")
            active_bits = np.concatenate([
                extract_segment(segments[index], bootstrap)
                for index in header_locations
            ])[:HEADER_SIZE * 8]
            try:
                metadata = parse_header(
                    unprotect_header(bits_to_bytes(active_bits), password)
                )
            except ValueError as exc:
                if legacy_error is not None:
                    raise ValueError(
                        f"{legacy_error} Nagłówek z aktywnych segmentów jest niepoprawny."
                    ) from exc
                raise
            header_offset = (
                header_locations[-1] + 1
            ) * bootstrap["segment_size"]

        config = {
            "segment_size": metadata["segment_size"],
            "bins_per_segment": metadata["bins_per_segment"],
            "bits_per_bin": metadata["bits_per_bin"],
            "start_bin": metadata["start_bin"],
            "bin_stride": metadata["bin_stride"],
            "min_magnitude": 0.0,
            "password": password,
        }
        payload_bit_count = metadata["secret_length"] * metadata["secret_bit_depth"]
        repeated_count = payload_bit_count * repetition
        payload_per_segment = metadata["bins_per_segment"] * metadata["bits_per_bin"]
        needed_segments = int(np.ceil(repeated_count / payload_per_segment))

        payload_segments, _ = segment_signal(
            channel[header_offset:], metadata["segment_size"]
        )
        if len(payload_segments) < needed_segments:
            raise ValueError("Nagranie nie zawiera pełnych osadzonych danych.")

        extracted = np.concatenate([
            extract_segment(segment, config)
            for segment in payload_segments[:needed_segments]
        ])[:repeated_count]
        if repetition != 1:
            extracted = majority_vote(extracted, repetition)
        extracted = extracted[:payload_bit_count]

        secret_samples = bits_to_samples(extracted, metadata["secret_bit_depth"])
        secret_samples = secret_samples[:metadata["secret_length"]]
        write_wav(output_path, metadata["secret_rate"], secret_samples)
        return {
            "output": str(output_path),
            "secret_rate": metadata["secret_rate"],
            "secret_length": metadata["secret_length"],
            "secret_bit_depth": metadata["secret_bit_depth"],
            "repetition": repetition,
        }

    @staticmethod
    def calculate_snr(cover_audio_path: str | Path, stego_audio_path: str | Path) -> float:
        _, cover = read_wav(cover_audio_path)
        _, stego = read_wav(stego_audio_path)
        cover_channel = cover[:, 0] if cover.ndim == 2 else cover
        stego_channel = stego[:, 0] if stego.ndim == 2 else stego
        return snr_db(cover_channel, stego_channel)


PhaseCodingStego = MultiBinPhaseCoding
PhaseCoding = MultiBinPhaseCoding


def embed(
    cover_audio_path: str | Path,
    secret_audio_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> dict:
    return MultiBinPhaseCoding.embed(
        cover_audio_path,
        secret_audio_path,
        output_path,
        **kwargs,
    )


def extract(
    stego_audio_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> dict:
    return MultiBinPhaseCoding.extract(
        stego_audio_path,
        output_path,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed or extract audio with multi-bin phase coding."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    encode_parser = commands.add_parser("encode", help="Embed a secret WAV in a cover WAV.")
    encode_parser.add_argument("--cover", required=True, help="16-bit PCM cover WAV.")
    encode_parser.add_argument("--secret", required=True, help="16-bit PCM secret WAV.")
    encode_parser.add_argument("--out", required=True, help="Output stego WAV.")
    encode_parser.add_argument("--password", help="Optional header obfuscation password.")
    encode_parser.add_argument("--segment-size", type=int, default=2048)
    encode_parser.add_argument("--bins", type=int, default=4, dest="bins_per_segment")
    encode_parser.add_argument("--bits-per-bin", type=int, default=2)
    encode_parser.add_argument("--start-bin", type=int, default=1)
    encode_parser.add_argument("--bin-stride", type=int, default=1)
    encode_parser.add_argument("--secret-rate", type=int, default=8000)
    encode_parser.add_argument("--secret-bits", type=int, choices=(8, 16), default=16)
    encode_parser.add_argument("--repetition", type=int, default=1)

    extract_parser = commands.add_parser("extract", help="Extract a secret WAV from a stego WAV.")
    extract_parser.add_argument("--stego", required=True, help="Input stego WAV.")
    extract_parser.add_argument("--out", required=True, help="Output recovered secret WAV.")
    extract_parser.add_argument("--password", help="Password used while encoding.")
    extract_parser.add_argument("--repetition", type=int, default=1)

    args = parser.parse_args(argv)
    try:
        if args.command == "encode":
            result = MultiBinPhaseCoding.embed(
                args.cover,
                args.secret,
                args.out,
                password=args.password,
                segment_size=args.segment_size,
                bins_per_segment=args.bins_per_segment,
                bits_per_bin=args.bits_per_bin,
                start_bin=args.start_bin,
                bin_stride=args.bin_stride,
                secret_rate=args.secret_rate,
                secret_bits=args.secret_bits,
                repetition=args.repetition,
            )
        else:
            result = MultiBinPhaseCoding.extract(
                args.stego,
                args.out,
                password=args.password,
                repetition=args.repetition,
            )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
