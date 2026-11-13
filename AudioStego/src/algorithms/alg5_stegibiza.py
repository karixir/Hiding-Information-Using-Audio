from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample, stft

# Poprawka ścieżek, żeby Python zawsze widział folder 'core'
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.audio_utils import load_audio, save_audio


class StegIbiza:
    MORSE_TABLE = {
        "a": "+-", "b": "-+++", "c": "-+-+", "d": "-++", "e": "+",
        "f": "++-+", "g": "--+", "h": "++++", "i": "++", "j": "+---",
        "k": "-+-", "l": "+-++", "m": "--", "n": "-+", "o": "---",
        "p": "+--+", "q": "--+-", "r": "+-+", "s": "+++", "t": "-",
        "u": "++-", "v": "+++-", "w": "+--", "x": "-++-", "y": "-+--",
        "z": "--++", "1": "+----", "2": "++---", "3": "+++--", "4": "++++-",
        "5": "+++++", "6": "-++++", "7": "--+++", "8": "---++", "9": "----+",
        "0": "-----", " ": " ", "!": "-+-+--", "?": "++--++", ".": "+-+-+-",
        ",": "--++--",
    }

    REVERSE_MORSE = {v: k for k, v in MORSE_TABLE.items() if k != " "}

    @classmethod
    def text_to_stegibiza_stream(cls, text: str) -> str:
        words = text.lower().split()
        if not words:
            raise ValueError("Wiadomość do ukrycia nie może być pusta.")

        unsupported = sorted(
            {character for character in "".join(words) if character not in cls.MORSE_TABLE}
        )
        if unsupported:
            raise ValueError(
                "Wiadomość zawiera znaki nieobsługiwane przez tabelę Morse'a: "
                + ", ".join(repr(character) for character in unsupported)
            )

        encoded_words = [
            "0".join(cls.MORSE_TABLE[character] for character in word)
            for word in words
        ]
        return "00".join(encoded_words)

    @classmethod
    def stegibiza_stream_to_text(cls, stream: str) -> str:
        words = stream.split("00")
        decoded_words = []

        for word in words:
            if not word:
                raise ValueError("Niepoprawny strumień StegIbiza: puste słowo.")
            chars = word.split("0")
            decoded_chars = []
            for c in chars:
                if c not in cls.REVERSE_MORSE:
                    raise ValueError(f"Nieznany kod Morse'a w strumieniu: {c!r}")
                decoded_chars.append(cls.REVERSE_MORSE[c])
            decoded_words.append("".join(decoded_chars))

        return " ".join(decoded_words)

    @staticmethod
    def _stretch_beat(audio_segment: np.ndarray, ratio: float) -> np.ndarray:
        new_len = max(1, int(round(len(audio_segment) * ratio)))
        return resample(audio_segment, new_len).astype(np.float64, copy=False)

    @staticmethod
    def _validate_timing(bpm: float, delta_bpm: float, phi: int) -> None:
        if not math.isfinite(bpm) or bpm <= 0:
            raise ValueError("bpm musi być dodatnią, skończoną liczbą.")
        if not math.isfinite(delta_bpm) or not 0 < delta_bpm < bpm:
            raise ValueError("delta_bpm musi być dodatnie i mniejsze od bpm.")
        if isinstance(phi, bool) or not isinstance(phi, int) or phi <= 0:
            raise ValueError("phi musi być dodatnią liczbą całkowitą.")

    @staticmethod
    def _lowpass_bass(audio: np.ndarray, sample_rate: int, cutoff: float = 150.0) -> np.ndarray:
        """NOWOŚĆ: Filtr dolnoprzepustowy. Zostawia tylko głęboki bas (stopę perkusji),
        wycinając wokal, gitary i talerze, które myliły detektor."""
        if audio.size < 15:
            return audio
        nyquist = 0.5 * sample_rate
        normal_cutoff = min(cutoff / nyquist, 0.99)
        b, a = butter(4, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, audio)

    @classmethod
    def _onset_times(cls, audio: np.ndarray, sample_rate: int, bpm: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
        """Ulepszony wykrywacz rytmu: filtruje bas i szuka uderzeń w minimalnych odstępach."""
        if audio.size < 2:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

        # 1. Izolujemy sam bas, żeby wokal nie udawał perkusji
        bass_audio = cls._lowpass_bass(audio, sample_rate)

        frame_length = min(1024, bass_audio.size)
        hop_length = max(1, min(32, frame_length // 4))
        _, times, spectrum = stft(
            bass_audio,
            fs=sample_rate,
            nperseg=frame_length,
            noverlap=frame_length - hop_length,
            boundary="zeros",
        )
        magnitude = np.abs(spectrum)
        flux = np.maximum(np.diff(magnitude, axis=1), 0.0).sum(axis=0)
        flux = np.pad(flux, (1, 0))
        if not np.any(flux > 0):
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

        # 2. Szukamy tylko wyraźnych szczytów (distance pilnuje, by nie łapać podwójnych uderzeń w jednym bicie)
        min_beat_frames = max(1, int((0.6 * (60.0 / bpm) * sample_rate) / hop_length))
        prominence = np.percentile(flux[flux > 0], 50) * 0.5
        peaks, _ = find_peaks(flux, distance=min_beat_frames, prominence=prominence)

        if peaks.size == 0:
            peaks, _ = find_peaks(flux)

        return times[peaks], flux[peaks]

    @classmethod
    def embed(
        cls,
        cover_path: str,
        message: str,
        output_path: str,
        bpm: float = 120.0,
        delta_bpm: float = 1.0,
        phi: int = 1,
    ) -> dict[str, int | str]:
        cls._validate_timing(bpm, delta_bpm, phi)
        data, fs = load_audio(cover_path)
        if data.size == 0:
            raise ValueError("Plik audio cover jest pusty.")
        symbols = cls.text_to_stegibiza_stream(message)

        # NOWOŚĆ: Szukamy pierwszego prawdziwego uderzenia bębna (synchronizacja początku!)
        onset_times, _ = cls._onset_times(data, fs, bpm)
        if onset_times.size == 0:
            raise ValueError("Nie wykryto żadnego rytmu w pliku audio.")

        # Zostawiamy nietknięte intro przed pierwszym bębnem
        first_beat_sample = int(round(onset_times[0] * fs))
        stego_chunks: list[np.ndarray] = []
        if first_beat_sample > 0:
            stego_chunks.append(data[:first_beat_sample])

        samples_per_beat = int(round((60.0 / bpm) * fs * phi))
        required_samples = first_beat_sample + (len(symbols) + 1) * samples_per_beat
        if required_samples > data.size:
            raise ValueError(
                f"Plik audio jest zbyt krótki: potrzeba co najmniej {required_samples} próbek, "
                f"dostępne {data.size}."
            )

        ratio_plus = bpm / (bpm + delta_bpm)
        ratio_minus = bpm / (bpm - delta_bpm)

        current_idx = first_beat_sample
        for symbol in symbols:
            chunk = data[current_idx : current_idx + samples_per_beat]
            current_idx += samples_per_beat

            if symbol == "+":
                modified_chunk = cls._stretch_beat(chunk, ratio_plus)
            elif symbol == "-":
                modified_chunk = cls._stretch_beat(chunk, ratio_minus)
            else:
                modified_chunk = chunk

            stego_chunks.append(modified_chunk)

        if current_idx < data.size:
            stego_chunks.append(data[current_idx:])

        stego_audio = np.concatenate(stego_chunks, axis=0)
        save_audio(output_path, stego_audio, fs)
        return {
            "symbols": len(symbols),
            "sample_rate": fs,
            "output": str(output_path),
        }

    @classmethod
    def extract(
        cls,
        stego_path: str,
        num_symbols: int,
        bpm: float = 120.0,
        delta_bpm: float = 1.0,
        phi: int = 1,
        cover_path: str | Path | None = None,
    ) -> str:
        cls._validate_timing(bpm, delta_bpm, phi)
        if isinstance(num_symbols, bool) or not isinstance(num_symbols, int) or num_symbols <= 0:
            raise ValueError("num_symbols musi być dodatnią liczbą całkowitą.")

        if cover_path is None:
            stego_file = Path(stego_path)
            suffix = "_stegibiza"
            if stego_file.stem.endswith(suffix):
                sibling_cover = stego_file.with_name(
                    stego_file.stem[: -len(suffix)] + stego_file.suffix
                )
                if sibling_cover.is_file():
                    cover_path = sibling_cover

        data, fs = load_audio(stego_path)
        if cover_path is not None:
            cover, cover_sample_rate = load_audio(cover_path)
            if cover_sample_rate != fs:
                raise ValueError(
                    "Plik cover i stego muszą mieć tę samą częstotliwość próbkowania."
                )
            return cls._extract_with_cover(
                cover, data, fs, num_symbols, bpm, delta_bpm, phi
            )

        nominal_len = int(round((60.0 / bpm) * fs * phi))
        len_plus = int(round(nominal_len * bpm / (bpm + delta_bpm)))
        len_minus = int(round(nominal_len * bpm / (bpm - delta_bpm)))
        if len_plus == nominal_len or len_minus == nominal_len or len_plus == len_minus:
            raise ValueError(
                "Zmiana BPM jest zbyt mała, aby odróżnić symbole przy tej "
                "częstotliwości próbkowania i wartości phi."
            )

        onset_times, onset_strengths = cls._onset_times(data, fs, bpm)
        if onset_times.size < num_symbols + 1:
            raise ValueError(
                "Za mało transjentów rytmicznych do odczytu wiadomości; "
                f"potrzeba {num_symbols + 1}, wykryto {onset_times.size}."
            )

        expected_durations = np.asarray(
            [len_plus, nominal_len, len_minus], dtype=np.float64
        ) / fs
        detected_symbols: list[str] = []

        # NOWOŚĆ: Detektyw zaczyna mierzyć od PIERWSZEGO BĘBNA (onset_times[0]), a nie od 0.0!
        current_time = float(onset_times[0])
        timing_tolerance = max(
            2.0 * max(abs(len_plus - nominal_len), abs(len_minus - nominal_len)) / fs,
            0.15 * (60.0 / bpm) * phi,
        )

        for symbol_index in range(num_symbols):
            expected_time = current_time + (nominal_len / fs)
            min_time = expected_time - timing_tolerance
            max_time = expected_time + timing_tolerance
            candidates = np.flatnonzero(
                (onset_times >= min_time) & (onset_times <= max_time)
            )
            if candidates.size == 0:
                raise ValueError(
                    "Nie wykryto transjentu dla symbolu "
                    f"{symbol_index + 1}; sprawdź bpm, phi i synchronizację początku. "
                    "Dla utworów z nieregularnym rytmem użyj opcji --cover-audio."
                )

            candidate = candidates[np.argmax(onset_strengths[candidates])]
            duration = onset_times[candidate] - current_time
            symbol = ("+", "0", "-")[int(np.argmin(np.abs(expected_durations - duration)))]
            current_time = onset_times[candidate]
            detected_symbols.append(symbol)

        return cls.stegibiza_stream_to_text("".join(detected_symbols))

    @classmethod
    def _extract_with_cover(
        cls,
        cover: np.ndarray,
        stego: np.ndarray,
        sample_rate: int,
        num_symbols: int,
        bpm: float,
        delta_bpm: float,
        phi: int,
    ) -> str:
        """Recover symbols by matching each stego segment against the known cover."""
        nominal_len = int(round((60.0 / bpm) * sample_rate * phi))
        onset_times, _ = cls._onset_times(cover, sample_rate, bpm)
        if onset_times.size == 0:
            raise ValueError("Nie wykryto rytmu w pliku cover.")

        cover_index = int(round(onset_times[0] * sample_rate))
        required_samples = cover_index + (num_symbols + 1) * nominal_len
        if required_samples > cover.size:
            raise ValueError(
                "Plik cover jest zbyt krótki dla podanej liczby symboli i parametrów."
            )

        ratio_plus = bpm / (bpm + delta_bpm)
        ratio_minus = bpm / (bpm - delta_bpm)
        stego_index = cover_index
        detected_symbols: list[str] = []

        for symbol_index in range(num_symbols):
            cover_chunk = cover[cover_index : cover_index + nominal_len]
            candidates = {
                "+": cls._stretch_beat(cover_chunk, ratio_plus),
                "0": cover_chunk,
                "-": cls._stretch_beat(cover_chunk, ratio_minus),
            }
            scores: dict[str, float] = {}
            for symbol, candidate in candidates.items():
                end = stego_index + candidate.size
                if end > stego.size:
                    continue
                observed = stego[stego_index:end]
                reference_power = float(np.mean(candidate**2))
                error = float(np.mean((observed - candidate) ** 2))
                scores[symbol] = error / max(reference_power, 1e-12)

            if len(scores) != len(candidates):
                raise ValueError(
                    f"Za mało danych stego do odczytu symbolu {symbol_index + 1}."
                )

            ranked = sorted(scores, key=scores.get)
            best, second_best = ranked[:2]
            if scores[best] > 1e-3 or scores[second_best] - scores[best] < 1e-4:
                raise ValueError(
                    "Plik cover nie pasuje do stego albo symbol "
                    f"{symbol_index + 1} jest niejednoznaczny."
                )

            detected_symbols.append(best)
            cover_index += nominal_len
            stego_index += candidates[best].size

        return cls.stegibiza_stream_to_text("".join(detected_symbols))


def _default_output_path(cover_audio_path: str | Path) -> Path:
    cover = Path(cover_audio_path)
    return cover.with_name(f"{cover.stem}_stegibiza.wav")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hide or extract text using the StegIbiza tempo-modulation method."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    embed_parser = subparsers.add_parser("embed", help="Hide text in a beat-synchronized WAV file.")
    embed_parser.add_argument("-c", "--cover-audio", required=True)
    embed_parser.add_argument("-m", "--message", required=True)
    embed_parser.add_argument("-o", "--output-path")
    embed_parser.add_argument("--bpm", type=float, default=120.0)
    embed_parser.add_argument("--delta-bpm", type=float, default=1.0)
    embed_parser.add_argument("--phi", type=int, default=1)

    extract_parser = subparsers.add_parser("extract", help="Extract text from a StegIbiza WAV file.")
    extract_parser.add_argument("-s", "--stego-audio", required=True)
    extract_parser.add_argument(
        "-c",
        "--cover-audio",
        help="Original cover WAV used for embedding; recommended for music with irregular beats.",
    )
    extract_parser.add_argument("-n", "--num-symbols", required=True, type=int)
    extract_parser.add_argument("--bpm", type=float, default=120.0)
    extract_parser.add_argument("--delta-bpm", type=float, default=1.0)
    extract_parser.add_argument("--phi", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "embed":
        output_path = Path(args.output_path) if args.output_path else _default_output_path(args.cover_audio)
        result = StegIbiza.embed(
            args.cover_audio,
            args.message,
            output_path,
            bpm=args.bpm,
            delta_bpm=args.delta_bpm,
            phi=args.phi,
        )
        print(f"Ukryto {result['symbols']} symboli w pliku: {result['output']}")
    else:
        message = StegIbiza.extract(
            args.stego_audio,
            args.num_symbols,
            bpm=args.bpm,
            delta_bpm=args.delta_bpm,
            phi=args.phi,
            cover_path=args.cover_audio,
        )
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())