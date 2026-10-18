import numpy as np
from scipy.io import wavfile


class LSBAudioStego:

    @staticmethod
    def _text_to_bits(text: str) -> list[int]:
        # Konwertuje tekst na listę bitów (8 bitów na znak ASCII / UTF-8).
        byte_data = text.encode("utf-8")
        bits = []
        for byte in byte_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_text(bits: list[int]) -> str:
        # Konwertuje listę bitów z powrotem na tekst UTF-8.
        chars = bytearray()
        for i in range(0, len(bits), 8):
            byte_chunk = bits[i : i + 8]
            if len(byte_chunk) < 8:
                break
            byte_val = 0
            for bit in byte_chunk:
                byte_val = (byte_val << 1) | bit
            chars.append(byte_val)
        return chars.decode("utf-8", errors="replace")

    @classmethod
    def embed(cls, cover_audio_path: str, secret_text: str, output_path: str):
        """
        1. Wczytanie audio cover (k próbek).
        2. Konwersja tekstu na bity (m bitów).
        3. Sprawdzenie warunku pojemności: k > m.
        4. Zastąpienie LSB próbek audio bitami wiadomości.
        """
        # 1. Wczytanie pliku WAV

        # Sample rate - a'la rozdzielczość muzyki, komputer nie potrafi 
        # odtworzyć fali ciągłej, więc dzieli ją tysiące razy na sekunde
        # Data - typ danych
        
        sample_rate, data = wavfile.read(cover_audio_path)


        # Upewniamy się, że próbki to liczby całkowite (np. int16)
        if data.dtype != np.int16:
            raise ValueError(
                f"Oczekiwano formatu PCM 16-bit (int16), otrzymano: {data.dtype}"
            )

        # Kopia danych, aby nie modyfikować oryginału w pamięci
        stego_data = data.copy()

        # Obsługa stereo (spłaszczenie do 1D na czas modyfikacji lub jeśli jest mono to idziemy dalej)
        #Sprawdzenie czy ma dwa wymiary, bo stereo L - lewy i P - prawy
        is_stereo = len(stego_data.shape) == 2
        flat_samples = (
            stego_data.reshape(-1) if is_stereo else stego_data
        )  # spłaszczony widok
        k = len(flat_samples)

        # 2. Przygotowanie bitów wiadomości
        message_bits = cls._text_to_bits(secret_text)

        # Dodajemy 32-bitowy nagłówek z długością wiadomości (w bitach),
        # aby algorytm ekstrakcji wiedział dokładnie, kiedy zakończyć odczyt.
        # czyli liczymi ile bitów ma wiadomość tą liczbę int zamieniamy na zapis binarny
        msg_len = len(message_bits)
        header_bits = [(msg_len >> i) & 1 for i in range(31, -1, -1)]
        all_bits_to_hide = header_bits + message_bits
        m = len(all_bits_to_hide)

        # 3. Sprawdzenie warunku: k > m
        if k < m:
            raise ValueError(
                f"Plik audio jest zbyt krótki! Dostępne próbki (k): {k}, "
                f"Wymagane bity (m): {m}."
            )

        # 4. Nadpisywanie LSB (Least Significant Bit)
        for i in range(m):
            # Zerujemy najmłodszy bit (& ~1) i wstawiamy nasz bit (| all_bits_to_hide[i])
            flat_samples[i] = (flat_samples[i] & ~1) | all_bits_to_hide[i]

        # 5. Zapis zmodyfikowanego pliku audio
        wavfile.write(output_path, sample_rate, stego_data)
        print(f"Wiadomość ukryta pomyślnie w: {output_path}")
        print(f"Ukryto bitów: {m} (w tym 32 bity nagłówka)")

    @classmethod
    def extract(cls, stego_audio_path: str) -> str:
        """
        1. Wczytanie pliku stego audio.
        2. Odczyt LSB z kolejnych próbek.
        3. Rekonstrukcja bitów na znaki tekstu.
        """
        _, data = wavfile.read(stego_audio_path)
        flat_samples = data.reshape(-1) if len(data.shape) == 2 else data

        # 1. Odczyt 32-bitowego nagłówka informującego o długości wiadomości
        msg_len = 0
        for i in range(32):
            bit = int(flat_samples[i] & 1)
            msg_len = (msg_len << 1) | bit

        # Sprawdzenie poprawności nagłówka
        if msg_len <= 0 or (32 + msg_len) > len(flat_samples):
            raise ValueError(
                "Nie wykryto poprawnej wiadomości lub plik jest uszkodzony."
            )

        # 2. Odczyt właściwych bitów wiadomości
        secret_bits = []
        for i in range(32, 32 + msg_len):
            secret_bits.append(int(flat_samples[i] & 1))

        # 3. Konwersja bitów na tekst
        return cls._bits_to_text(secret_bits)

    @staticmethod
    def calculate_snr(cover_audio_path: str, stego_audio_path: str) -> float:
        # Obliczenie SNR 

        _, orig = wavfile.read(cover_audio_path)
        _, stego = wavfile.read(stego_audio_path)

        orig = orig.astype(np.float64)
        stego = stego.astype(np.float64)

        # Energia sygnału oryginalnego
        signal_power = np.sum(orig**2)
        # Energia szumu (różnicy wprowadzonej przez LSB)
        noise_power = np.sum((orig - stego) ** 2)

        if noise_power == 0:
            return float("inf")  # Brak jakichkolwiek zniekształceń

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

# TO DO: DELETE
# Przykład użycia / Walidacja działania

if __name__ == "__main__":
    # 1. Tworzymy krótki syntetyczny sygnał audio (1 sekunda, 16 kHz - jak w artykule)
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    # Dźwięk sinusoidalny 440 Hz (ton A)
    audio_signal = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)

    cover_filename = "cover_audio.wav"
    stego_filename = "stego_audio.wav"

    wavfile.write(cover_filename, fs, audio_signal)
    print(f"[*] Wygenerowano plik testowy: {cover_filename} ({fs} próbek)")

    # 2. Tekst do ukrycia
    secret_message = (
        "Projekt Steganografii: Tajna wiadomosc ukryta w bitach LSB audio!"
    )
    print(f"[*] Tekst do ukrycia: '{secret_message}'")

    # 3. Ukrywanie (Embedding)
    LSBAudioStego.embed(cover_filename, secret_message, stego_filename)

    # 4. Odzyskiwanie (Extraction)
    recovered_text = LSBAudioStego.extract(stego_filename)
    print(f"[+] Odzyskana wiadomosc: '{recovered_text}'")

    # 5. Obliczenie współczynnika SNR (zgodnie z sekcją IV pracy)
    snr_val = LSBAudioStego.calculate_snr(cover_filename, stego_filename)
    print(f"[📊] Obliczone SNR: {snr_val:.2f} dB")