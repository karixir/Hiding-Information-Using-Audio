# Hiding-Information-Using-Audio

## Architektura projektu

```text
Hiding-Information-Using-Audio/
├── README.md
├── requirements.txt
└── AudioStego/
    ├── docs/
    │   └── Research_papers.md
    ├── src/
    │   ├── algorithms/
    │   │   ├── __init__.py
    │   │   ├── alg1_lsb.py
    │   │   ├── alg2_chmap.py
    │   │   ├── alg2_dwt.py
    │   │   ├── alg3_stft.py
    │   │   ├── alg4_phase.py
    │   │   ├── alg5_stegibiza.py
    │   │   ├── alg6_hybrid.py
    │   │   └── chaotic_maps.py
    │   ├── core/
    │   │   ├── audio_utils.py
    │   │   ├── image_utils.py
    │   │   └── metrics.py
    │   ├── gui/
    │   │   ├── app.py
    │   │   ├── tabs.py
    │   │   └── plotter.py
    │   └── main.py
    ├── assets/
    │   └── przykładowe pliki audio i obrazów
    └── results/
        └── katalog wynikowy dla wygenerowanych plików
```

Uwaga: na ten etap rozwoju projektu moduły algorytmów są używane osobno. 

## Jak używać alg1_lsb.py


## Jak używać alg2_chmap

### 1. Szyfrowanie obrazu do pliku audio

```bash
python AudioStego/src/algorithms/alg2_chmap.py encrypt \
  -s assets/secret.png \
  -c assets/cover.wav \
  -o assets/cover_encrypted.wav \
  -m hybrid \
  -p 1 \
  -q 1 \
  -i 8 \
  -k "8,8,8,8,8,8,8,8" \
  -a 0.08
```

Jeśli nie podasz `-o` / `--output-path`, program zapisze wynik obok oryginalnego pliku cover z nazwą w formacie:

```text
cover_encrypted.wav
```

dla pliku `cover.wav`.

### 2. Odszyfrowanie obrazu z audio stego

```bash
python AudioStego/src/algorithms/alg2_chmap.py decrypt \
  -s assets/cover_encrypted.wav \
  -o results/recovered.png \
  -m hybrid \
  -p 1 \
  -q 1 \
  -i 8 \
  -k "8,8,8,8,8,8,8,8"
```

### Parametry wspierane przez program

#### Parametry dla trybu `encrypt`

- `-s`, `--secret-image` — ścieżka do obrazu tajnego (PNG/JPG),
- `-c`, `--cover-audio` — ścieżka do pliku cover WAV,
- `-o`, `--output-path` — opcjonalna pełna ścieżka do pliku stego; jeśli pominięte, używane jest `cover_encrypted.wav` w tym samym katalogu co plik cover,
- `-m`, `--map-type` — typ mapy: `arnold`, `baker`, `hybrid`,
- `-p`, `--p` — parametr `p` dla mapy Arnolda,
- `-q`, `--q` — parametr `q` dla mapy Arnolda,
- `-i`, `--iterations` — liczba iteracji mapy chaotycznej,
- `-k`, `--key-vector` — wektor klucza w formacie `"x,x,x,..."` dla Bakera lub hybrydy,
- `-a`, `--alpha` — współczynnik osadzania w sygnale audio.
- `-h`, `--help` — wyświetla pomoc dla trybu `encrypt`.

#### Parametry dla trybu `decrypt`

- `-s`, `--stego-audio` — ścieżka do pliku stego WAV,
- `-o`, `--output-path` — ścieżka do zapisania odzyskanego obrazu,
- `-m`, `--map-type` — typ mapy użytej podczas szyfrowania,
- `-p`, `--p`, `-q`, `--q`, `-i`, `--iterations`, `-k`, `--key-vector` — parametry synchronizujące proces odwracania mapy.
- `-h`, `--help` — wyświetla pomoc dla trybu `decrypt`.

### Obsługiwane typy map

- `arnold` — transformacja Arnolda,
- `baker` — mapa Baker,
- `hybrid` — łączenie Arnold + Baker.

## Obsługiwane rozszerzenia plików

### Obrazy wejściowe

- `*.png`, `*.jpg`, `*.jpeg`, `*.bmp`, `*.tif`, `*.tiff` — obrazy tajne wczytywane przez Pillow, rozszerzane i konwertowane do skali szarości.

### Audio wejściowe i wyjściowe

- `*.wav` — jedyny wspierany format pliku cover/stego w tej wersji projektu. Kod wczytuje i zapisuje pliki audio przez `soundfile` w formacie WAV 16-bit PCM.
- `*.wav` jest również używany jako format wyjściowy dla pliku stego generowanego przez `alg2_chmap.py`.

### Obraz wyjściowy odzyskany

- Zalecany format wyjściowy to `*.png`, ponieważ odzyskany obraz jest zapisywany w skali szarości.
- W praktyce Pillow obsługuje wyżej wymienione formaty, ale `PNG` jest rekomendowanym wyborem dla tej implementacji.

## Dodatkowe informacje

`main.py` jest obecnie plikiem pustym / placeholderem. W przyszłości będzie mogło pojawić się osobne wejście CLI, ale na ten etap projekt powinien być uruchamiany bezpośrednio z modułów algorytmów.

## Wymagania

```bash
pip install -r requirements.txt
```

