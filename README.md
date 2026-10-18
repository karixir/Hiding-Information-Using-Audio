# Hiding-Information-Using-Audio

## Proponowana architektura projektu

```text
AudioStego/
├── assets/                  # Ikony, przykładowe próbki audio (.wav)
├── docs/                    # Prezentacja (.pptx) oraz dokumentacja PDF
├── src/
│   ├── algorithms/          # Implementacje 6 algorytmów
│   │   ├── __init__.py
│   │   ├── alg1_lsb.py
│   │   ├── alg2_dwt.py
│   │   ├── alg3_stft.py
│   │   ├── alg4_phase.py
│   │   ├── alg5_stegibiza.py
│   │   └── alg6_hybrid.py
│   ├── core/                # Moduły pomocnicze
│   │   ├── audio_handler.py # Obsługa plików WAV i konwersji bitowych
│   │   └── metrics.py       # Funkcje liczące SNR, PSNR, MSE
│   ├── gui/                 # Warstwa interfejsu graficznego
│   │   ├── app.py           # Główne okno aplikacji
│   │   ├── tabs.py          # Logika poszczególnych zakładek
│   │   └── plotter.py       # Osadzanie wykresów matplotlib w CustomTkinter
│   └── main.py              # Punkt wejściowy aplikacji
├── requirements.txt         # Zależności projektu
└── README.md
