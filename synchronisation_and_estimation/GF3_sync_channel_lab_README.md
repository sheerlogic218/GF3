# GF3 synchronisation + channel-estimation lab

This is a measurement harness, not a complete modem. It is designed to produce evidence quickly:

- synchronisation peaks from matched filtering of a chirp or stepped-frequency template;
- a repeated-half self-similarity metric, similar in spirit to Schmidl-Cox;
- impulse-response estimates from a known band-limited white-noise probe;
- channel frequency-response magnitude and phase plots;
- a simulated OFDM/DMT one-tap equalisation constellation plot.

## Install

```bash
python -m pip install numpy scipy matplotlib
```

Optional live play/record:

```bash
python -m pip install sounddevice
```

## First run: simulated data

```bash
python gf3_sync_channel_lab.py demo --out results_demo
```

This creates simulated TX/RX WAVs and figures:

- `00_recording_spectrogram.png`
- `01_matched_filter_sync.png`
- `02_repeated_half_metric.png`
- `03_channel_impulse_response.png`
- `04_channel_frequency_response_magnitude.png`
- `05_channel_frequency_response_phase.png`
- `06_ofdm_constellation_equalisation.png`
- `results.json`

These are not real measurements. They are a sanity check that the pipeline works.

## Real data workflow

Create a probe waveform:

```bash
python gf3_sync_channel_lab.py make-probes --out probes --sync-kind chirp
```

This writes:

- `probes/measurement_tx.wav` — play this through the laptop speaker;
- `probes/sync_template.wav` — known synchronisation template;
- `probes/repeated_template.wav` — repeated-half template;
- `probes/training_template.wav` — known channel-estimation waveform;
- `probes/metadata.json` — timing offsets.

Record the played signal using Audacity, QuickTime, Python, or another laptop. Save it as `recordings/rx.wav`.

Analyse it:

```bash
python gf3_sync_channel_lab.py analyse --probe-dir probes --rx recordings/rx.wav --out results_real
```

Optional direct play-record helper:

```bash
python gf3_sync_channel_lab.py live --tx probes/measurement_tx.wav --rx-out recordings/rx.wav
python gf3_sync_channel_lab.py analyse --probe-dir probes --rx recordings/rx.wav --out results_real
```

On macOS, Terminal/VS Code must have microphone permission.

## What to report from the plots

Use these as concrete claims:

1. Matched-filter synchronisation gives a sharp timing peak. Quote the peak-to-sidelobe ratio from `results.json`.
2. Repeated-half synchronisation is a separate self-similarity method. It should peak near the repeated random segment.
3. The impulse response shows direct path plus echoes; at 48 kHz, 1500 samples is 31.25 ms.
4. The channel magnitude response is not flat. This justifies pilot/channel estimation and one-tap equalisation per OFDM subcarrier.
5. If the real recording clips, rerun at lower speaker volume and microphone gain.

## Notes on interpretation

- A chirp generally gives a strong matched-filter peak over a broad frequency band.
- A stepped template is more like a deterministic fingerprint; it can still work if some frequencies are attenuated, but it may fail when particular tones are badly suppressed.
- White-noise channel estimation is useful because it excites many frequencies at once.
- Long impulse-response estimates are not automatically good. Compare lengths such as 512, 1024, 2048, and 2400 samples and look for whether the extra taps contain meaningful echo energy or mostly noise.
