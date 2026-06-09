# plugtest_modem_2 — corrected JOSS-F modem

This build follows the supplied JOSS-F vF document and the official Group 1
`golay_pilot.py` waveform.

## Files

- `modem.py` — JOSS-F parameters, pilots, OFDM, QPSK, framing, LDPC and audio I/O
- `transmitter.py` — reads `payload.txt` and creates a named WAV
- `receiver.py` — decodes a named WAV
- `payload.txt` — file transmitted by default
- `requirements.txt` — Python dependencies

## Frame layout

Relative to the start of the first chirp:

```text
10 contiguous chirps                                  40,960 samples
Golay prefix gap                                       2,048 samples
4 × [A, gap, B, gap] = 8 Golay pulses                49,152 samples
First OFDM block begins at                            92,160 samples
```

Each Golay A/B pulse is 4096 samples and each gap is 2048 samples. The complete
Golay section is exactly 51,200 samples, matching Group 1's reference generator.

## Use

```bash
python3 transmitter.py
```

Enter `tx` to create `tx.wav` beside the scripts.

```bash
python3 receiver.py
```

Enter `recorded` to decode `recorded.wav` beside the scripts. Names with or
without `.wav` are accepted.
