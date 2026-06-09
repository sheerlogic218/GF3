# JOSS-F diagnostic modem

The transmitter and receiver algorithms are unchanged. The added code only records intermediate values, prints stage summaries, and writes diagnostic artefacts.

Place `spliced_seed_qpsk.npy` beside `modem.py`.

## Receive with diagnostics

```bash
python receiver.py other_group.wav
```

A timestamped directory is created under `diagnostics/`. It contains:

- `ASSESSMENT.txt`: first likely failure boundary.
- `diagnostic_report.json`: all structured values.
- `diagnostic_arrays.npz`: numerical arrays for deeper inspection.
- CSV tables for chirps, Golay pulses, OFDM blocks, pilot scans, constellation quality and LDPC results.
- PNG plots of the recording, chirp metric, channel, CP timing, pilot positions and constellation.
- Raw decoded-byte previews after each LDPC group.

Use `--shallow` to skip the diagnostic QPSK/interleaver hypothesis scan, or `--no-plots` when matplotlib is unavailable.

## Transmit with fingerprints

```bash
python transmitter.py payload.txt --wav tx.wav
```

This creates `diagnostics/tx_tx/REFERENCE_FINGERPRINTS.txt` and a JSON report. Other groups can compare hashes for the chirp, Golay sequences, known pilot, interleaver and LDPC test vector.

## Compare two JSON reports

```bash
python compare_reports.py report_a.json report_b.json
```

A mismatch in a component fingerprint identifies an exact implementation disagreement before acoustic effects are considered.
