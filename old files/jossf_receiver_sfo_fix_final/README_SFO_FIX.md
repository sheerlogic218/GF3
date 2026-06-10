# JOSS-F receiver-only SFO and channel-estimation fix

## Files to replace

Replace these files in the existing modem directory:

- `modem.py`
- `receiver.py`
- `test_interop.py`

`transmitter.py` is included as a complete reference copy, but its transmitted signal processing is unchanged.

Keep the cohort's real `spliced_seed_qpsk.npy` beside `modem.py`. Keep `payload.txt` beside `transmitter.py` if using its command-line entry point.

## Standardised behaviour preserved

The change does **not** alter:

- the chirp or Golay waveforms;
- the occupied bins, FFT length or cyclic prefix;
- the known OFDM pilot or its schedule;
- QPSK mapping;
- the Appendix-B interleaver;
- packet framing;
- the intentionally bug-compatible Jossy LDPC graph and fingerprint;
- the transmitter waveform.

Only receiver-side synchronisation, resampling, channel estimation, equalisation, reliability weighting and diagnostics are changed.

## Run

```bash
python receiver.py recorded.wav
```

The decoded file is written under:

```text
received/
```

Diagnostics are overwritten cleanly for each WAV stem under:

```text
diagnostics/<wav-stem>/
```

The dashboard and impulse-response plot open automatically. To save them without opening windows:

```bash
python receiver.py recorded.wav --no-plots
```

To disable all diagnostics:

```bash
python receiver.py recorded.wav --no-diagnostics
```

## Receiver processing

1. Detect all ten repeated chirps and fit their sub-sample positions to estimate the coarse sample-rate ratio.
2. Resample from the original recording onto the transmitter's nominal time grid.
3. Measure residual payload timing drift from known pilots and from the fourth power of unknown QPSK data symbols.
4. Re-estimate the total SFO and resample again from the original recording rather than cascading interpolators.
5. Estimate the channel from all Golay A/B pulses.
6. Replace/refine the occupied-band estimate using every standard known OFDM pilot.
7. Separate pilot-to-pilot gain, common phase and linear phase ramp, then interpolate these across the payload.
8. Apply regularised one-tap equalisation plus a small decision-directed residual phase-ramp correction.
9. Form carrier reliability weights from channel strength, out-of-band noise and pilot-channel dispersion before LDPC decoding.

## Diagnostic outputs

- `receiver_dashboard.png`: chirp fit, timing drift, constellation smear before correction, corrected constellation, channel response and EVM.
- `channel_impulse_response.png`: pilot-refined impulse response.
- `timing_drift.csv`: per-block timing/phase-ramp estimates before and after correction.
- `channel_estimate.csv`: complex occupied-band channel estimate.
- `evm_by_data_block.csv`: EVM over the packet.
- `summary.json`: estimated SFO and summary metrics.

## Validation

Run:

```bash
python test_interop.py
```

This checks the Jossy LDPC fingerprint, interleaver/QPSK round trip, header repair, text salvage, residual-SFO fit and a synthetic acoustic multipath/SFO decode when the real known-pilot file is present.
