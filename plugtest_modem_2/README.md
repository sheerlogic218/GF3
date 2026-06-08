# JOSS-F plugtest modem

Only three Python files are used:

- `modem.py` — shared protocol and signal-processing classes.
- `transmitter.py` — reads `payload.txt` and creates a WAV file.
- `receiver.py` — asks which recorded WAV file to decode.

All input and output paths are resolved relative to this folder, not the terminal's
current directory. The folder can therefore be moved to another location or laptop.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Transmit

Put the file to send at `payload.txt`, then run:

```bash
python3 transmitter.py
```

The program asks:

```text
Output WAV name [tx]:
```

Enter `tx` or `tx.wav`. Press Enter to use `tx.wav`. An existing file with the
same name is overwritten.

## Decode a recording

Place the recording in this folder and run:

```bash
python3 receiver.py
```

The program asks:

```text
WAV file to decode [recorded]:
```

Enter `recorded` or `recorded.wav`. Press Enter to use `recorded.wav`. The
recovered payload is saved in this folder as `received_<original filename>`.

## Standard interpretation

The implementation follows JOSS-F vF. It uses 854 carriers (bins 171–1024)
because Appendix B's mandatory interleaver is dimensioned for 854 carriers,
despite the written strict 2–12 kHz rule yielding 853. The Golay layout is
`A`, 2048 zero samples, then `B`, matching the 51,200-sample preamble length.
