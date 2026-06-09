# JOSS-F standard modem

This workspace is the supplied modem with one functional change: the known OFDM
pilot is loaded from the agreed `spliced_seed_qpsk.npy` standardisation file.

Place these four files together:

- `modem.py`
- `transmitter.py`
- `receiver.py`
- `spliced_seed_qpsk.npy`

The `.npy` file is expected to contain exactly 854 complex QPSK values in
ascending occupied-bin order, corresponding to FFT bins 171 through 1024
inclusive. Each value must be one of `1+1j`, `-1+1j`, `1-1j`, `-1-1j`.

All other modem behaviour is unchanged.
