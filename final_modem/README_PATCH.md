# JOSS-F interoperability patch

## Critical correction

The previous `WiMaxLDPC` implementation scaled the IEEE 802.16 protograph shifts by `Z/96` for `Z=61`. Jossy's supplied `ldpc.py` does not do this: it uses the listed shifts directly and reduces them modulo `Z` through cyclic indexing. Those two choices define different LDPC codes.

The patched `modem.py` deliberately follows Jossy's supplied implementation. Its deterministic reference fingerprint is:

```text
270111995b23bc1d540f4f1bdbaf418aa3538d376cf0826bb5cdd67603c76f2d
```

The patched encoder was checked against Jossy's encoder for 20 random 732-bit information blocks; every 1464-bit codeword matched exactly.

## Header fallback

`receiver.py` now separates physical-layer decoding from packet parsing:

1. Every complete 35-codeword/30-data-symbol group is LDPC-decoded.
2. The normal JOSS-F header is parsed strictly.
3. If that fails, the receiver tries:
   - preserving valid A/B lengths while salvaging a damaged UTF-8 filename;
   - an unambiguous repair of up to three flipped bits in the fixed 48-bit A/B fields.
4. If no safe repair exists, it writes every recovered byte to:

```text
received_<wav-name>_raw_ldpc_stream.bin
```

A JSON sidecar records synchronisation, LDPC success, fingerprints and the first 64 decoded bytes. The raw stream includes the corrupt header and post-payload padding because the exact payload boundaries cannot be determined uniquely without A and B.

When payload boundaries are known externally:

```bash
python receiver.py recording.wav \
  --header-bytes 17 \
  --payload-bytes 12345 \
  --filename recovered.dat
```

To retain the old abort-on-header-error behaviour:

```bash
python receiver.py recording.wav --strict-header
```

## Required pilot file

Place the canonical `spliced_seed_qpsk.npy` beside `modem.py`. Both transmitter and receiver print a content hash of its 854 complex symbols. Other groups must print the same hash; merely having a file with the same name is insufficient.

## Checks

Run:

```bash
python test_interop.py
```

This checks the Jossy LDPC fingerprint, QPSK mapping, Appendix-B interleaver round-trip and a damaged-header repair. It also checks and prints the pilot fingerprint when `spliced_seed_qpsk.npy` is present.

## Additional header-failure .txt salvage

The receiver now tries a human-readable text salvage path when the JOSS-F header
is unrecoverable.  If the decoded LDPC byte stream contains an early filename
ending in `.txt`, the receiver assumes the payload begins immediately after that
marker and writes:

```text
received_<recording>_best_effort.txt
```

If the `.txt` marker itself is corrupted, it also tries short plausible header
lengths and picks the offset whose following bytes look most like text.  This is
only a best-effort inspection file: the exact payload length still comes from the
header, so with a dead header the receiver cannot prove the final byte boundary.
The raw LDPC stream is still always preserved as:

```text
received_<recording>_raw_ldpc_stream.bin
```
