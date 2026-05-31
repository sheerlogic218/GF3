"""
JOSS-D Audio Modem  (JOSSy OFDM Signal Standard, v.D)
=====================================================

STANDARD-COMPLIANT, interoperability build for the plugfest. This deliberately
uses ONLY what the JOSS-D standard document fixes -- no Golay pairs (that was our
group's private channel-estimation choice). Channel estimation here is done from
the repeated chirps themselves ("chirp only", the option that won the meeting-6
vote), and the 1-second post-chirp space is left empty as clause 2.2 requires.

Frame (exactly per JOSS-D):

    [silence] [10 chirps, 4000-sample gaps] [1 second empty] [OFDM data] [silence]

Clause mapping
--------------
  1.0  Sample rate ............ 48 kHz
  2.1  Initial chirp .......... linear, 1024 samples (0.0213 s), 20 Hz - 20 kHz
  2.2  Chirps & spacing ....... 10 chirps, 4000-sample gaps; then 1 s left EMPTY
                                (reserved for future channel-estimation symbols)
  3.0  OFDM symbols ........... N = 1024, cyclic prefix L = 1024, band 4-13 kHz
  4.0  Constellation .......... QPSK
  5.0  Source coding .......... none agreed yet -> raw bytes
  6.0  Header ................. fixed format: [2-byte A = header length]
                                [4-byte B = file size][file name bytes]

Channel estimation (clause 2.2 "chirp only" reading)
----------------------------------------------------
Each chirp is exactly N samples. For each received chirp we deconvolve the known
chirp by a regularised frequency-domain division, take the impulse response, and
average over all 10 chirps. The 1-second space stays silent until the cohort
agrees on pilot symbols to put there.

Interoperability conventions that MUST match other groups (typical plugfest
mismatches -- confirm these on Slack):
  * QPSK Gray map, anti-clockwise from pi/4: 00->(+,+) 01->(-,+) 11->(-,-)
    10->(+,-); bits taken MSB-first. (Matches the Week-1 challenge spec.)
  * Data bins = ceil(4000/df) .. floor(13000/df) with df = 48000/1024
    -> bins 86..277 inclusive (192 carriers), filled in ascending order.
  * Hermitian symmetry X[N-k] = conj(X[k]); big-endian header integers.

Only FFT / convolution / correlation primitives are used; chirp generation,
QPSK, synchronisation and channel estimation are written from scratch.

Usage
-----
    python3 am_jossd.py selftest
    python3 am_jossd.py tx --file acoustics.txt --wav tx.wav
    python3 am_jossd.py decode --wav recorded.wav
    python3 am_jossd.py rx --seconds 16
"""

from __future__ import annotations

import argparse
import struct
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import fftconvolve

EPS = 1e-12


class AudioModem:
    # ---- JOSS-D fixed parameters -----------------------------------------
    FS = 48_000            # clause 1.0

    N = 1024               # clause 3.0  DFT / data-section length
    CP = 1024              # clause 3.0  cyclic prefix length
    DATA_F_MIN = 4_000     # clause 3.0  band lower edge
    DATA_F_MAX = 13_000    # clause 3.0  band upper edge

    CHIRP_LEN = 1024       # clause 2.1  1024 samples (0.0213 s)
    CHIRP_F0 = 20          # clause 2.1  20 Hz
    CHIRP_F1 = 20_000      # clause 2.1  20 kHz
    CHIRP_TYPE = "linear"  # clause 2.1
    CHIRP_REPEATS = 10     # clause 2.2
    CHIRP_GAP = 4_000      # clause 2.2  samples between chirps
    RESERVED = 48_000      # clause 2.2  1 second empty after the chirps
    CHIRP_FADE = 32        # edge fade to avoid clicks (implementation detail)
    CHIRP_AMP = 0.6

    SILENCE = 4_800        # leading/trailing silence (0.1 s)
    VOLUME = 0.7           # final wav peak
    DATA_RMS = 0.22        # data section target RMS (power-normalisation)

    # channel-estimation tuning (receiver-side, not part of the standard)
    EST_WIN = 3_072        # samples captured per chirp (chirp + channel tail)
    EST_REG = 1e-3         # deconvolution regularisation
    EQ_NOISE = 0.02        # MMSE equaliser regularisation

    def __init__(self):
        self.chirp = self._make_chirp()
        self.preamble = self._make_preamble()
        self.chirp_step = self.CHIRP_LEN + self.CHIRP_GAP
        self.data_bins = self._data_bins()

    # ================= chirp (clause 2.1) =================
    def _make_chirp(self) -> NDArray[np.float64]:
        n = self.CHIRP_LEN
        t = np.arange(n) / self.FS
        T = n / self.FS
        k = (self.CHIRP_F1 - self.CHIRP_F0) / T          # linear sweep
        phase = 2 * np.pi * (self.CHIRP_F0 * t + 0.5 * k * t ** 2)
        x = np.sin(phase)
        f = min(self.CHIRP_FADE, n // 2)
        fade = np.ones(n)
        fade[:f] = np.linspace(0, 1, f)
        fade[-f:] = np.linspace(1, 0, f)
        return self.CHIRP_AMP * x * fade

    def _make_preamble(self) -> NDArray[np.float64]:
        gap = np.zeros(self.CHIRP_GAP)
        parts = []
        for i in range(self.CHIRP_REPEATS):
            parts.append(self.chirp)
            if i != self.CHIRP_REPEATS - 1:
                parts.append(gap)              # gaps BETWEEN chirps only
        return np.concatenate(parts)

    # ================= QPSK (clause 4.0) =================
    # Gray, anti-clockwise from pi/4; first bit (MSB) -> imag, second -> real.
    def qpsk_map(self, bits: NDArray[np.uint8]) -> NDArray[np.complex128]:
        if len(bits) % 2:
            bits = np.append(bits, 0)
        pairs = bits.reshape(-1, 2)
        real = 1 - 2 * pairs[:, 1].astype(float)   # LSB -> real sign
        imag = 1 - 2 * pairs[:, 0].astype(float)   # MSB -> imag sign
        return (real + 1j * imag) / np.sqrt(2)

    def qpsk_demap(self, syms: NDArray[np.complex128]) -> NDArray[np.uint8]:
        bits = np.empty(2 * len(syms), dtype=np.uint8)
        bits[0::2] = (syms.imag < 0).astype(np.uint8)   # MSB
        bits[1::2] = (syms.real < 0).astype(np.uint8)   # LSB
        return bits

    # ================= OFDM (clause 3.0) =================
    def _data_bins(self) -> NDArray[np.int64]:
        df = self.FS / self.N
        lo = int(np.ceil(self.DATA_F_MIN / df))
        hi = min(int(np.floor(self.DATA_F_MAX / df)), self.N // 2 - 1)
        return np.arange(lo, hi + 1, dtype=int)

    def ofdm_modulate(self, bits: NDArray[np.uint8]) -> NDArray[np.float64]:
        syms = self.qpsk_map(bits)
        nb = len(self.data_bins)
        pad = (-len(syms)) % nb
        if pad:
            syms = np.concatenate([syms, np.zeros(pad, dtype=complex)])
        rows = syms.reshape(-1, nb)
        out = []
        for row in rows:
            X = np.zeros(self.N, dtype=complex)
            X[self.data_bins] = row
            X[self.N - self.data_bins] = np.conj(row)    # Hermitian symmetry
            x = np.fft.ifft(X).real
            out.append(np.concatenate([x[-self.CP:], x]))  # prepend prefix
        return np.concatenate(out)

    def ofdm_demodulate(self, data, H):
        L = self.N + self.CP
        nsym = len(data) // L
        blocks = data[: nsym * L].reshape(nsym, L)[:, self.CP:]
        Y = np.fft.fft(blocks, axis=1)
        noise = self.EQ_NOISE * np.mean(np.abs(H) ** 2)   # MMSE equaliser
        EQ = Y * np.conj(H)[None, :] / (np.abs(H)[None, :] ** 2 + noise)
        return self.qpsk_demap(EQ[:, self.data_bins].reshape(-1))

    # ================= header (clause 6.0) =================
    def build_payload(self, file_bytes: bytes, filename: str) -> bytes:
        name = filename.encode("utf-8")
        header_len = 2 + 4 + len(name)
        header = struct.pack(">H", header_len) + struct.pack(">I", len(file_bytes)) + name
        return header + file_bytes

    def parse_payload(self, data: bytes):
        if len(data) < 6:
            raise ValueError("payload too short for a header")
        header_len = struct.unpack(">H", data[0:2])[0]
        file_size = struct.unpack(">I", data[2:6])[0]
        name = data[6:header_len].decode("utf-8", errors="replace")
        file_bytes = data[header_len: header_len + file_size]
        return name, file_size, file_bytes

    # ================= channel estimation from the chirps =================
    def estimate_channel(self, rx: NDArray[np.float64], sync_start: int):
        """Per-chirp regularised deconvolution, averaged over all 10 chirps."""
        M = self.EST_WIN
        X = np.fft.rfft(self.chirp, M)
        denom = np.abs(X) ** 2
        lam = self.EST_REG * denom.max()
        h_acc = np.zeros(self.N)
        for i in range(self.CHIRP_REPEATS):
            s = sync_start + i * self.chirp_step
            y = rx[s: s + M]
            if len(y) < M:
                y = np.pad(y, (0, M - len(y)))
            Hf = np.fft.rfft(y, M) * np.conj(X) / (denom + lam)
            hi = np.fft.irfft(Hf, M)[: self.N]
            h_acc += hi
        h = h_acc / self.CHIRP_REPEATS
        return h, np.fft.fft(h, self.N)

    # ================= transmit =================
    def transmit(self, file_bytes: bytes, filename: str,
                 wav_path: str = "tx.wav") -> NDArray[np.float64]:
        payload = self.build_payload(file_bytes, filename)
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        data = self.ofdm_modulate(bits)

        # power-normalisation: keep the data section's average power up near the
        # chirp level (it would otherwise sit ~20 dB down behind the mic AGC),
        # clipping only the rarest OFDM peaks.
        pre = self.preamble / (np.max(np.abs(self.preamble)) + EPS) * 0.6
        data = data / (np.std(data) + EPS) * self.DATA_RMS
        data = np.clip(data, -1.0, 1.0)

        tx = np.concatenate([
            np.zeros(self.SILENCE),
            pre,
            np.zeros(self.RESERVED),     # clause 2.2: 1 second left empty
            data,
            np.zeros(self.SILENCE),
        ])
        tx = tx / (np.max(np.abs(tx)) + EPS) * self.VOLUME
        wavfile.write(wav_path, self.FS, (tx * 32767).astype(np.int16))

        print(f"Wrote {wav_path}")
        print(f"  payload bytes : {len(payload)} ({len(file_bytes)} file + "
              f"{len(payload) - len(file_bytes)} header)")
        print(f"  data carriers : {len(self.data_bins)} "
              f"(bins {self.data_bins[0]}..{self.data_bins[-1]})")
        print(f"  duration      : {len(tx) / self.FS:.2f} s")
        return tx

    # ================= receive / decode =================
    @staticmethod
    def _load_wav(path, fs_expected):
        fs, x = wavfile.read(path)
        if fs != fs_expected:
            raise ValueError(f"expected fs={fs_expected}, got {fs}")
        x = x.astype(np.float64)
        if x.ndim > 1:
            x = x[:, 0]
        return x / (np.max(np.abs(x)) + EPS)

    def synchronise(self, rx: NDArray[np.float64]) -> int:
        corr = fftconvolve(rx, self.preamble[::-1], mode="valid")
        return int(np.argmax(np.abs(corr)))

    def decode_signal(self, rx: NDArray[np.float64]):
        sync_start = self.synchronise(rx)
        h, H = self.estimate_channel(rx, sync_start)
        data_start = sync_start + len(self.preamble) + self.RESERVED
        bits = self.ofdm_demodulate(rx[data_start:], H)
        nbytes = len(bits) // 8
        payload = np.packbits(bits[: nbytes * 8]).tobytes()
        name, file_size, file_bytes = self.parse_payload(payload)
        print(f"  sync start    : {sync_start} ({sync_start / self.FS:.3f} s)")
        print(f"  file name     : {name}")
        print(f"  file size     : {file_size} bytes")
        print(f"  recovered     : {len(file_bytes)} bytes")
        return name, file_bytes, h, H

    def decode_wav(self, wav_path: str):
        rx = self._load_wav(wav_path, self.FS)
        name, file_bytes, h, H = self.decode_signal(rx)
        if name and file_bytes:
            with open(name, "wb") as f:
                f.write(file_bytes)
            print(f"  saved -> {name}")
        return name, file_bytes

    def record_and_decode(self, seconds: float = 16.0,
                          wav_path: str = "recorded_rx.wav"):
        import sounddevice as sd
        print(f"Recording {seconds:.1f} s ...")
        audio = sd.rec(int(seconds * self.FS), samplerate=self.FS,
                       channels=1, dtype="float64")
        sd.wait()
        audio = audio[:, 0]
        audio = audio / (np.max(np.abs(audio)) + EPS)
        wavfile.write(wav_path, self.FS, (audio * 32767).astype(np.int16))
        print(f"Saved recording -> {wav_path}")
        return self.decode_wav(wav_path)


# ==========================================================================
def _simulated_channel(tx, fs=48_000, snr_db=25.0, rolloff=12_000, seed=7):
    rng = np.random.default_rng(seed)
    h = np.zeros(700)
    for d, g in [(0, 1.0), (95, 0.4), (270, -0.25), (540, 0.13)]:
        h[d] += g
    h /= np.sqrt(np.sum(h ** 2))
    y = fftconvolve(tx, h, mode="full")
    n = len(y)
    f = np.fft.rfftfreq(n, 1 / fs)
    Hf = 1 / np.sqrt(1 + (f / rolloff) ** 6)        # speaker high-freq roll-off
    y = np.fft.irfft(np.fft.rfft(y) * Hf, n)
    y = np.concatenate([np.zeros(3000), y, np.zeros(3000)])
    p = np.mean(y ** 2)
    y += rng.standard_normal(len(y)) * np.sqrt(p / (10 ** (snr_db / 10)))
    return y / (np.max(np.abs(y)) + EPS)


def _selftest():
    m = AudioModem()
    msg = ("JOSS-D plugfest interoperability self test. " * 240).encode("utf-8")
    print(f"Original message: {len(msg)} bytes (~{len(msg)/1024:.1f} kB)\n")
    tx = m.transmit(msg, "selftest.txt", "selftest_tx.wav")
    rx = _simulated_channel(tx, snr_db=30, rolloff=15_000)
    print("\nDecoding simulated reception:")
    name, fb, h, H = m.decode_signal(rx)
    n = min(len(msg), len(fb))
    ber = np.mean(np.unpackbits(np.frombuffer(msg[:n], np.uint8))
                  != np.unpackbits(np.frombuffer(bytes(fb[:n]), np.uint8))) if n else 1.0
    print(f"\n  BER           : {ber:.4%}")
    print(f"  exact match   : {bytes(fb) == msg}")


def main():
    p = argparse.ArgumentParser(description="GF3 audio modem")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    t = sub.add_parser("tx"); t.add_argument("--msg"); t.add_argument("--file"); t.add_argument("--wav", default="tx.wav")
    d = sub.add_parser("decode"); d.add_argument("--wav", required=True)
    r = sub.add_parser("rx"); r.add_argument("--seconds", type=float, default=16.0); r.add_argument("--wav", default="recorded_rx.wav")
    args = p.parse_args()
    m = AudioModem()
    if args.cmd == "selftest":
        _selftest()
    elif args.cmd == "tx":
        if args.file:
            with open(args.file, "rb") as f:
                data = f.read()
            name = args.file.split("/")[-1]
        else:
            data = (args.msg or "hello from group 9").encode("utf-8")
            name = "message.txt"
        m.transmit(data, name, args.wav)
    elif args.cmd == "decode":
        m.decode_wav(args.wav)
    elif args.cmd == "rx":
        m.record_and_decode(args.seconds, args.wav)


if __name__ == "__main__":
    main()