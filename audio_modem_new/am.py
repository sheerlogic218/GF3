"""
GF3 Audio Modem - Group 9
=========================

Single-file, class-based implementation of the modulation/synchronisation/
channel-estimation scheme agreed in the standardisation meetings.

Frame layout that this code transmits and decodes:

    [silence] [PREAMBLE] [guard] [GOLAY block] [guard] [OFDM data symbols] [silence]

    PREAMBLE    : 10 linear up-chirps, each 1024 samples, 20 Hz -> 20 kHz,
                  with 4000-sample gaps between chirps. Used for synchronisation
                  (matched filter over the WHOLE preamble).
    GOLAY block : [a] [gap] [b]   (Golay pair) -> used for channel estimation.
                  Sits in the "1 s space reserved for pilots" agreed in meeting 6.
    OFDM data   : QPSK on N=1024 DFT, cyclic prefix CP=1024,
                  data carriers in the 4 kHz - 13 kHz band.

The byte stream carried by the OFDM data section is:

    [2-byte header length][4-byte file size][file name ...][file bytes ...]

NOTE on the rules: only FFT / convolution / correlation primitives are used.
Chirp generation, Golay generation, QPSK (de)mapping, synchronisation and
channel estimation are all written from scratch (not pulled from a library).

Usage
-----
    python audio_modem.py selftest                 # simulated channel, no audio
    python audio_modem.py tx  --msg "hello" --wav tx.wav
    python audio_modem.py tx  --file myfile.bin    --wav tx.wav
    python audio_modem.py decode --wav recorded.wav
    python audio_modem.py rx  --seconds 14         # record from mic then decode
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
    # ---- agreed standard parameters --------------------------------------
    FS = 48_000            # 48 kHz sample rate (meeting 3)

    # OFDM data section (meeting 5: data length & CP = 1024 each)
    N = 1024               # DFT / data-section length
    CP = 1024              # cyclic prefix length
    DATA_F_MIN = 4_000     # data band lower edge (meeting 6: 4-13 kHz)
    DATA_F_MAX = 13_000    # data band upper edge

    # Synchronisation chirp (meeting 6 votes)
    CHIRP_LEN = 1024       # samples per chirp (vote: 1024)
    CHIRP_F0 = 20          # Hz   (vote: 20 Hz - 20 kHz)
    CHIRP_F1 = 20_000      # Hz
    CHIRP_REPEATS = 10     # number of chirps (vote: 10)
    CHIRP_GAP = 4_000      # samples of silence between chirps (vote: 4000)
    CHIRP_FADE = 32        # fade in/out samples to avoid clicks
    CHIRP_AMP = 0.6

    # Golay channel-estimation block (Group 1 white paper, our group's choice)
    GOLAY_LEN = 1024       # length of each Golay sequence (power of 2)
    GOLAY_GAP = 1024       # gap between a and b (>= channel length, ~CP)
    GOLAY_AMP = 0.6

    # generic
    GUARD = 2048           # guard samples around the golay block
    SILENCE = 4800         # leading/trailing silence (0.1 s)
    VOLUME = 0.7           # final peak of the transmitted wav

    def __init__(self):
        self.preamble = self._make_preamble()
        self.golay_a, self.golay_b = self._make_golay(self.GOLAY_LEN)
        self.golay_block = self._make_golay_block()
        self.data_bins = self._data_bins()

    # ======================================================================
    # 1. SYNCHRONISATION SIGNAL  (repeated linear up-chirp)
    # ======================================================================
    def _single_chirp(self) -> NDArray[np.float64]:
        n = self.CHIRP_LEN
        t = np.arange(n) / self.FS
        T = n / self.FS
        # linear (up) chirp: instantaneous freq sweeps f0 -> f1 over T
        k = (self.CHIRP_F1 - self.CHIRP_F0) / T
        phase = 2 * np.pi * (self.CHIRP_F0 * t + 0.5 * k * t ** 2)
        x = np.sin(phase)
        # short fade so the chirp edges do not click
        fade = np.ones(n)
        f = min(self.CHIRP_FADE, n // 2)
        fade[:f] = np.linspace(0, 1, f)
        fade[-f:] = np.linspace(1, 0, f)
        return self.CHIRP_AMP * x * fade

    def _make_preamble(self) -> NDArray[np.float64]:
        chirp = self._single_chirp()
        gap = np.zeros(self.CHIRP_GAP)
        parts = []
        for i in range(self.CHIRP_REPEATS):
            parts.append(chirp)
            if i != self.CHIRP_REPEATS - 1:
                parts.append(gap)         # gap BETWEEN chirps only
        return np.concatenate(parts)

    # ======================================================================
    # 2. CHANNEL-ESTIMATION SIGNAL  (Golay pair)
    # ======================================================================
    def _make_golay(self, n: int):
        """Recursively generate a complementary Golay pair of length n."""
        a = np.array([1.0])
        b = np.array([1.0])
        while len(a) < n:
            na = np.concatenate([a, b])
            nb = np.concatenate([a, -b])
            a, b = na, nb
        return a[:n], b[:n]

    def _make_golay_block(self) -> NDArray[np.float64]:
        gap = np.zeros(self.GOLAY_GAP)
        return self.GOLAY_AMP * np.concatenate([self.golay_a, gap, self.golay_b])

    def estimate_channel(self, rx: NDArray[np.float64], golay_start: int):
        """
        Recover the impulse response from the received Golay block.

        Property used (sum of Golay autocorrelations is a delta):
            Raa[k] + Rbb[k] = 2N * delta[k]
        So correlating the received a-segment with a, the received b-segment
        with b, and summing, yields 2N * h[k].
        """
        L = self.GOLAY_LEN
        cap = L + self.GOLAY_GAP                       # capture a's full response
        a_rx = rx[golay_start: golay_start + cap]
        b_off = golay_start + L + self.GOLAY_GAP
        b_rx = rx[b_off: b_off + cap]

        # pad to equal length (in case the recording ends early)
        if len(a_rx) < cap:
            a_rx = np.pad(a_rx, (0, cap - len(a_rx)))
        if len(b_rx) < cap:
            b_rx = np.pad(b_rx, (0, cap - len(b_rx)))

        # cross-correlation = convolution with the time-reversed sequence
        corr_a = fftconvolve(a_rx, self.golay_a[::-1], mode="full")
        corr_b = fftconvolve(b_rx, self.golay_b[::-1], mode="full")
        h_sum = corr_a + corr_b

        # zero lag sits at index L-1; take N taps from there
        zero_lag = L - 1
        h = h_sum[zero_lag: zero_lag + self.N] / (2.0 * L)
        if len(h) < self.N:
            h = np.pad(h, (0, self.N - len(h)))

        H = np.fft.fft(h, self.N)
        return h, H

    # ======================================================================
    # 3. QPSK  (Gray mapping, matches the Week-1 / modulator convention)
    #    first bit (MSB) -> imag sign, second bit (LSB) -> real sign
    # ======================================================================
    def qpsk_map(self, bits: NDArray[np.uint8]) -> NDArray[np.complex128]:
        if len(bits) % 2:
            bits = np.append(bits, 0)
        pairs = bits.reshape(-1, 2)
        msb = pairs[:, 0]
        lsb = pairs[:, 1]
        real = 1 - 2 * lsb.astype(float)
        imag = 1 - 2 * msb.astype(float)
        return (real + 1j * imag) / np.sqrt(2)

    def qpsk_demap(self, syms: NDArray[np.complex128]) -> NDArray[np.uint8]:
        bits = np.empty(2 * len(syms), dtype=np.uint8)
        bits[0::2] = (syms.imag < 0).astype(np.uint8)   # MSB
        bits[1::2] = (syms.real < 0).astype(np.uint8)   # LSB
        return bits

    # ======================================================================
    # 4. OFDM
    # ======================================================================
    def _data_bins(self) -> NDArray[np.int64]:
        res = self.FS / self.N
        lo = int(np.ceil(self.DATA_F_MIN / res))
        hi = min(int(np.floor(self.DATA_F_MAX / res)), self.N // 2 - 1)
        return np.arange(lo, hi + 1, dtype=int)

    def ofdm_modulate(self, bits: NDArray[np.uint8]) -> NDArray[np.float64]:
        syms = self.qpsk_map(bits)
        nbins = len(self.data_bins)
        pad = (-len(syms)) % nbins
        if pad:
            syms = np.concatenate([syms, np.zeros(pad, dtype=complex)])
        rows = syms.reshape(-1, nbins)

        out = []
        for row in rows:
            X = np.zeros(self.N, dtype=complex)
            X[self.data_bins] = row
            X[self.N - self.data_bins] = np.conj(row)   # Hermitian symmetry
            x = np.fft.ifft(X).real
            x_cp = np.concatenate([x[-self.CP:], x])    # prepend cyclic prefix
            out.append(x_cp)
        return np.concatenate(out)

    def ofdm_demodulate(self, data: NDArray[np.float64],
                        H: NDArray[np.complex128]) -> NDArray[np.uint8]:
        sym_len = self.N + self.CP
        nsym = len(data) // sym_len
        data = data[: nsym * sym_len].reshape(nsym, sym_len)
        blocks = data[:, self.CP:]                       # drop cyclic prefix
        Y = np.fft.fft(blocks, axis=1)
        EQ = Y / (H[None, :] + EPS)                      # one-tap equalisation
        syms = EQ[:, self.data_bins].reshape(-1)
        return self.qpsk_demap(syms)

    # ======================================================================
    # 5. HEADER   [2B header length][4B file size][filename][file bytes]
    # ======================================================================
    def build_payload(self, file_bytes: bytes, filename: str) -> bytes:
        name = filename.encode("utf-8")
        header_len = 2 + 4 + len(name)
        header = struct.pack(">H", header_len) + struct.pack(">I", len(file_bytes)) + name
        return header + file_bytes

    def parse_payload(self, data: bytes):
        if len(data) < 6:
            raise ValueError("payload too short to contain a header")
        header_len = struct.unpack(">H", data[0:2])[0]
        file_size = struct.unpack(">I", data[2:6])[0]
        name = data[6:header_len].decode("utf-8", errors="replace")
        file_bytes = data[header_len: header_len + file_size]
        return name, file_size, file_bytes

    # ======================================================================
    # 6. TRANSMIT  (build a .wav)
    # ======================================================================
    def transmit(self, file_bytes: bytes, filename: str,
                 wav_path: str = "tx.wav") -> NDArray[np.float64]:
        payload = self.build_payload(file_bytes, filename)
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        data_signal = self.ofdm_modulate(bits)

        tx = np.concatenate([
            np.zeros(self.SILENCE),
            self.preamble,
            np.zeros(self.GUARD),
            self.golay_block,
            np.zeros(self.GUARD),
            data_signal,
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

    # ======================================================================
    # 7. RECEIVE / DECODE
    # ======================================================================
    @staticmethod
    def _load_wav(path: str, fs_expected: int) -> NDArray[np.float64]:
        fs, x = wavfile.read(path)
        if fs != fs_expected:
            raise ValueError(f"expected fs={fs_expected}, got {fs}")
        x = x.astype(np.float64)
        if x.ndim > 1:
            x = x[:, 0]
        return x / (np.max(np.abs(x)) + EPS)

    def synchronise(self, rx: NDArray[np.float64]) -> int:
        """Matched filter over the whole repeated-chirp preamble."""
        corr = fftconvolve(rx, self.preamble[::-1], mode="valid")
        start = int(np.argmax(np.abs(corr)))
        return start

    def decode_signal(self, rx: NDArray[np.float64]):
        sync_start = self.synchronise(rx)
        preamble_end = sync_start + len(self.preamble)

        golay_start = preamble_end + self.GUARD
        h, H = self.estimate_channel(rx, golay_start)

        data_start = golay_start + len(self.golay_block) + self.GUARD
        data = rx[data_start:]

        bits = self.ofdm_demodulate(data, H)
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

    def record_and_decode(self, seconds: float = 14.0,
                          wav_path: str = "recorded_rx.wav"):
        import sounddevice as sd      # imported lazily so tx works without it
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
# Self-test: push a transmission through a simulated room channel and decode.
# ==========================================================================
def _simulated_channel(tx, fs=48_000, snr_db=25.0, seed=7):
    rng = np.random.default_rng(seed)
    h = np.zeros(900)
    for delay, gain in [(0, 1.0), (155, 0.35), (430, -0.22), (760, 0.15)]:
        h[delay] += gain
    h /= np.sqrt(np.sum(h ** 2))
    y = fftconvolve(tx, h, mode="full")
    y = np.concatenate([np.zeros(3000), y, np.zeros(3000)])   # unknown delay
    p = np.mean(y ** 2)
    y += rng.standard_normal(len(y)) * np.sqrt(p / (10 ** (snr_db / 10)))
    return y


def _selftest():
    modem = AudioModem()
    msg = ("This is a Group 9 audio-modem self test. " * 260).encode("utf-8")
    print(f"Original message: {len(msg)} bytes (~{len(msg)/1024:.1f} kB)\n")

    tx = modem.transmit(msg, "selftest.txt", "selftest_tx.wav")
    rx = _simulated_channel(tx)

    print("\nDecoding simulated reception:")
    name, file_bytes, h, H = modem.decode_signal(rx)

    n = min(len(msg), len(file_bytes))
    a = np.frombuffer(msg[:n], dtype=np.uint8)
    b = np.frombuffer(bytes(file_bytes[:n]), dtype=np.uint8)
    ber = np.mean(np.unpackbits(a) != np.unpackbits(b)) if n else 1.0
    ok = (bytes(file_bytes) == msg)
    print(f"\n  BER           : {ber:.4%}")
    print(f"  exact match   : {ok}")
    return ok


def main():
    p = argparse.ArgumentParser(description="GF3 audio modem")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest")

    t = sub.add_parser("tx")
    t.add_argument("--msg")
    t.add_argument("--file")
    t.add_argument("--wav", default="tx.wav")

    d = sub.add_parser("decode")
    d.add_argument("--wav", required=True)

    r = sub.add_parser("rx")
    r.add_argument("--seconds", type=float, default=14.0)
    r.add_argument("--wav", default="recorded_rx.wav")

    args = p.parse_args()
    modem = AudioModem()

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
        modem.transmit(data, name, args.wav)
    elif args.cmd == "decode":
        modem.decode_wav(args.wav)
    elif args.cmd == "rx":
        modem.record_and_decode(args.seconds, args.wav)


if __name__ == "__main__":
    main()