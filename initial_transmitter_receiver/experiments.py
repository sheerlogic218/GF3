import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import fftconvolve
import zlib


class MatchedAudioOFDM:
    def __init__(
        self,
        fs=48000,
        N=1024,
        CP=256,
        f_min=800,
        f_max=8000,
        pilot_spacing=8,
        header_repetition=5,
        volume=0.75,
    ):
        self.fs = fs
        self.N = N
        self.CP = CP
        self.symbol_len = N + CP

        self.f_min = f_min
        self.f_max = f_max
        self.pilot_spacing = pilot_spacing
        self.header_repetition = header_repetition
        self.volume = volume

        freqs = np.arange(N) * fs / N
        self.active_bins = np.where(
            (freqs >= f_min) & (freqs <= f_max) & (np.arange(N) < N // 2)
        )[0]

        self.pilot_bins = self.active_bins[::pilot_spacing]
        self.data_bins = np.array(
            [b for b in self.active_bins if b not in set(self.pilot_bins)]
        )

        self.preamble = self.make_chirp_preamble()

    # ---------------- bits ----------------

    def bytes_to_bits(self, b):
        return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

    def bits_to_bytes(self, bits):
        bits = bits[: len(bits) // 8 * 8]
        return np.packbits(bits).tobytes()

    # ---------------- header ----------------
    # Header format:
    # magic 2 bytes | version 1 byte | payload length 2 bytes | crc32 4 bytes
    # total = 9 bytes = 72 bits
    # then repeated header_repetition times

    def make_header_bits(self, payload_bytes):
        magic = 0xBEEF
        version = 1
        length = len(payload_bytes)
        crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

        header = (
            magic.to_bytes(2, "big")
            + version.to_bytes(1, "big")
            + length.to_bytes(2, "big")
            + crc.to_bytes(4, "big")
        )

        return self.bytes_to_bits(header)

    def parse_header_bits(self, bits):
        b = self.bits_to_bytes(bits)

        magic = int.from_bytes(b[0:2], "big")
        version = int.from_bytes(b[2:3], "big")
        length = int.from_bytes(b[3:5], "big")
        crc = int.from_bytes(b[5:9], "big")

        return magic, version, length, crc

    # ---------------- modulation ----------------

    def bpsk_mod(self, bits):
        return 2 * bits.astype(float) - 1

    def bpsk_demod(self, symbols):
        return (symbols.real > 0).astype(np.uint8)

    def qpsk_mod(self, bits):
        if len(bits) % 2:
            bits = np.append(bits, 0)

        pairs = bits.reshape(-1, 2)
        out = []

        for a, b in pairs:
            if a == 0 and b == 0:
                out.append(1 + 1j)
            elif a == 0 and b == 1:
                out.append(-1 + 1j)
            elif a == 1 and b == 1:
                out.append(-1 - 1j)
            else:
                out.append(1 - 1j)

        return np.array(out) / np.sqrt(2)

    def qpsk_demod(self, symbols):
        bits = []

        for s in symbols:
            if s.real >= 0 and s.imag >= 0:
                bits += [0, 0]
            elif s.real < 0 and s.imag >= 0:
                bits += [0, 1]
            elif s.real < 0 and s.imag < 0:
                bits += [1, 1]
            else:
                bits += [1, 0]

        return np.array(bits, dtype=np.uint8)

    # ---------------- waveform ----------------

    def make_chirp_preamble(self, duration=1.0, f0=500, f1=8000):
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        k = (f1 - f0) / duration
        phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
        chirp = np.sin(phase)

        fade_len = int(0.02 * self.fs)
        fade = np.ones_like(chirp)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)

        return 0.6 * chirp * fade

    def make_ofdm_symbol(self, bin_values):
        X = np.zeros(self.N, dtype=complex)

        for k, v in bin_values.items():
            X[k] = v
            X[-k] = np.conj(v)

        x = np.fft.ifft(X).real
        x_cp = np.concatenate([x[-self.CP:], x])
        return x_cp

    def make_full_pilot_symbol(self):
        bins = {k: 1 + 0j for k in self.active_bins}
        return self.make_ofdm_symbol(bins)

    def make_data_symbols(self, mod_symbols):
        symbols = []
        i = 0

        while i < len(mod_symbols):
            bins = {}

            # interlaced comb pilots
            for k in self.pilot_bins:
                bins[k] = 1 + 0j

            # data only goes in non-pilot bins
            for k in self.data_bins:
                if i < len(mod_symbols):
                    bins[k] = mod_symbols[i]
                    i += 1

            symbols.append(self.make_ofdm_symbol(bins))

        return symbols

    # ---------------- transmitter ----------------

    def make_tx_wav(self, message, filename="tx_message.wav"):
        payload_bytes = message.encode("utf-8")
        payload_bits = self.bytes_to_bits(payload_bytes)

        header_bits = self.make_header_bits(payload_bytes)
        header_bits_rep = np.repeat(header_bits, self.header_repetition)

        header_symbols = self.bpsk_mod(header_bits_rep)
        payload_symbols = self.qpsk_mod(payload_bits)

        silence = np.zeros(int(0.3 * self.fs))
        guard = np.zeros(int(0.1 * self.fs))

        tx_parts = [
            silence,
            self.preamble,
            guard,
            self.make_full_pilot_symbol(),
            *self.make_data_symbols(header_symbols),
            *self.make_data_symbols(payload_symbols),
            silence,
        ]

        tx = np.concatenate(tx_parts)
        tx = tx / np.max(np.abs(tx)) * self.volume

        wavfile.write(filename, self.fs, (tx * 32767).astype(np.int16))

        print("Saved:", filename)
        print("Message:", message)
        print("Payload bytes:", len(payload_bytes))
        print("Active bins:", len(self.active_bins))
        print("Pilot bins:", len(self.pilot_bins))
        print("Data bins:", len(self.data_bins))

        return tx

    # ---------------- recording/loading ----------------

    def record_rx_wav(self, duration=10, filename="recorded_rx.wav"):
        print("Recording...")
        audio = sd.rec(
            int(duration * self.fs),
            samplerate=self.fs,
            channels=1,
            dtype="float64",
        )
        sd.wait()

        audio = audio[:, 0]
        audio = audio / (np.max(np.abs(audio)) + 1e-12)

        wavfile.write(filename, self.fs, (audio * 32767).astype(np.int16))
        print("Saved recording:", filename)

        return audio

    def load_wav(self, filename):
        fs, x = wavfile.read(filename)

        if x.ndim > 1:
            x = x[:, 0]

        x = x.astype(float)
        x = x / (np.max(np.abs(x)) + 1e-12)

        if fs != self.fs:
            raise ValueError(f"Expected fs={self.fs}, got fs={fs}")

        return x

    # ---------------- receiver ----------------

    def find_preamble_end(self, rx, show_plot=True):
        corr = fftconvolve(rx, self.preamble[::-1], mode="valid")
        mag = np.abs(corr)

        preamble_start = np.argmax(mag)
        preamble_end = preamble_start + len(self.preamble)

        print("Preamble start:", preamble_start)
        print("Preamble end:", preamble_end)
        print("Preamble time:", preamble_start / self.fs, "s")

        if show_plot:
            plt.figure(figsize=(10, 3))
            plt.plot(mag)
            plt.axvline(preamble_start, color="red")
            plt.title("Preamble correlation")
            plt.xlabel("Sample")
            plt.ylabel("Correlation")
            plt.grid(True)
            plt.show()

        return preamble_end

    def extract_symbols_after(self, rx, start):
        x = rx[start:]
        n_symbols = len(x) // self.symbol_len
        x = x[: n_symbols * self.symbol_len]

        blocks = x.reshape(n_symbols, self.symbol_len)
        symbols_no_cp = blocks[:, self.CP:]

        return symbols_no_cp

    def estimate_channel_from_full_pilot(self, pilot_symbol):
        Y = np.fft.fft(pilot_symbol)

        H_active = np.array([Y[k] / (1 + 0j) for k in self.active_bins])

        H_real = np.interp(np.arange(self.N), self.active_bins, H_active.real)
        H_imag = np.interp(np.arange(self.N), self.active_bins, H_active.imag)

        H = H_real + 1j * H_imag
        return H

    def equalise_with_comb_pilots(self, symbol):
        Y = np.fft.fft(symbol)

        H_pilot = np.array([Y[k] / (1 + 0j) for k in self.pilot_bins])

        H_real = np.interp(np.arange(self.N), self.pilot_bins, H_pilot.real)
        H_imag = np.interp(np.arange(self.N), self.pilot_bins, H_pilot.imag)

        H = H_real + 1j * H_imag
        EQ = Y / (H + 1e-12)

        return EQ, H

    def decode_rx_wav(self, filename="recorded_rx.wav", show_plots=True):
        rx = self.load_wav(filename)

        preamble_end = self.find_preamble_end(rx, show_plot=show_plots)

        # transmitter inserted 0.1 seconds guard after preamble
        start = preamble_end + int(0.1 * self.fs)

        symbols = self.extract_symbols_after(rx, start)

        if len(symbols) < 2:
            raise ValueError("Not enough OFDM symbols found.")

        # first OFDM symbol is full pilot
        full_pilot_symbol = symbols[0]
        H0 = self.estimate_channel_from_full_pilot(full_pilot_symbol)

        header_bits_raw = 9 * 8
        header_bits_rep_len = header_bits_raw * self.header_repetition
        header_symbols_needed = header_bits_rep_len
        header_ofdm_symbols = int(np.ceil(header_symbols_needed / len(self.data_bins)))

        header_rx_symbols = []

        for s in symbols[1 : 1 + header_ofdm_symbols]:
            EQ, _ = self.equalise_with_comb_pilots(s)

            for k in self.data_bins:
                if len(header_rx_symbols) < header_symbols_needed:
                    header_rx_symbols.append(EQ[k])

        header_rx_symbols = np.array(header_rx_symbols)
        header_bits_rep = self.bpsk_demod(header_rx_symbols)

        header_bits = []

        for i in range(header_bits_raw):
            chunk = header_bits_rep[
                i * self.header_repetition : (i + 1) * self.header_repetition
            ]
            header_bits.append(1 if np.mean(chunk) > 0.5 else 0)

        header_bits = np.array(header_bits, dtype=np.uint8)

        magic, version, payload_len, expected_crc = self.parse_header_bits(header_bits)

        print("\nHEADER")
        print("magic:", hex(magic))
        print("version:", version)
        print("payload_len bytes:", payload_len)
        print("expected CRC:", expected_crc)

        if magic != 0xBEEF:
            print("WARNING: bad magic. Sync/header likely failed.")

        payload_bits_needed = payload_len * 8
        payload_qpsk_symbols_needed = int(np.ceil(payload_bits_needed / 2))

        payload_start_symbol = 1 + header_ofdm_symbols
        payload_rx_symbols = []

        comb_Hs = []

        for s in symbols[payload_start_symbol:]:
            EQ, H_comb = self.equalise_with_comb_pilots(s)
            comb_Hs.append(H_comb)

            for k in self.data_bins:
                if len(payload_rx_symbols) < payload_qpsk_symbols_needed:
                    payload_rx_symbols.append(EQ[k])

        payload_rx_symbols = np.array(payload_rx_symbols)

        payload_bits = self.qpsk_demod(payload_rx_symbols)
        payload_bits = payload_bits[:payload_bits_needed]

        payload_bytes = self.bits_to_bytes(payload_bits)
        got_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
        crc_ok = got_crc == expected_crc

        message = payload_bytes.decode("utf-8", errors="replace")

        print("\nMESSAGE")
        print(message)

        print("\nCRC")
        print("got:     ", got_crc)
        print("expected:", expected_crc)
        print("passed:  ", crc_ok)

        self.last_rx = rx
        self.last_H0 = H0
        self.last_payload_symbols = payload_rx_symbols
        self.last_message = message
        self.last_crc_ok = crc_ok

        if show_plots:
            self.plot_channel()
            self.plot_constellation()

        return message, crc_ok

    # ---------------- plots ----------------

    def plot_channel(self):
        freqs = np.arange(self.N) * self.fs / self.N

        plt.figure(figsize=(10, 4))
        plt.plot(
            freqs[self.active_bins],
            20 * np.log10(np.abs(self.last_H0[self.active_bins]) + 1e-12),
        )
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Magnitude / dB")
        plt.title("Initial channel estimate from full pilot symbol")
        plt.grid(True)
        plt.show()

    def plot_constellation(self):
        z = self.last_payload_symbols

        plt.figure(figsize=(5, 5))
        plt.scatter(z.real, z.imag, s=8)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title("Equalised QPSK payload constellation")
        plt.grid(True)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    modem = MatchedAudioOFDM(
        fs=48000,
        N=1024,
        CP=512,
        f_min=800,
        f_max=8000,
        pilot_spacing=1,
        header_repetition=5,
    )

    
    modem.make_tx_wav(
        "hello mate this is a full matched OFDM audio packet with pilots header and CRC",
        filename="tx_message1.wav",
    )
    """
    
    modem.record_rx_wav(duration=10, filename="recorded_rx.wav")


    modem.decode_rx_wav("recorded_rx.wav")
    """
    
    