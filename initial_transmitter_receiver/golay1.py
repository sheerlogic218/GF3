import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly
import zlib
import struct
import time

try:
    import sounddevice as sd
except ImportError:
    sd = None


class GF3AudioModem:
    def __init__(
        self,
        fs=48000,
        n_fft=1024,
        cp_len=1024,
        chirp_len=1024,
        num_chirps=10,
        chirp_gap=4000,
        chirp_f0=20,
        chirp_f1=20000,
        data_f_min=800,
        data_f_max=13000,
        golay_len=1024,
        golay_gap=4000,
        pilot_space_sec=1.0,
        tx_volume=0.45,
    ):
        self.fs = fs
        self.n_fft = n_fft
        self.cp_len = cp_len

        self.chirp_len = chirp_len
        self.num_chirps = num_chirps
        self.chirp_gap = chirp_gap
        self.chirp_f0 = chirp_f0
        self.chirp_f1 = chirp_f1

        self.data_f_min = data_f_min
        self.data_f_max = data_f_max
        self.golay_len = golay_len
        self.golay_gap = golay_gap
        self.pilot_space_sec = pilot_space_sec
        self.tx_volume = tx_volume

        self.used_bins = self._get_used_bins()
        self.bits_per_ofdm_symbol = len(self.used_bins)

    # basic helpers

    def _get_used_bins(self):
        freqs = np.fft.rfftfreq(self.n_fft, d=1 / self.fs)
        bins = np.where((freqs >= self.data_f_min) & (freqs <= self.data_f_max))[0]
        bins = bins[(bins > 0) & (bins < self.n_fft // 2)]
        return bins.astype(int)

    def _normalise(self, x, peak=None):
        x = np.asarray(x, dtype=np.float64)
        m = np.max(np.abs(x)) + 1e-12
        if peak is None:
            peak = self.tx_volume
        return peak * x / m

    def _bytes_to_bits(self, data):
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def _bits_to_bytes(self, bits):
        bits = np.asarray(bits, dtype=np.uint8)
        usable = (len(bits) // 8) * 8
        bits = bits[:usable]
        return np.packbits(bits).tobytes()

    # chirp synchronisation

    def make_linear_chirp(self):
        t = np.arange(self.chirp_len) / self.fs
        T = self.chirp_len / self.fs
        k = (self.chirp_f1 - self.chirp_f0) / T
        phase = 2 * np.pi * (self.chirp_f0 * t + 0.5 * k * t**2)
        chirp = np.sin(phase)

        fade_len = max(8, int(0.002 * self.fs))
        fade_len = min(fade_len, len(chirp) // 2)
        fade = np.ones_like(chirp)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)
        return self._normalise(chirp * fade, peak=1.0)

    def make_repeated_chirp_preamble(self):
        chirp = self.make_linear_chirp()
        gap = np.zeros(self.chirp_gap)
        parts = []
        for _ in range(self.num_chirps):
            parts.append(chirp)
            parts.append(gap)
        return np.concatenate(parts)

    def find_sync_start(self, rx):
        preamble = self.make_repeated_chirp_preamble()
        corr = correlate(rx, preamble, mode="valid")
        metric = np.abs(corr)
        start = int(np.argmax(metric))
        peak = float(metric[start])
        floor = float(np.median(metric) + 1e-12)
        confidence = peak / floor
        return start, confidence

    # Golay channel estimation

    def make_golay_pair(self, length=None):
        if length is None:
            length = self.golay_len

        a = np.array([1.0])
        b = np.array([1.0])

        while len(a) < length:
            a_old = a.copy()
            b_old = b.copy()
            a = np.concatenate([a_old, b_old])
            b = np.concatenate([a_old, -b_old])

        a = a[:length]
        b = b[:length]
        return a, b

    def make_golay_block(self):
        a, b = self.make_golay_pair()
        gap = np.zeros(self.golay_gap)
        block = np.concatenate([a, gap, b, gap])
        return self._normalise(block, peak=1.0)

    def estimate_channel_from_golay(self, rx, golay_start):
        a, b = self.make_golay_pair()

        a_rx_start = golay_start
        b_rx_start = golay_start + len(a) + self.golay_gap

        a_rx = rx[a_rx_start:a_rx_start + len(a) + self.cp_len]
        b_rx = rx[b_rx_start:b_rx_start + len(b) + self.cp_len]

        if len(a_rx) < len(a) or len(b_rx) < len(b):
            raise ValueError("Recording too short to extract Golay pair.")

        a_rx = a_rx[:len(a)]
        b_rx = b_rx[:len(b)]

        ca = correlate(a_rx, a, mode="full")
        cb = correlate(b_rx, b, mode="full")

        mid = len(a) - 1
        h = ca[mid:mid + self.n_fft] + cb[mid:mid + self.n_fft]
        h = h / (2 * len(a))

        if len(h) < self.n_fft:
            h = np.pad(h, (0, self.n_fft - len(h)))

        H = np.fft.rfft(h, n=self.n_fft)
        H[np.abs(H) < 1e-6] = 1e-6
        return H

    # packet format

    def build_packet_bytes(self, payload):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        crc = zlib.crc32(payload) & 0xFFFFFFFF

        header = {
            "magic": b"GF3A",
            "version": 1,
            "payload_len": len(payload),
            "crc32": crc,
            "modulation": 1,
        }

        header_bytes = struct.pack(
            ">4sBIB",
            header["magic"],
            header["version"],
            header["payload_len"],
            header["modulation"],
        ) + struct.pack(">I", header["crc32"])

        header_size = len(header_bytes)

        fixed_prefix = struct.pack(">HI", header_size, len(payload))
        return fixed_prefix + header_bytes + payload

    def parse_packet_bytes(self, packet):
        if len(packet) < 6:
            raise ValueError("Packet too short.")

        header_size, payload_size = struct.unpack(">HI", packet[:6])
        header_start = 6
        header_end = header_start + header_size

        header = packet[header_start:header_end]
        payload = packet[header_end:header_end + payload_size]

        magic, version, payload_len, modulation = struct.unpack(">4sBIB", header[:10])
        crc_expected = struct.unpack(">I", header[10:14])[0]

        if magic != b"GF3A":
            raise ValueError("Bad magic number. Sync/decode probably failed.")

        if payload_len != len(payload):
            raise ValueError(f"Payload length mismatch: expected {payload_len}, got {len(payload)}.")

        crc_actual = zlib.crc32(payload) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            raise ValueError("CRC failed. Message decoded, but contains errors.")

        return payload

    # OFDM BPSK

    def bits_to_ofdm(self, bits):
        bits = np.asarray(bits, dtype=np.uint8)

        pad_len = (-len(bits)) % self.bits_per_ofdm_symbol
        if pad_len:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])

        symbols = []

        for i in range(0, len(bits), self.bits_per_ofdm_symbol):
            chunk = bits[i:i + self.bits_per_ofdm_symbol]
            bpsk = 2 * chunk.astype(float) - 1

            X = np.zeros(self.n_fft, dtype=np.complex128)
            X[self.used_bins] = bpsk
            X[-self.used_bins] = np.conj(bpsk)

            x = np.fft.ifft(X).real
            x = self._normalise(x, peak=0.8)

            x_cp = np.concatenate([x[-self.cp_len:], x])
            symbols.append(x_cp)

        return np.concatenate(symbols), pad_len

    def ofdm_to_bits(self, rx_data, H, expected_num_bits):
        sym_len = self.n_fft + self.cp_len
        num_symbols = len(rx_data) // sym_len
        rx_data = rx_data[:num_symbols * sym_len]

        bits = []

        for i in range(num_symbols):
            sym = rx_data[i * sym_len:(i + 1) * sym_len]
            sym = sym[self.cp_len:]

            Y = np.fft.rfft(sym, n=self.n_fft)
            Y_eq = Y / H

            vals = Y_eq[self.used_bins].real
            bits.extend((vals > 0).astype(np.uint8))

        bits = np.array(bits[:expected_num_bits], dtype=np.uint8)
        return bits

    # TX

    def make_wav_from_message(self, message_bytes=None, out_wav="gf3_tx.wav"):
        if message_bytes is None:
            message_bytes = bytes((i % 256 for i in range(10 * 1024)))

        packet = self.build_packet_bytes(message_bytes)
        bits = self._bytes_to_bits(packet)

        preamble = self.make_repeated_chirp_preamble()
        golay = self.make_golay_block()

        pilot_space_total = int(self.pilot_space_sec * self.fs)
        if len(golay) < pilot_space_total:
            golay_section = np.concatenate([golay, np.zeros(pilot_space_total - len(golay))])
        else:
            golay_section = golay

        data_wave, _ = self.bits_to_ofdm(bits)

        silence_start = np.zeros(int(0.5 * self.fs))
        silence_end = np.zeros(int(0.5 * self.fs))

        tx = np.concatenate([
            silence_start,
            preamble,
            golay_section,
            data_wave,
            silence_end,
        ])

        tx = self._normalise(tx, peak=self.tx_volume)

        wavfile.write(out_wav, self.fs, np.int16(tx * 32767))

        print(f"Wrote: {out_wav}")
        print(f"Payload bytes: {len(message_bytes)}")
        print(f"Total packet bytes: {len(packet)}")
        print(f"OFDM data bits: {len(bits)}")
        print(f"Used OFDM bins: {len(self.used_bins)}")
        print(f"Duration: {len(tx) / self.fs:.2f} seconds")

        return out_wav

    # RX

    def record_to_wav(self, seconds=20, out_wav="gf3_recording.wav"):
        if sd is None:
            raise ImportError("sounddevice is not installed. Run: pip install sounddevice")

        print(f"Recording for {seconds} seconds...")
        rx = sd.rec(int(seconds * self.fs), samplerate=self.fs, channels=1, dtype="float64")
        sd.wait()
        rx = rx[:, 0]

        wavfile.write(out_wav, self.fs, np.int16(self._normalise(rx, peak=0.95) * 32767))
        print(f"Wrote recording: {out_wav}")
        return out_wav

    def load_wav_mono(self, wav_path):
        fs_in, x = wavfile.read(wav_path)

        x = x.astype(np.float64)
        if x.ndim > 1:
            x = x[:, 0]

        if np.max(np.abs(x)) > 2:
            x = x / np.max(np.abs(x))

        if fs_in != self.fs:
            g = np.gcd(fs_in, self.fs)
            x = resample_poly(x, self.fs // g, fs_in // g)

        return x

    def decode_wav(self, wav_path, out_file="decoded_message.bin"):
        rx = self.load_wav_mono(wav_path)

        sync_start, confidence = self.find_sync_start(rx)
        preamble_len = len(self.make_repeated_chirp_preamble())

        golay_start = sync_start + preamble_len
        data_start = golay_start + int(self.pilot_space_sec * self.fs) + 200

        H = self.estimate_channel_from_golay(rx, golay_start)

        # First decode enough bits to read fixed prefix and header
        fixed_prefix_bits = 6 * 8
        first_bytes_guess = 64
        first_bits_guess = first_bytes_guess * 8

        rx_data = rx[data_start:]
        first_bits = self.ofdm_to_bits(rx_data, H, first_bits_guess)
        first_bytes = self._bits_to_bytes(first_bits)

        if len(first_bytes) < 6:
            raise ValueError("Could not decode fixed prefix.")

        header_size, payload_size = struct.unpack(">HI", first_bytes[:6])
        total_bytes = 6 + header_size + payload_size
        total_bits = total_bytes * 8

        all_bits = self.ofdm_to_bits(rx_data, H, total_bits)
        packet = self._bits_to_bytes(all_bits)

        payload = self.parse_packet_bytes(packet)

        with open(out_file, "wb") as f:
            f.write(payload)

        print(f"Sync start: {sync_start}")
        print(f"Sync confidence: {confidence:.2f}")
        print(f"Decoded payload bytes: {len(payload)}")
        print(f"Wrote decoded file: {out_file}")

        try:
            text = payload.decode("utf-8")
            print("\nDecoded text preview:")
            print(text[:1000])
        except UnicodeDecodeError:
            print("\nDecoded binary payload, not valid UTF-8.")

        return payload


if __name__ == "__main__":
    modem = GF3AudioModem()

    # TRANSMISSION RUN:
    # This creates a roughly 10 KB payload and writes a WAV to play from your speaker.
    #modem.make_wav_from_message(out_wav="gf3_tx_10kb.wav")

    # RECEIVING RUN:
    # Uncomment these when you want to record and decode in a separate run.
    #
    modem.record_to_wav(seconds=25, out_wav="gf3_recording.wav")
    modem.decode_wav("gf3_recording.wav", out_file="decoded_message.bin")

    # If you already recorded manually:
    #
    #modem.decode_wav("your_recording.wav", out_file="decoded_message.bin")
