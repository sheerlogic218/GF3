from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from Audio_Modem.Appendix_A import APPENDIX_A_DATA
from Audio_Modem.Modulator import Modulator
from Audio_Modem.Utilities import bytes_to_bits, bits_to_bytes, Header
from Audio_Modem.LDPC_Wrapper import LdpcWrapper, StandardInterleaver


class Chirp:
    def __init__(
        self,
        duration: float = 4096 / 48000,
        repeats: int = 10,
        f0: int = 750,
        f1: int = 18000,
        amplitude: float = 0.8,
    ):
        self.duration = duration
        self.repeats = repeats
        self.f0 = f0
        self.f1 = f1
        self.amplitude = amplitude
        self.sampling_freq = 48000

    def make_chirp(self) -> npt.NDArray[np.float64]:
        t = np.linspace(
            0,
            self.duration,
            int(self.duration * self.sampling_freq),
            endpoint=False,
        )
        k = (self.f1 - self.f0) / self.duration
        phase = 2 * np.pi * (self.f0 * t + 0.5 * k * t**2)
        chirp_signal = self.amplitude * np.sin(phase)
        return np.tile(chirp_signal, self.repeats)


class Golay:
    def __init__(self, order: int = 12, cp_length: int = 2048, amplitude: float = 0.8):
        self.order = order
        self.cp_length = cp_length
        self.amplitude = amplitude

    def get_pilot_symbol(self) -> npt.NDArray[np.float64]:
        a = np.array([1.0])
        b = np.array([1.0])

        for _ in range(self.order):
            a_next = np.concatenate((a, b))
            b_next = np.concatenate((a, -b))
            a, b = a_next, b_next

        # A + gap + B
        gap = np.zeros(self.cp_length, dtype=np.float64)
        return self.amplitude * np.concatenate((a, gap, b))


class OFDM:
    """Generates the Clause 4.0 OFDM Signal"""

    def __init__(
        self,
        sampling_freq: int = 48000,
        subcarriers: int = 4096,
        prefix_length: int = 2048,
        min_freq: int = 2000,
        max_freq: int = 12000,
    ):
        self.sampling_freq = sampling_freq
        self.subcarriers = subcarriers
        self.prefix_length = prefix_length
        self.min_freq = min_freq
        self.max_freq = max_freq

        # Calculate strict bins 171->1024
        freqs = np.fft.rfftfreq(self.subcarriers, d=1 / self.sampling_freq)
        self.data_bins = np.where((freqs > self.min_freq) & (freqs <= self.max_freq))[0]

    def to_OFDM_symbol(
        self, qpsk_symbols: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Maps 854 active symbols to the strict data bins."""
        X = np.zeros(self.subcarriers // 2 + 1, dtype=complex)
        X[self.data_bins[: len(qpsk_symbols)]] = qpsk_symbols
        x = np.fft.irfft(X)
        return np.concatenate([x[-self.prefix_length :], x])

    def to_OFDM_pilot_symbol(
        self, qpsk_symbols: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        """Maps exactly 2048 Pilot symbols starting from bin 1 (Appendix A style)."""
        X = np.zeros(self.subcarriers // 2 + 1, dtype=complex)
        X[1 : len(qpsk_symbols) + 1] = qpsk_symbols
        x = np.fft.irfft(X)
        return np.concatenate([x[-self.prefix_length :], x])


class AudioTransmitter:
    def __init__(self):
        # Instantiate DSP components
        self.chirp = Chirp()
        self.golay = Golay()
        self.ofdm = OFDM()
        self.modulator = Modulator()
        self.ldpc = LdpcWrapper(z=61)
        self.interleaver = StandardInterleaver()
        self.pilot_qpsk = self.modulator.to_qpsk(APPENDIX_A_DATA)

    def build_frame(
        self, payload_bytes: bytes, filename: str
    ) -> npt.NDArray[np.float64]:
        print(f"Building Transmission Frame for: {filename}")

        # Generate Preamble
        chirp_waveform = self.chirp.make_chirp()
        golay_waveform = self.golay.get_pilot_symbol()

        preamble_block = np.concatenate(
            [
                chirp_waveform,
                golay_waveform,
            ]
        )

        # Generate Header and Merge Data Bits
        header_bits = Header.form_header(payload_bytes, filename)
        payload_bits = bytes_to_bits(payload_bytes)
        full_data_bits = np.concatenate([header_bits, payload_bits])

        # LDPC
        coded_blocks = self.ldpc.encode_blocks(full_data_bits)

        # Process Groups & Map OFDM Symbols
        num_groups = len(coded_blocks) // 35
        data_waveform = []
        block_counter = 1

        for i in range(num_groups):
            chunk = coded_blocks[i * 35 : (i + 1) * 35]
            qpsk_symbols_2d = self.interleaver.interleave(chunk)

            for qpsk_row in qpsk_symbols_2d:
                if block_counter % 20 == 0:
                    pilot_time = self.ofdm.to_OFDM_pilot_symbol(self.pilot_qpsk)
                    data_waveform.append(pilot_time)
                    block_counter += 1

                # Modulate Data Symbol
                data_time = self.ofdm.to_OFDM_symbol(qpsk_row)
                data_waveform.append(data_time)
                block_counter += 1

        silence = np.zeros(4800)
        final_signal = np.concatenate(
            [silence, preamble_block, *data_waveform, silence]
        )

        # volume normalisation
        final_signal = final_signal / (np.max(np.abs(final_signal)) + 1e-12) * 0.92
        return final_signal


if __name__ == "__main__":
    import os

    # Example Usage
    transmitter = AudioTransmitter()

    test_file = "payload.txt"
    out_file = "tx.wav"

    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write(
                "CUGFKGKGAIGSDUBSNDHAWEUDAKSJUDYTGVBNJKIUYTGHBNMJKIUYHGTBNJMKIUYHGTBNMJKIUYHGBNJuyhtgvbnjuyTGBNJUYHGtbnjmkiuyhgBNJMKUYHGb"
                * 20
            )

    with open(test_file, "rb") as f:
        file_data = f.read()

    signal = transmitter.build_frame(file_data, test_file)

    # Save Output
    wavfile.write(out_file, 48000, (signal * 32767).astype(np.int16))
    print(f"Success! Output saved to {out_file} ({len(signal) / 48000:.2f} seconds)")
