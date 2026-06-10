import os
import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve
from scipy.io import wavfile

from Audio_Modem.Appendix_A import APPENDIX_A_DATA
from Audio_Modem.Modulator import Modulator
from Audio_Modem.Utilities import bits_to_bytes, Header
from Audio_Modem.LDPC_Wrapper import LdpcWrapper, StandardInterleaver
from Audio_Modem.Transmitter import Chirp, Golay, OFDM
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey


class AudioReceiver:
    def __init__(self, eq_noise: float = 0.5):
        # Mirror the transmitter's DSP configuration
        self.chirp = Chirp()
        self.golay = Golay()
        self.ofdm = OFDM()
        self.modulator = Modulator()
        self.ldpc = LdpcWrapper(z=61)
        self.interleaver = StandardInterleaver()
        self.pilot_qpsk = self.modulator.to_qpsk(APPENDIX_A_DATA)

        self.symbol_len = self.ofdm.subcarriers + self.ofdm.prefix_length
        self.eq_noise = eq_noise  # MMSE Noise floor parameter

        self.chirps = self.chirp.make_chirp()
        self.golay_pairs = self.golay.get_pilot_symbol()
        self.ref_chirp_len = len(self.chirps)
        self.preamble_len = self.ref_chirp_len + len(self.golay_pairs)

    def _load_normalise(self, wav_path: str):
        """Read a WAV file and normalise it to [-1, 1]."""
        print(f"Reading received signal from: {wav_path}")
        fs, rx_signal = wavfile.read(wav_path)
        rx_signal = rx_signal.astype(np.float64)
        rx_signal = rx_signal / (np.max(np.abs(rx_signal)) + 1e-12)

        return fs, rx_signal

    def _synchronize(self, rx_signal: npt.NDArray[np.float64]) -> int:
        """Finds the start of the transmission frame using fftconvolve on the reversed preamble."""
        corr = fftconvolve(rx_signal, self.chirps[::-1], mode="valid")
        start_idx = int(np.argmax(np.abs(corr)))
        return start_idx

    def _estimate_channel_golay(self, rx_signal, sync_start):
        a, b = self.golay.generate_golay_pairs()
        N = len(a)
        golay_a_start = sync_start + self.ref_chirp_len + self.golay.cp_length
        golay_b_start = golay_a_start + self.golay.cp_length + N

        y_a = rx_signal[golay_a_start : golay_a_start + N]
        y_b = rx_signal[golay_b_start : golay_b_start + N]

        # cross-correlation (time domain)
        h = np.correlate(y_a, a, mode="same") + np.correlate(y_b, b, mode="same")
        h = h / (2 * self.ofdm.subcarriers)
        return np.fft.rfft(h, n=self.ofdm.subcarriers) / (4096 * 8)

    def _remove_cp_and_fft(
        self, sym_time: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        """Strip the cyclic prefix from one OFDM symbol and return its FFT."""
        sym_no_cp = sym_time[self.ofdm.prefix_length :]
        return np.fft.rfft(sym_no_cp)

    def _equalize_symbol(
        self,
        R: npt.NDArray[np.complex128],
        H_est: npt.NDArray[np.complex128],
        noise_floor: float,
    ) -> npt.NDArray[np.complex128]:
        """
        Apply MMSE equalization to one OFDM symbol.

        Extracts the 854 strict data bins, normalises the received spectrum,
        then applies:  X_est = (R * H*) / (|H|^2 + noise_floor)
        """
        R_data = R[self.ofdm.data_bins[:854]]
        H_data = H_est[self.ofdm.data_bins[:854]]

        R_data = R_data / np.mean(np.abs(R_data))
        X_est = (R_data * np.conj(H_data)) / (np.abs(H_data) ** 2 + noise_floor)
        return X_est

    def _process_ofdm_symbols(
        self,
        rx_data: npt.NDArray[np.float64],
        H_est: npt.NDArray[np.complex128],
        noise_floor: float,
    ) -> list[npt.NDArray[np.complex128]]:
        """
        Walk through all OFDM symbols in rx_data, skipping pilot positions,
        and return the full ordered list of equalized QPSK symbol vectors.

        Each entry in the returned list is one symbol's worth of 854 equalized
        QPSK bins, ready for deinterleaving.
        """
        block_counter = 1
        sample_idx = 0
        all_equalized = []

        while sample_idx + self.symbol_len <= len(rx_data):
            sym_time = rx_data[sample_idx : sample_idx + self.symbol_len]
            sample_idx += self.symbol_len

            if block_counter % 20 == 0:
                # Pilot symbol — channel tracking can be inserted here
                pass
            else:
                R = self._remove_cp_and_fft(sym_time)
                X_est = self._equalize_symbol(R, H_est, noise_floor)
                all_equalized.append(X_est)

            block_counter += 1

        return all_equalized

    def _decode_ldpc_group(
        self, qpsk_buffer: list[npt.NDArray[np.complex128]]
    ) -> list[int]:
        """
        Deinterleave and LDPC-decode one group of 30 OFDM symbols (35 LDPC blocks).

        Steps:
          1. Stack the 30 equalized symbol rows into a 2-D array (30 x 854).
          2. Build uniform weight matrix (ones) — no per-bin SNR weighting yet.
          3. Call deinterleave_llrs: reverses the stride permutation and
             computes LLRs, producing a (35 x 1464) array.
          4. Feed each of the 35 LLR vectors to the belief-propagation decoder.
          5. Return all decoded information bits as a flat list.
        """
        qpsk_2d = np.array(qpsk_buffer)  # (30, 854)
        weights = np.ones_like(qpsk_2d, dtype=np.float64)  # uniform SNR weights

        llrs_2d = self.interleaver.deinterleave_llrs(qpsk_2d, weights)  # (35, 1464)

        bits = []
        for llr_block in llrs_2d:
            block_bits, iters, success = self.ldpc.decode_block(llr_block)
            bits.extend(block_bits)
        return bits

    def _decode_all_groups(
        self, equalized_symbols: list[npt.NDArray[np.complex128]]
    ) -> npt.NDArray[np.uint8]:
        """
        Split the full list of equalized symbols into groups of 30 and decode
        each group, accumulating all decoded bits into one array.
        """
        decoded_bits = []
        num_complete_groups = len(equalized_symbols) // 30

        print(
            f"Decoding {num_complete_groups} LDPC groups "
            f"({num_complete_groups * 30} symbols)..."
        )

        for i in range(num_complete_groups):
            group = equalized_symbols[i * 30 : (i + 1) * 30]
            group_bits = self._decode_ldpc_group(group)
            decoded_bits.extend(group_bits)

        return np.array(decoded_bits, dtype=np.uint8)

    def _try_extract_payload(
        self, bits: npt.NDArray[np.uint8], output_dir: str
    ) -> bool:
        # Minimum bits needed to evaluate A and B
        if len(bits) < 48:
            return False

        try:
            header_length, data_length, filename = Header.decode_header(bits)
        except ValueError:
            return False

        total_bits_needed = (header_length + data_length) * 8

        if len(bits) < total_bits_needed:
            return False

        print(
            f"Header parsed. Target File: {filename} | Data Length: {data_length} bytes."
        )
        payload_bits = bits[header_length * 8 : total_bits_needed]
        payload_bytes = bits_to_bytes(payload_bits)

        output_path = os.path.join(output_dir, f"rx_{filename}")

        with open(output_path, "wb") as f:
            f.write(payload_bytes)

        print(f"Success! Payload extracted and written to: {output_path}")
        return True

    def process_file(self, file_path: str, output_dir: str = ".") -> None:
        fs, rx_signal = wavfile.read(file_path)

        start_index = self._synchronize(rx_signal)
        print(f"Synchronization successful. Frame located at index: {start_index}")

        H_est = self._estimate_channel_golay(rx_signal, start_index)
        # self.debug_channel_estimate(H_est)

        noise_floor = self.eq_noise * np.mean(np.abs(H_est) ** 2)

        data_start_idx = start_index + self.preamble_len
        rx_data = rx_signal[data_start_idx:]
        print("Equalizing OFDM symbols...")
        equalized_symbols = self._process_ofdm_symbols(rx_data, H_est, noise_floor)

        decoded_bits = self._decode_all_groups(equalized_symbols)

        success = self._try_extract_payload(decoded_bits, output_dir)
        if not success:
            print(
                "Error: Reached the end of the waveform without successfully "
                "decoding the payload."
            )
        return success

    def debug_channel_estimate(self, H_est):
        """
        Plots the magnitude and phase of the frequency domain channel estimate.
        For an ideal channel, magnitude should be constant and phase should be linear.
        """
        freqs = np.fft.rfftfreq(self.ofdm.subcarriers, d=1 / self.ofdm.sampling_freq)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(freqs, np.abs(H_est))
        plt.title("H_est Magnitude")
        plt.xlabel("Frequency (Hz)")

        plt.subplot(1, 2, 2)
        plt.plot(freqs, np.angle(H_est))
        plt.title("H_est Phase")
        plt.xlabel("Frequency (Hz)")

        plt.show()

    def debug_constellation(
        self,
        wav_path: str,
        group_idx: int = 0,
        symbol_idx: int = 0,
    ) -> None:
        """
        Plot the QPSK constellation for one equalized symbol.

        Use this when LDPC blocks are not converging.  A healthy signal shows
        four tight clusters at (±1, ±j).  Common failure patterns:
          - Uniform cloud     → sync offset wrong, CP removal landing mid-symbol
          - 4 clusters at 45° → QPSK phase reference mismatch (constellation rotation)
          - 2 clusters only   → BPSK being decoded as QPSK, or I/Q imbalance
          - Spread clusters   → low SNR or H_est scale error causing poor equalization
        """
        fs, rx_signal = self._load_normalise(wav_path)
        start_idx = self._synchronize(rx_signal)
        H_est = self._estimate_channel_golay(rx_signal, start_idx)
        noise_floor = self.eq_noise * np.mean(np.abs(H_est) ** 2)

        rx_data = rx_signal[start_idx + self.preamble_len :]
        equalized = self._process_ofdm_symbols(rx_data, H_est, noise_floor)

        flat_idx = group_idx * 30 + symbol_idx
        if flat_idx >= len(equalized):
            print(
                f"Only {len(equalized)} equalized symbols available; "
                f"requested index {flat_idx}."
            )
            return

        X = equalized[flat_idx]
        plt.figure(figsize=(5, 5))
        plt.scatter(X.real, X.imag, s=2, alpha=0.6)
        plt.axhline(0, color="k", linewidth=0.5)
        plt.axvline(0, color="k", linewidth=0.5)
        plt.title(f"Constellation — group {group_idx}, symbol {symbol_idx}")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    receiver = AudioReceiver()

    target_audio = "rx_recorded_louis3.wav"
    # target_audio = "tx_signal_802.wav"
    # target_audio = "tx.wav"
    if os.path.exists(target_audio):
        # receiver.process_file(target_audio)
        # receiver.demodulate_file(target_audio)
        receiver.debug_constellation(target_audio, group_idx=0, symbol_idx=0)
    else:
        print(f"Could not locate '{target_audio}' in the current directory.")
