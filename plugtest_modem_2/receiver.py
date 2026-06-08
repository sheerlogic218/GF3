from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve, find_peaks

from modem import (
    CONFIG, DecodedPacket, GolayPilot, KnownOFDMPilot, LinearChirp, ModemConfig,
    OFDM, PacketCodec, PilotScheduler, QPSK, StandardInterleaver, WiMaxLDPC,
    bits_to_bytes, read_wav, safe_output_name,
)


EPS = 1e-12


@dataclass(frozen=True)
class ReceptionInfo:
    sync_sample: int
    sample_rate_offset_ppm: float
    received_ofdm_blocks: int
    data_ofdm_blocks: int
    decoded_ldpc_groups: int
    successful_ldpc_blocks: int
    total_ldpc_blocks: int


class Receiver:
    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        self.ldpc = WiMaxLDPC(config)
        self.interleaver = StandardInterleaver(config)
        self.ofdm = OFDM(config)
        self.scheduler = PilotScheduler(config.pilot_interval)
        self.chirp = LinearChirp(config)
        self.golay = GolayPilot(config)
        self.known_pilot = KnownOFDMPilot(config)

    def synchronise_and_correct(
        self, received: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], int, float]:
        """Find the ten chirps and remove the measured sampling-rate offset.

        The spacing of the chirp correlation peaks estimates received samples per
        nominal 4096-sample chirp. The signal is then interpolated onto the
        transmitter's nominal sample grid from the first chirp onwards.
        """
        received = np.asarray(received, dtype=float)
        chirp = self.chirp.samples()
        n = self.config.chirp_length
        if len(received) < self.config.chirp_train_length:
            raise ValueError("recording is shorter than the JOSS-F chirp train")

        metric = np.abs(fftconvolve(received, chirp[::-1], mode="valid"))
        peaks, _ = find_peaks(
            metric,
            distance=int(0.75 * n),
            prominence=max(float(np.max(metric)) * 0.05, EPS),
        )

        best_chain: list[int] | None = None
        best_score = -np.inf
        tolerance = int(0.08 * n)
        for first in peaks:
            chain = [int(first)]
            score = float(metric[first])
            current = int(first)
            for _ in range(1, self.config.chirp_count):
                target = current + n
                candidates = peaks[np.abs(peaks - target) <= tolerance]
                if not len(candidates):
                    break
                chosen = int(candidates[np.argmax(metric[candidates])])
                chain.append(chosen)
                score += float(metric[chosen])
                current = chosen
            if len(chain) == self.config.chirp_count:
                spacing = np.diff(chain)
                regularity = 1.0 + float(np.std(spacing))
                score /= regularity
                if score > best_score:
                    best_score = score
                    best_chain = chain

        if best_chain is None:
            # Conservative fallback for very poor recordings.
            template = self.chirp.train()
            coarse_metric = fftconvolve(received, template[::-1], mode="valid")
            sync = int(np.argmax(np.abs(coarse_metric)))
            return received[sync:], sync, 0.0

        chirp_number = np.arange(self.config.chirp_count, dtype=float)
        observed_values: list[float] = []
        for peak in best_chain:
            if 0 < peak < len(metric) - 1:
                left, centre, right = metric[peak - 1], metric[peak], metric[peak + 1]
                denominator = left - 2.0 * centre + right
                delta = 0.0 if abs(denominator) < EPS else 0.5 * (left - right) / denominator
                observed_values.append(float(peak) + float(np.clip(delta, -0.5, 0.5)))
            else:
                observed_values.append(float(peak))
        observed = np.asarray(observed_values, dtype=float)
        period, intercept = np.polyfit(chirp_number, observed, 1)
        ratio = float(period / n)
        if not 0.995 <= ratio <= 1.005:
            raise ValueError(f"implausible sampling-rate ratio estimated: {ratio:.7f}")
        sync = int(round(intercept))

        # Keep the fitted fractional start rather than rounding it before
        # resampling; a half-sample timing error produces a measurable phase
        # slope across the wide 2--12 kHz data band.
        output_length = int(np.floor((len(received) - intercept - 1) / ratio)) + 1
        source_positions = intercept + np.arange(output_length, dtype=float) * ratio
        corrected = np.interp(source_positions, np.arange(len(received), dtype=float), received)
        offset_ppm = (ratio - 1.0) * 1e6
        return corrected, sync, offset_ppm

    def _golay_channel(self, received: NDArray[np.float64], sync_start: int) -> NDArray[np.complex128]:
        """Regularised LS estimate using both known Golay sequences.

        For the A-gap-B layout, A has a clean 2048-sample tail. B is padded at
        the receiver to avoid using the following data samples in the estimate.
        """
        c = self.config
        a_start = sync_start + c.chirp_train_length
        b_start = a_start + c.fft_length + c.cyclic_prefix
        transform_length = c.fft_length + c.cyclic_prefix

        y_a = received[a_start : a_start + transform_length]
        if len(y_a) < transform_length:
            y_a = np.pad(y_a, (0, transform_length - len(y_a)))
        y_b_useful = received[b_start : b_start + c.fft_length]
        if len(y_b_useful) < c.fft_length:
            y_b_useful = np.pad(y_b_useful, (0, c.fft_length - len(y_b_useful)))
        y_b = np.pad(y_b_useful, (0, c.cyclic_prefix))

        x_a = np.pad(c.chirp_amplitude * self.golay.a, (0, c.cyclic_prefix))
        x_b = np.pad(c.chirp_amplitude * self.golay.b, (0, c.cyclic_prefix))
        x_a_f = np.fft.rfft(x_a, n=transform_length)
        x_b_f = np.fft.rfft(x_b, n=transform_length)
        numerator = np.fft.rfft(y_a, n=transform_length) * np.conj(x_a_f)
        numerator += np.fft.rfft(y_b, n=transform_length) * np.conj(x_b_f)
        denominator = np.abs(x_a_f) ** 2 + np.abs(x_b_f) ** 2
        regularisation = c.golay_regularisation * float(np.max(denominator))
        h = np.fft.irfft(numerator / (denominator + regularisation), n=transform_length)
        h = h[: c.cyclic_prefix]
        return np.fft.rfft(h, n=c.fft_length)

    def _equalise(
        self,
        spectrum: NDArray[np.complex128],
        channel: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        bins = self.config.data_bins
        reference = float(np.median(np.abs(channel[bins]) ** 2)) + EPS
        regularisation = self.config.equaliser_regularisation * reference
        estimate = spectrum[bins] * np.conj(channel[bins]) / (
            np.abs(channel[bins]) ** 2 + regularisation
        )

        # Decision-directed affine phase correction. A sampling-frequency
        # mismatch appears approximately as a phase slope across subcarriers,
        # while oscillator/timing error contributes the intercept.
        hard = QPSK.hard_symbols(estimate)
        phase_error = np.unwrap(np.angle(estimate * np.conj(hard)))
        x = bins.astype(float)
        x -= float(np.mean(x))
        fit_weights = np.clip(np.abs(estimate), 0.25, 3.0)
        design = np.column_stack([x, np.ones_like(x)])
        weighted_design = design * fit_weights[:, None]
        weighted_phase = phase_error * fit_weights
        slope, intercept = np.linalg.lstsq(weighted_design, weighted_phase, rcond=None)[0]
        estimate *= np.exp(-1j * (slope * x + intercept))
        return estimate

    def _update_channel(
        self,
        spectrum: NDArray[np.complex128],
        previous: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        known = self.known_pilot.transmitted_frequency
        updated = previous.copy()
        bins = self.config.data_bins
        valid = np.abs(known[bins]) > EPS
        fresh = spectrum[bins][valid] / known[bins][valid]
        alpha = self.config.pilot_update_weight
        current = updated[bins]
        current[valid] = (1.0 - alpha) * current[valid] + alpha * fresh
        updated[bins] = current
        return updated

    def _extract_data_rows(
        self,
        received: NDArray[np.float64],
        data_start: int,
        initial_channel: NDArray[np.complex128],
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], int]:
        c = self.config
        symbol_length = c.ofdm_symbol_length
        available = max(0, len(received) - data_start)
        block_count = available // symbol_length
        rows: list[NDArray[np.complex128]] = []
        weights: list[NDArray[np.float64]] = []
        channel = initial_channel.copy()

        for zero_based in range(block_count):
            one_based = zero_based + 1
            start = data_start + zero_based * symbol_length
            block = received[start : start + symbol_length]
            spectrum = self.ofdm.spectrum(block)
            if self.scheduler.is_pilot(one_based):
                channel = self._update_channel(spectrum, channel)
                continue

            estimate = self._equalise(spectrum, channel)
            bins = c.data_bins
            reliability = np.abs(channel[bins]) ** 2
            reliability /= float(np.median(reliability) + EPS)
            reliability = np.clip(reliability, 0.15, 8.0)
            # Keep the median LLR magnitude in a stable numerical range.
            hard = QPSK.hard_symbols(estimate)
            residual = float(np.median(np.abs(estimate - hard) ** 2))
            common = 2.0 / max(residual, 0.08)
            rows.append(estimate)
            weights.append(common * reliability)

        if not rows:
            raise ValueError("no JOSS-F data OFDM blocks were found")
        return np.stack(rows), np.stack(weights), block_count

    def decode_signal(self, received: NDArray[np.float64]) -> tuple[DecodedPacket, ReceptionInfo]:
        received = np.asarray(received, dtype=float)
        corrected, raw_sync_start, offset_ppm = self.synchronise_and_correct(received)
        sync_start = 0
        channel = self._golay_channel(corrected, sync_start)
        data_start = self.config.preamble_length
        rows, weights, received_blocks = self._extract_data_rows(corrected, data_start, channel)

        complete_groups = len(rows) // self.config.data_symbols_per_group
        if complete_groups == 0:
            raise ValueError("fewer than 30 data OFDM symbols were recovered")

        recovered_bits: list[NDArray[np.uint8]] = []
        successful_blocks = 0
        total_blocks = 0
        decoded_groups = 0
        requested_bytes: int | None = None

        for group_index in range(complete_groups):
            lo = group_index * self.config.data_symbols_per_group
            hi = lo + self.config.data_symbols_per_group
            llr_blocks = self.interleaver.deinterleave_llrs(rows[lo:hi], weights[lo:hi])
            information_blocks: list[NDArray[np.uint8]] = []
            for llr in llr_blocks:
                information, _, success = self.ldpc.decode_block(llr)
                information_blocks.append(information)
                successful_blocks += int(success)
                total_blocks += 1
            recovered_bits.append(np.concatenate(information_blocks))
            decoded_groups += 1

            decoded_bytes = bits_to_bytes(np.concatenate(recovered_bits))
            if requested_bytes is None and len(decoded_bytes) >= 6:
                requested_bytes = PacketCodec.expected_total_bytes(decoded_bytes)
            if requested_bytes is not None and len(decoded_bytes) >= requested_bytes:
                packet = PacketCodec.parse(decoded_bytes[:requested_bytes])
                info = ReceptionInfo(
                    sync_sample=raw_sync_start,
                    sample_rate_offset_ppm=offset_ppm,
                    received_ofdm_blocks=received_blocks,
                    data_ofdm_blocks=len(rows),
                    decoded_ldpc_groups=decoded_groups,
                    successful_ldpc_blocks=successful_blocks,
                    total_ldpc_blocks=total_blocks,
                )
                return packet, info

        decoded_bytes = bits_to_bytes(np.concatenate(recovered_bits))
        packet = PacketCodec.parse(decoded_bytes)
        info = ReceptionInfo(
            sync_sample=raw_sync_start,
            sample_rate_offset_ppm=offset_ppm,
            received_ofdm_blocks=received_blocks,
            data_ofdm_blocks=len(rows),
            decoded_ldpc_groups=decoded_groups,
            successful_ldpc_blocks=successful_blocks,
            total_ldpc_blocks=total_blocks,
        )
        return packet, info

    def decode_wav(self, wav_path: str | Path, output_directory: str | Path = ".") -> Path:
        received = read_wav(wav_path, self.config)
        packet, info = self.decode_signal(received)
        output_path = Path(output_directory) / safe_output_name(packet.filename)
        output_path.write_bytes(packet.payload)
        print(f"Decoded {wav_path}")
        print(f"  sync sample      : {info.sync_sample}")
        print(f"  sample offset    : {info.sample_rate_offset_ppm:+.1f} ppm")
        print(f"  original name    : {packet.filename}")
        print(f"  file size        : {packet.file_size} bytes")
        print(f"  LDPC success     : {info.successful_ldpc_blocks}/{info.total_ldpc_blocks}")
        print(f"  saved            : {output_path}")
        return output_path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = BASE_DIR


def ask_input_wav() -> Path:
    """Ask for a recording name, accepting either ``recorded`` or ``recorded.wav``."""
    name = input("WAV file to decode [recorded]: ").strip() or "recorded"
    path = Path(name).expanduser()
    if path.suffix.lower() != ".wav":
        path = path.with_suffix(".wav")
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        available = ", ".join(sorted(item.name for item in BASE_DIR.glob("*.wav")))
        detail = f" Available WAV files: {available}" if available else ""
        raise FileNotFoundError(f"WAV file not found: {path}.{detail}")
    return path


if __name__ == "__main__":
    Receiver().decode_wav(ask_input_wav(), OUTPUT_DIRECTORY)
