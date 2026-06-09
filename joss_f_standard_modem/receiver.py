from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve, find_peaks

from modem import (
    CONFIG,
    DecodedPacket,
    GolayPilot,
    KnownOFDMPilot,
    LinearChirp,
    ModemConfig,
    OFDM,
    PacketCodec,
    PilotScheduler,
    QPSK,
    StandardInterleaver,
    WiMaxLDPC,
    bits_to_bytes,
    read_wav,
    safe_output_name,
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
    data_start_adjustment: int


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
        """Find the ten chirps and correct transmitter/receiver clock mismatch."""
        received = np.asarray(received, dtype=float)
        chirp = self.chirp.samples()
        n = self.config.chirp_length
        if len(received) < self.config.chirp_train_length:
            raise ValueError("recording is shorter than the JOSS-F chirp train")

        metric = np.abs(fftconvolve(received, chirp[::-1], mode="valid"))
        peaks, _ = find_peaks(
            metric,
            distance=int(0.75 * n),
            prominence=max(float(np.max(metric)) * 0.035, EPS),
        )

        # scipy.signal.find_peaks does not return index zero, but compliant WAV
        # files may begin immediately with the first chirp.
        candidates = np.unique(
            np.concatenate([
                peaks.astype(int),
                np.array([0, int(np.argmax(metric))], dtype=int),
            ])
        )

        best_chain: list[int] | None = None
        best_score = -np.inf
        tolerance = int(0.08 * n)
        for first in candidates:
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
                score /= 1.0 + float(np.std(spacing))
                if score > best_score:
                    best_score = score
                    best_chain = chain

        if best_chain is None:
            template = self.chirp.train()
            coarse_metric = np.abs(fftconvolve(received, template[::-1], mode="valid"))
            sync = int(np.argmax(coarse_metric))
            return received[sync:], sync, 0.0

        observed_values: list[float] = []
        for peak in best_chain:
            if 0 < peak < len(metric) - 1:
                left, centre, right = metric[peak - 1], metric[peak], metric[peak + 1]
                denominator = left - 2.0 * centre + right
                delta = 0.0 if abs(denominator) < EPS else 0.5 * (left - right) / denominator
                observed_values.append(float(peak) + float(np.clip(delta, -0.5, 0.5)))
            else:
                observed_values.append(float(peak))

        chirp_number = np.arange(self.config.chirp_count, dtype=float)
        period, intercept = np.polyfit(chirp_number, np.asarray(observed_values), 1)
        ratio = float(period / n)
        if not 0.995 <= ratio <= 1.005:
            raise ValueError(f"implausible sampling-rate ratio estimated: {ratio:.7f}")

        sync = int(round(intercept))
        output_length = int(np.floor((len(received) - intercept - 1) / ratio)) + 1
        source_positions = intercept + np.arange(output_length, dtype=float) * ratio

        # Cubic interpolation preserves the upper data subcarriers substantially
        # better than linear interpolation when fractional delay accumulates.
        sample_axis = np.arange(len(received), dtype=float)
        interpolator = CubicSpline(sample_axis, received, extrapolate=False)
        corrected = np.asarray(interpolator(source_positions), dtype=float)
        corrected = np.nan_to_num(corrected)
        offset_ppm = (ratio - 1.0) * 1e6
        return corrected, sync, offset_ppm

    def _golay_channel(self, received: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Estimate the channel from all four official Golay A/B repeats.

        Each pulse is followed by a 2048-sample zero gap, so the pulse plus gap
        can be used as an uncontaminated channel-estimation window. Estimates
        from all eight pulses are combined by regularised least squares.
        """
        c = self.config
        transform_length = c.fft_length + c.golay_gap
        golay_start = c.chirp_train_length

        x_a = np.pad(c.golay_amplitude * self.golay.a, (0, c.golay_gap))
        x_b = np.pad(c.golay_amplitude * self.golay.b, (0, c.golay_gap))
        x_a_f = np.fft.rfft(x_a, n=transform_length)
        x_b_f = np.fft.rfft(x_b, n=transform_length)

        numerator = np.zeros(transform_length // 2 + 1, dtype=np.complex128)
        denominator = np.zeros(transform_length // 2 + 1, dtype=float)

        for a_relative, b_relative in self.golay.pulse_starts:
            for relative, reference_f in (
                (a_relative, x_a_f),
                (b_relative, x_b_f),
            ):
                start = golay_start + relative
                window = received[start : start + transform_length]
                if len(window) < transform_length:
                    window = np.pad(window, (0, transform_length - len(window)))
                received_f = np.fft.rfft(window, n=transform_length)
                numerator += received_f * np.conj(reference_f)
                denominator += np.abs(reference_f) ** 2

        regularisation = c.golay_regularisation * float(np.max(denominator))
        h = np.fft.irfft(
            numerator / (denominator + regularisation),
            n=transform_length,
        )
        h = h[: c.cyclic_prefix]
        return np.fft.rfft(h, n=c.fft_length)

    def _cp_score(self, received: NDArray[np.float64], start: int) -> float:
        c = self.config
        length = c.ofdm_symbol_length
        available = (len(received) - start) // length
        count = min(max(available, 0), 8)
        if count <= 0:
            return -np.inf
        scores: list[float] = []
        for index in range(count):
            lo = start + index * length
            block = received[lo : lo + length]
            prefix = block[: c.cyclic_prefix]
            tail = block[c.fft_length : c.fft_length + c.cyclic_prefix]
            denominator = np.linalg.norm(prefix) * np.linalg.norm(tail) + EPS
            scores.append(float(abs(np.vdot(prefix, tail)) / denominator))
        return float(np.median(scores))

    def _refine_data_start(self, received: NDArray[np.float64], nominal: int) -> tuple[int, int]:
        """Refine the first OFDM boundary using cyclic-prefix self-correlation."""
        radius = self.config.data_start_refine_radius
        coarse_offsets = range(-radius, radius + 1, 4)
        coarse = max(coarse_offsets, key=lambda offset: self._cp_score(received, nominal + offset))
        fine_offsets = range(max(-radius, coarse - 5), min(radius, coarse + 5) + 1)
        best = max(fine_offsets, key=lambda offset: self._cp_score(received, nominal + offset))
        return nominal + best, int(best)

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

        # Remove residual phase slope and common phase. This absorbs small timing
        # errors and residual sampling-frequency offset without changing the TX.
        hard = QPSK.hard_symbols(estimate)
        phase_error = np.unwrap(np.angle(estimate * np.conj(hard)))
        x = bins.astype(float)
        x -= float(np.mean(x))
        fit_weights = np.clip(np.abs(estimate), 0.10, 3.0)
        design = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(
            design * fit_weights[:, None], phase_error * fit_weights, rcond=None
        )[0]
        estimate *= np.exp(-1j * (slope * x + intercept))

        # AGC is predominantly a block-wise scalar gain. Normalising each QPSK
        # cloud makes decoding insensitive to slow gain recovery after the loud
        # chirp/Golay section while preserving all symbol signs and phases.
        robust_magnitude = float(np.median(np.abs(estimate)))
        if robust_magnitude > EPS:
            estimate *= np.sqrt(2.0) / robust_magnitude
        return estimate

    def _pilot_channel(
        self,
        spectrum: NDArray[np.complex128],
        fallback: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        known = self.known_pilot.transmitted_frequency
        bins = self.config.data_bins
        updated = fallback.copy()
        valid = np.abs(known[bins]) > EPS
        updated[bins[valid]] = spectrum[bins[valid]] / known[bins[valid]]
        return updated

    def _extract_data_rows(
        self,
        received: NDArray[np.float64],
        data_start: int,
        golay_channel: NDArray[np.complex128],
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], int]:
        c = self.config
        symbol_length = c.ofdm_symbol_length
        available = max(0, len(received) - data_start)
        block_count = available // symbol_length
        if block_count == 0:
            raise ValueError("no complete JOSS-F OFDM blocks were found")

        spectra = [
            self.ofdm.spectrum(
                received[
                    data_start + index * symbol_length : data_start + (index + 1) * symbol_length
                ]
            )
            for index in range(block_count)
        ]

        # This is an offline modem, so a later standard pilot may improve even
        # earlier data blocks. It is substantially less vulnerable than the
        # time-domain Golay pair to phone-speaker limiting.
        pilot_channels: dict[int, NDArray[np.complex128]] = {}
        for index, spectrum in enumerate(spectra, start=1):
            if self.scheduler.is_pilot(index):
                pilot_channels[index] = self._pilot_channel(spectrum, golay_channel)

        rows: list[NDArray[np.complex128]] = []
        weights: list[NDArray[np.float64]] = []
        pilot_positions = np.asarray(sorted(pilot_channels), dtype=int)

        for one_based, spectrum in enumerate(spectra, start=1):
            if self.scheduler.is_pilot(one_based):
                continue
            if len(pilot_positions):
                nearest = int(pilot_positions[np.argmin(np.abs(pilot_positions - one_based))])
                channel = pilot_channels[nearest]
            else:
                channel = golay_channel

            estimate = self._equalise(spectrum, channel)
            bins = c.data_bins
            reliability = np.abs(channel[bins]) ** 2
            reliability /= float(np.median(reliability) + EPS)
            reliability = np.clip(reliability, 0.10, 8.0)
            hard = QPSK.hard_symbols(estimate)
            residual = float(np.median(np.abs(estimate - hard) ** 2))
            common = 2.0 / max(residual, 0.05)
            rows.append(estimate)
            weights.append(common * reliability)

        if not rows:
            raise ValueError("no JOSS-F data OFDM blocks were found")
        return np.stack(rows), np.stack(weights), block_count

    def decode_signal(self, received: NDArray[np.float64]) -> tuple[DecodedPacket, ReceptionInfo]:
        received = np.asarray(received, dtype=float)
        corrected, raw_sync_start, offset_ppm = self.synchronise_and_correct(received)
        golay_channel = self._golay_channel(corrected)
        data_start, adjustment = self._refine_data_start(corrected, self.config.preamble_length)
        rows, weights, received_blocks = self._extract_data_rows(
            corrected, data_start, golay_channel
        )

        complete_groups = len(rows) // self.config.data_symbols_per_group
        if complete_groups == 0:
            raise ValueError("fewer than 30 data OFDM symbols were recovered")

        recovered_bits: list[NDArray[np.uint8]] = []
        successful_blocks = 0
        total_blocks = 0
        decoded_groups = 0
        requested_bytes: int | None = None
        header_error: str | None = None
        maximum_recoverable = (
            complete_groups * self.config.ldpc_blocks_per_group * self.config.ldpc_info_bits // 8
        )

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
                try:
                    candidate = PacketCodec.expected_total_bytes(decoded_bytes)
                    if candidate is not None and candidate <= maximum_recoverable:
                        requested_bytes = candidate
                    elif candidate is not None:
                        header_error = (
                            f"decoded header requests {candidate} bytes, but this recording "
                            f"can contain at most {maximum_recoverable}"
                        )
                except (ValueError, UnicodeError) as exc:
                    header_error = str(exc)

            if requested_bytes is not None and len(decoded_bytes) >= requested_bytes:
                packet = PacketCodec.parse(decoded_bytes[:requested_bytes])
                info = ReceptionInfo(
                    raw_sync_start,
                    offset_ppm,
                    received_blocks,
                    len(rows),
                    decoded_groups,
                    successful_blocks,
                    total_blocks,
                    adjustment,
                )
                return packet, info

        decoded_bytes = bits_to_bytes(np.concatenate(recovered_bits))
        try:
            packet = PacketCodec.parse(decoded_bytes)
        except (ValueError, UnicodeError) as exc:
            detail = header_error or str(exc)
            raise ValueError(
                "JOSS-F header could not be decoded. "
                f"LDPC parity checks passed for {successful_blocks}/{total_blocks} blocks; "
                f"data-start adjustment was {adjustment:+d} samples. {detail}"
            ) from exc

        info = ReceptionInfo(
            raw_sync_start,
            offset_ppm,
            received_blocks,
            len(rows),
            decoded_groups,
            successful_blocks,
            total_blocks,
            adjustment,
        )
        return packet, info

    def decode_wav(self, wav_path: str | Path, output_directory: str | Path = ".") -> Path:
        received = read_wav(wav_path, self.config)
        peak = float(np.max(np.abs(received))) if len(received) else 0.0
        clipped = float(np.mean(np.abs(received) >= 0.995)) if len(received) else 0.0
        packet, info = self.decode_signal(received)
        output_path = Path(output_directory) / safe_output_name(packet.filename)
        output_path.write_bytes(packet.payload)
        print(f"Decoded {wav_path}")
        print(f"  input peak       : {peak:.3f} ({100.0 * clipped:.3f}% clipped)")
        print(f"  sync sample      : {info.sync_sample}")
        print(f"  sample offset    : {info.sample_rate_offset_ppm:+.1f} ppm")
        print(f"  OFDM adjustment  : {info.data_start_adjustment:+d} samples")
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