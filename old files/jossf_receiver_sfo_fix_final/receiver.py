from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import argparse
import csv
import json

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
    initial_sample_rate_offset_ppm: float = 0.0
    residual_sample_rate_offset_ppm: float = 0.0
    pilot_ofdm_blocks: int = 0
    median_evm_percent: float = float("nan")


@dataclass(frozen=True)
class TimingObservation:
    block_index: int
    timing_samples: float
    phase_slope_rad_per_bin: float
    phase_intercept_rad: float
    fit_error_rad: float
    weight: float
    is_pilot: bool


@dataclass(frozen=True)
class ChannelModel:
    base_channel: NDArray[np.complex128]
    pilot_positions: NDArray[np.int64]
    pilot_phase_slopes: NDArray[np.float64]
    pilot_phase_intercepts: NDArray[np.float64]
    pilot_log_gains: NDArray[np.float64]
    pilot_channels: NDArray[np.complex128]
    channel_dispersion: NDArray[np.float64]


@dataclass(frozen=True)
class ReceptionDiagnostics:
    chirp_numbers: NDArray[np.float64]
    chirp_positions: NDArray[np.float64]
    chirp_fitted_positions: NDArray[np.float64]
    timing_before: tuple[TimingObservation, ...]
    timing_after: tuple[TimingObservation, ...]
    constellation_before: NDArray[np.complex128]
    constellation_after: NDArray[np.complex128]
    channel: NDArray[np.complex128]
    golay_channel: NDArray[np.complex128]
    evm_percent: NDArray[np.float64]
    data_block_positions: NDArray[np.int64]
    initial_sfo_ppm: float
    residual_sfo_ppm: float
    total_sfo_ppm: float
    data_start: int


@dataclass(frozen=True)
class RawReception:
    decoded_bytes: bytes
    info: ReceptionInfo
    diagnostics: ReceptionDiagnostics | None = None


@dataclass(frozen=True)
class HeaderRecovery:
    packet: DecodedPacket
    method: str
    flipped_fixed_header_bits: tuple[int, ...] = ()


@dataclass(frozen=True)
class TextSalvage:
    text: str
    payload_start: int
    payload_end: int
    method: str
    score: float


class Receiver:
    """JOSS-F receiver with receiver-only SFO and channel tracking.

    The transmitter, standardised preamble, pilot positions, occupied bins,
    QPSK mapping, interleaver, framing and deliberately Jossy-compatible LDPC
    graph are untouched.  All adaptive processing in this class is applied only
    after a WAV file has been received.
    """

    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        self.ldpc = WiMaxLDPC(config)
        self.interleaver = StandardInterleaver(config)
        self.ofdm = OFDM(config)
        self.scheduler = PilotScheduler(config.pilot_interval)
        self.chirp = LinearChirp(config)
        self.golay = GolayPilot(config)
        self.known_pilot = KnownOFDMPilot(config)

    @staticmethod
    def _quadratic_peak(metric: NDArray[np.float64], index: int) -> float:
        if not 0 < index < len(metric) - 1:
            return float(index)
        left, centre, right = metric[index - 1], metric[index], metric[index + 1]
        denominator = left - 2.0 * centre + right
        if abs(denominator) < EPS:
            return float(index)
        delta = 0.5 * (left - right) / denominator
        return float(index) + float(np.clip(delta, -0.5, 0.5))

    def _find_chirp_chain(
        self, received: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return sub-sample chirp starts and their matched-filter metric."""
        chirp = self.chirp.samples()
        n = self.config.chirp_length
        metric = np.abs(fftconvolve(received, chirp[::-1], mode="valid"))
        peaks, _ = find_peaks(
            metric,
            distance=int(0.75 * n),
            prominence=max(float(np.max(metric)) * 0.035, EPS),
        )
        candidate_starts = np.unique(
            np.concatenate(
                [peaks.astype(int), np.array([0, int(np.argmax(metric))], dtype=int)]
            )
        )

        best_chain: list[int] | None = None
        best_score = -np.inf
        tolerance = int(0.08 * n)
        for first in candidate_starts:
            chain = [int(first)]
            score = float(metric[first])
            current = int(first)
            for _ in range(1, self.config.chirp_count):
                target = current + n
                nearby = peaks[np.abs(peaks - target) <= tolerance]
                if not len(nearby):
                    break
                chosen = int(nearby[np.argmax(metric[nearby])])
                chain.append(chosen)
                score += float(metric[chosen])
                current = chosen
            if len(chain) == self.config.chirp_count:
                score /= 1.0 + float(np.std(np.diff(chain)))
                if score > best_score:
                    best_score = score
                    best_chain = chain

        if best_chain is None:
            template = self.chirp.train()
            coarse = np.abs(fftconvolve(received, template[::-1], mode="valid"))
            sync = int(np.argmax(coarse))
            starts = sync + np.arange(self.config.chirp_count, dtype=float) * n
            return starts, metric

        starts = np.asarray(
            [self._quadratic_peak(metric, index) for index in best_chain], dtype=float
        )
        return starts, metric

    @staticmethod
    def _robust_line(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
        iterations: int = 4,
    ) -> tuple[float, float, NDArray[np.bool_]]:
        """Huber-like iteratively reweighted affine fit."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if weights is None:
            base_weights = np.ones_like(x)
        else:
            base_weights = np.clip(np.asarray(weights, dtype=float), EPS, np.inf)
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(base_weights)
        if np.count_nonzero(keep) < 2:
            raise ValueError("at least two finite observations are required")

        working_weights = base_weights.copy()
        slope = 0.0
        intercept = float(np.median(y[keep]))
        for _ in range(iterations):
            valid = keep & (working_weights > 0.0)
            design = np.column_stack([x[valid], np.ones(np.count_nonzero(valid))])
            root = np.sqrt(working_weights[valid])
            slope, intercept = np.linalg.lstsq(
                design * root[:, None], y[valid] * root, rcond=None
            )[0]
            residual = y - (slope * x + intercept)
            centre = float(np.median(residual[keep]))
            mad = float(np.median(np.abs(residual[keep] - centre)))
            scale = max(1.4826 * mad, 1e-6)
            huber = np.ones_like(x)
            large = np.abs(residual - centre) > 2.5 * scale
            huber[large] = (2.5 * scale) / np.abs(residual[large] - centre)
            working_weights = base_weights * huber
            keep &= np.abs(residual - centre) <= 6.0 * scale
        return float(slope), float(intercept), keep

    def _resample_from_sync(
        self,
        received: NDArray[np.float64],
        intercept: float,
        ratio: float,
    ) -> NDArray[np.float64]:
        """Sample the recording on the transmitter's nominal time grid.

        ``ratio`` is received samples per nominal transmitter sample.  Cubic
        interpolation is used once from the original recording for the final
        correction, avoiding cascaded interpolators.  The occupied OFDM band
        ends at 12 kHz, comfortably below Nyquist at 48 kHz.
        """
        if not 0.995 <= ratio <= 1.005:
            raise ValueError(f"implausible sampling-rate ratio estimated: {ratio:.9f}")
        output_length = int(np.floor((len(received) - intercept - 1.0) / ratio)) + 1
        if output_length <= 0:
            raise ValueError("synchronisation point lies beyond the recording")
        source_positions = intercept + np.arange(output_length, dtype=float) * ratio
        sample_axis = np.arange(len(received), dtype=float)
        interpolator = CubicSpline(sample_axis, received, extrapolate=False)
        corrected = np.asarray(interpolator(source_positions), dtype=float)
        return np.nan_to_num(corrected)

    def synchronise_and_correct(
        self, received: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], int, float]:
        """Backward-compatible chirp-only synchronisation API."""
        received = np.asarray(received, dtype=float)
        if len(received) < self.config.chirp_train_length:
            raise ValueError("recording is shorter than the JOSS-F chirp train")
        starts, _ = self._find_chirp_chain(received)
        numbers = np.arange(self.config.chirp_count, dtype=float)
        period, intercept, _ = self._robust_line(numbers, starts)
        ratio = period / self.config.chirp_length
        corrected = self._resample_from_sync(received, intercept, ratio)
        return corrected, int(round(intercept)), (ratio - 1.0) * 1e6

    def _golay_channel(self, received: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Regularised LS channel estimate from all eight Golay pulses."""
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
            for relative, reference_f in ((a_relative, x_a_f), (b_relative, x_b_f)):
                start = golay_start + relative
                window = received[start : start + transform_length]
                if len(window) < transform_length:
                    window = np.pad(window, (0, transform_length - len(window)))
                received_f = np.fft.rfft(window, n=transform_length)
                numerator += received_f * np.conj(reference_f)
                denominator += np.abs(reference_f) ** 2

        regularisation = c.golay_regularisation * float(np.max(denominator))
        impulse = np.fft.irfft(
            numerator / (denominator + regularisation), n=transform_length
        )[: c.cyclic_prefix]
        return np.fft.rfft(impulse, n=c.fft_length)

    def _cp_score(self, received: NDArray[np.float64], start: int) -> float:
        c = self.config
        length = c.ofdm_symbol_length
        available = (len(received) - start) // length
        count = min(max(available, 0), 10)
        if count <= 0:
            return -np.inf
        scores: list[float] = []
        for index in range(count):
            lo = start + index * length
            block = received[lo : lo + length]
            if len(block) != length:
                continue
            prefix = block[: c.cyclic_prefix]
            tail = block[c.fft_length : c.fft_length + c.cyclic_prefix]
            denominator = np.linalg.norm(prefix) * np.linalg.norm(tail) + EPS
            scores.append(float(abs(np.vdot(prefix, tail)) / denominator))
        return float(np.median(scores)) if scores else -np.inf

    def _refine_data_start(self, received: NDArray[np.float64], nominal: int) -> tuple[int, int]:
        """Refine the first OFDM boundary without changing symbol structure."""
        radius = self.config.data_start_refine_radius
        valid_min = max(-radius, -nominal)
        valid_max = min(radius, len(received) - nominal - self.config.ofdm_symbol_length)
        if valid_min > valid_max:
            raise ValueError("recording does not contain a complete OFDM block")
        coarse_offsets = range(valid_min, valid_max + 1, 4)
        coarse = max(coarse_offsets, key=lambda offset: self._cp_score(received, nominal + offset))
        fine_offsets = range(max(valid_min, coarse - 6), min(valid_max, coarse + 6) + 1)
        best = max(fine_offsets, key=lambda offset: self._cp_score(received, nominal + offset))
        return nominal + best, int(best)

    def _extract_spectra(
        self, received: NDArray[np.float64], data_start: int
    ) -> tuple[list[NDArray[np.complex128]], NDArray[np.float64]]:
        c = self.config
        length = c.ofdm_symbol_length
        available = max(0, len(received) - data_start)
        count = available // length
        if count == 0:
            raise ValueError("no complete JOSS-F OFDM blocks were found")

        blocks = [
            received[data_start + i * length : data_start + (i + 1) * length]
            for i in range(count)
        ]
        rms = np.asarray([float(np.sqrt(np.mean(block * block))) for block in blocks])

        # Recorded files often contain seconds of trailing room noise.  Stop at
        # a sustained energy collapse, but never trim within the first complete
        # 30-data-symbol LDPC group.
        first_group_blocks = self.scheduler.total_blocks_for_data_rows(c.data_symbols_per_group)
        if count > first_group_blocks:
            reference = float(np.median(rms[:first_group_blocks]))
            # A valid full-carrier OFDM symbol has nearly constant RMS.  Only
            # trim a sustained collapse far below that level; do not estimate
            # the noise floor from the end because the recording may end while
            # a long packet is still active.
            threshold = 0.15 * reference
            low_run = 0
            trim_at = count
            for i in range(first_group_blocks, count):
                if rms[i] < threshold:
                    low_run += 1
                else:
                    low_run = 0
                if low_run >= 2:
                    trim_at = i - low_run + 1
                    break
            blocks = blocks[:trim_at]
            rms = rms[:trim_at]

        spectra = [self.ofdm.spectrum(block) for block in blocks]
        return spectra, rms

    @staticmethod
    def _fit_unwrapped_phase(
        x: NDArray[np.float64],
        phase: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        slope, intercept, keep = Receiver._robust_line(x, phase, weights)
        residual = phase - (slope * x + intercept)
        error = float(np.median(np.abs(residual[keep]))) if np.any(keep) else float("inf")
        return slope, intercept, error

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

    def _timing_observations(
        self,
        spectra: list[NDArray[np.complex128]],
        reference_channel: NDArray[np.complex128],
    ) -> tuple[tuple[TimingObservation, ...], NDArray[np.complex128]]:
        """Measure residual symbol timing from QPSK's fourth-power phase.

        For data symbols, raising normalised QPSK to the fourth power removes
        the unknown data because every JOSS-F QPSK point has fourth power -1.
        A linear phase ramp across subcarriers then directly measures timing
        drift.  Known pilot blocks use their exact symbols instead.
        """
        c = self.config
        bins = c.data_bins
        x = bins.astype(float) - float(np.mean(bins))
        h = reference_channel[bins]
        regularisation = c.equaliser_regularisation * (float(np.median(np.abs(h) ** 2)) + EPS)
        observations: list[TimingObservation] = []
        constellation_samples: list[NDArray[np.complex128]] = []
        known = self.known_pilot.transmitted_frequency[bins]

        for block_index, spectrum in enumerate(spectra, start=1):
            equalised = spectrum[bins] * np.conj(h) / (np.abs(h) ** 2 + regularisation)
            if block_index <= 60:
                display_scale = float(np.median(np.abs(equalised)))
                display = equalised if display_scale < EPS else equalised * (np.sqrt(2.0) / display_scale)
                constellation_samples.append(display[::8])
            magnitude = np.abs(equalised)
            weights = np.clip(magnitude, 0.05, np.percentile(magnitude, 90) + EPS)

            if self.scheduler.is_pilot(block_index):
                ratio = equalised / known
                phase = np.unwrap(np.angle(ratio))
                slope, intercept, error = self._fit_unwrapped_phase(x, phase, weights)
                is_pilot = True
            else:
                unit = equalised / (magnitude + EPS)
                phase4 = np.unwrap(np.angle(-(unit ** 4)))
                slope4, intercept4, error4 = self._fit_unwrapped_phase(x, phase4, weights)
                slope = slope4 / 4.0
                intercept = intercept4 / 4.0
                error = error4 / 4.0
                is_pilot = False

            timing = -slope * c.fft_length / (2.0 * np.pi)
            if not np.isfinite(timing) or abs(timing) > c.sfo_max_timing_error_samples:
                continue
            quality = 1.0 / max(error, 0.01)
            if is_pilot:
                quality *= 4.0
            observations.append(
                TimingObservation(
                    block_index=block_index,
                    timing_samples=float(timing),
                    phase_slope_rad_per_bin=float(slope),
                    phase_intercept_rad=float(intercept),
                    fit_error_rad=float(error),
                    weight=float(quality),
                    is_pilot=is_pilot,
                )
            )

        constellation = (
            np.concatenate(constellation_samples)
            if constellation_samples
            else np.empty(0, dtype=np.complex128)
        )
        return tuple(observations), constellation

    def _estimate_residual_sfo(
        self, observations: tuple[TimingObservation, ...]
    ) -> tuple[float, NDArray[np.bool_]]:
        c = self.config
        if len(observations) < c.sfo_min_timing_observations:
            return 0.0, np.zeros(len(observations), dtype=bool)
        blocks = np.asarray([item.block_index for item in observations], dtype=float)
        timing = np.asarray([item.timing_samples for item in observations], dtype=float)
        weights = np.asarray([item.weight for item in observations], dtype=float)
        x = (blocks - float(np.median(blocks))) * c.ofdm_symbol_length
        slope, _, keep = self._robust_line(x, timing, weights)
        ppm = slope * 1e6
        if not np.isfinite(ppm) or abs(ppm) > c.sfo_max_residual_ppm:
            return 0.0, np.zeros(len(observations), dtype=bool)
        if abs(ppm) < c.sfo_min_correction_ppm:
            return 0.0, keep
        return float(ppm), keep

    def _build_channel_model(
        self,
        spectra: list[NDArray[np.complex128]],
        golay_channel: NDArray[np.complex128],
    ) -> ChannelModel:
        c = self.config
        bins = c.data_bins
        x = bins.astype(float) - float(np.mean(bins))
        positions: list[int] = []
        raw_channels: list[NDArray[np.complex128]] = []
        for block_index, spectrum in enumerate(spectra, start=1):
            if self.scheduler.is_pilot(block_index):
                positions.append(block_index)
                raw_channels.append(self._pilot_channel(spectrum, golay_channel))
        if not raw_channels:
            return ChannelModel(
                base_channel=golay_channel.copy(),
                pilot_positions=np.empty(0, dtype=np.int64),
                pilot_phase_slopes=np.empty(0),
                pilot_phase_intercepts=np.empty(0),
                pilot_log_gains=np.empty(0),
                pilot_channels=np.empty((0, len(golay_channel)), dtype=np.complex128),
                channel_dispersion=np.zeros(len(bins), dtype=float),
            )

        raw = np.stack(raw_channels)
        reference = raw[0, bins]
        reference_power = np.abs(reference) ** 2
        reference_weights = np.clip(reference_power, np.percentile(reference_power, 10), np.inf)
        aligned: list[NDArray[np.complex128]] = []
        for channel in raw:
            active = channel[bins]
            gain = float(np.median(np.abs(active) / (np.abs(reference) + EPS)))
            relative = active * np.conj(reference)
            phase = np.unwrap(np.angle(relative))
            slope, intercept, _ = self._fit_unwrapped_phase(x, phase, reference_weights)
            aligned.append(active / (max(gain, EPS) * np.exp(1j * (slope * x + intercept))))

        aligned_array = np.stack(aligned)
        base_active = np.median(aligned_array.real, axis=0) + 1j * np.median(
            aligned_array.imag, axis=0
        )
        base_channel = golay_channel.copy()
        base_channel[bins] = base_active

        slopes: list[float] = []
        intercepts: list[float] = []
        log_gains: list[float] = []
        aligned_again: list[NDArray[np.complex128]] = []
        base_weights = np.clip(np.abs(base_active) ** 2, EPS, np.inf)
        for channel in raw:
            active = channel[bins]
            gain = float(np.median(np.abs(active) / (np.abs(base_active) + EPS)))
            phase = np.unwrap(np.angle(active * np.conj(base_active)))
            slope, intercept, _ = self._fit_unwrapped_phase(x, phase, base_weights)
            slopes.append(slope)
            intercepts.append(intercept)
            log_gains.append(np.log(max(gain, EPS)))
            aligned_again.append(active / (max(gain, EPS) * np.exp(1j * (slope * x + intercept))))

        intercept_array = np.unwrap(np.asarray(intercepts, dtype=float))
        aligned_again_array = np.stack(aligned_again)
        if len(aligned_again_array) > 1:
            dispersion = np.median(np.abs(aligned_again_array - base_active[None, :]) ** 2, axis=0)
        else:
            dispersion = np.zeros(len(bins), dtype=float)

        return ChannelModel(
            base_channel=base_channel,
            pilot_positions=np.asarray(positions, dtype=np.int64),
            pilot_phase_slopes=np.asarray(slopes, dtype=float),
            pilot_phase_intercepts=intercept_array,
            pilot_log_gains=np.asarray(log_gains, dtype=float),
            pilot_channels=raw,
            channel_dispersion=np.asarray(dispersion, dtype=float),
        )

    @staticmethod
    def _interpolate_parameter(
        positions: NDArray[np.int64], values: NDArray[np.float64], block_index: int
    ) -> float:
        if not len(positions):
            return 0.0
        if len(positions) == 1:
            return float(values[0])
        return float(np.interp(block_index, positions.astype(float), values))

    def _channel_at(self, model: ChannelModel, block_index: int) -> NDArray[np.complex128]:
        if not len(model.pilot_positions):
            return model.base_channel
        c = self.config
        bins = c.data_bins
        x = bins.astype(float) - float(np.mean(bins))
        slope = self._interpolate_parameter(
            model.pilot_positions, model.pilot_phase_slopes, block_index
        )
        intercept = self._interpolate_parameter(
            model.pilot_positions, model.pilot_phase_intercepts, block_index
        )
        log_gain = self._interpolate_parameter(
            model.pilot_positions, model.pilot_log_gains, block_index
        )
        channel = model.base_channel.copy()
        channel[bins] *= np.exp(log_gain + 1j * (slope * x + intercept))
        return channel

    def _equalise_block(
        self,
        spectrum: NDArray[np.complex128],
        channel: NDArray[np.complex128],
        channel_dispersion: NDArray[np.float64],
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], float, NDArray[np.complex128]]:
        c = self.config
        bins = c.data_bins
        x = bins.astype(float) - float(np.mean(bins))
        h = channel[bins]
        power = np.abs(h) ** 2
        regularisation = c.equaliser_regularisation * (float(np.median(power)) + EPS)
        raw = spectrum[bins] * np.conj(h) / (power + regularisation)

        magnitude = np.abs(raw)
        unit = raw / (magnitude + EPS)
        phase4 = np.unwrap(np.angle(-(unit ** 4)))
        fit_weights = np.clip(magnitude, 0.05, np.percentile(magnitude, 90) + EPS)
        slope4, intercept4, error4 = self._fit_unwrapped_phase(x, phase4, fit_weights)
        slope = slope4 / 4.0
        intercept = intercept4 / 4.0
        # The QPSK fourth-power estimate has a pi/2 ambiguity.  The channel
        # pilots already establish the correct quadrant, so select the branch
        # closest to zero residual common phase.
        intercept = (intercept + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        if error4 / 4.0 > 0.35:
            slope = 0.0
            intercept = 0.0
        corrected = raw * np.exp(-1j * (slope * x + intercept))

        robust_magnitude = float(np.median(np.abs(corrected)))
        if robust_magnitude > EPS:
            corrected *= np.sqrt(2.0) / robust_magnitude

        hard = QPSK.hard_symbols(corrected)
        error_power = float(np.mean(np.abs(corrected - hard) ** 2))
        evm = 100.0 * np.sqrt(error_power / 2.0)

        all_bins = np.arange(len(spectrum))
        unused = (all_bins < c.first_data_bin - 8) | (all_bins > c.last_data_bin + 8)
        noise_power = float(np.median(np.abs(spectrum[unused]) ** 2)) + EPS
        reliability = power / (noise_power + channel_dispersion + regularisation)
        reliability /= float(np.median(reliability) + EPS)
        reliability = np.clip(reliability, 0.05, 12.0)
        common = 2.0 / max(error_power, 0.04)
        return corrected, common * reliability, evm, raw

    def _extract_data_rows(
        self,
        spectra: list[NDArray[np.complex128]],
        channel_model: ChannelModel,
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.complex128],
    ]:
        rows: list[NDArray[np.complex128]] = []
        weights: list[NDArray[np.float64]] = []
        evm: list[float] = []
        positions: list[int] = []
        raw_constellation: list[NDArray[np.complex128]] = []
        for block_index, spectrum in enumerate(spectra, start=1):
            if self.scheduler.is_pilot(block_index):
                continue
            channel = self._channel_at(channel_model, block_index)
            estimate, reliability, block_evm, raw = self._equalise_block(
                spectrum, channel, channel_model.channel_dispersion
            )
            rows.append(estimate)
            weights.append(reliability)
            evm.append(block_evm)
            positions.append(block_index)
            if len(raw_constellation) < 60:
                raw_constellation.append(raw[::8])
        if not rows:
            raise ValueError("no JOSS-F data OFDM blocks were found")
        return (
            np.stack(rows),
            np.stack(weights),
            np.asarray(evm, dtype=float),
            np.asarray(positions, dtype=np.int64),
            np.concatenate(raw_constellation) if raw_constellation else np.empty(0, complex),
        )

    @staticmethod
    def _limit_constellation(
        values: NDArray[np.complex128], maximum: int
    ) -> NDArray[np.complex128]:
        values = np.asarray(values, dtype=np.complex128).reshape(-1)
        if len(values) <= maximum:
            return values
        indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
        return values[indices]

    def decode_raw_signal(
        self,
        received: NDArray[np.float64],
        *,
        collect_diagnostics: bool = False,
    ) -> RawReception:
        """Decode complete LDPC groups while retaining a corrupt-header fallback."""
        received = np.asarray(received, dtype=float)
        if len(received) < self.config.preamble_length:
            raise ValueError("recording is shorter than the JOSS-F preamble")

        chirp_positions, _ = self._find_chirp_chain(received)
        chirp_numbers = np.arange(self.config.chirp_count, dtype=float)
        chirp_period, intercept, _ = self._robust_line(chirp_numbers, chirp_positions)
        initial_ratio = chirp_period / self.config.chirp_length
        initial_ppm = (initial_ratio - 1.0) * 1e6
        total_ratio = initial_ratio

        timing_before: tuple[TimingObservation, ...] = ()
        constellation_before = np.empty(0, dtype=np.complex128)

        # Diagnostics deliberately include a no-SFO-correction baseline so the
        # accumulated constellation smear and timing ramp are visible.  This
        # branch is never used for decoding.
        if collect_diagnostics:
            try:
                uncorrected = self._resample_from_sync(received, intercept, 1.0)
                uncorrected_start, _ = self._refine_data_start(
                    uncorrected, self.config.preamble_length
                )
                uncorrected_spectra, _ = self._extract_spectra(
                    uncorrected, uncorrected_start
                )
                uncorrected_golay = self._golay_channel(uncorrected)
                uncorrected_pilots = [
                    self._pilot_channel(spectrum, uncorrected_golay)
                    for index, spectrum in enumerate(uncorrected_spectra, start=1)
                    if self.scheduler.is_pilot(index)
                ]
                if uncorrected_pilots:
                    timing_before, constellation_before = self._timing_observations(
                        uncorrected_spectra, uncorrected_pilots[0]
                    )
            except (ValueError, np.linalg.LinAlgError):
                timing_before = ()
                constellation_before = np.empty(0, dtype=np.complex128)

        residual_ppm_accumulated = 0.0
        corrected = self._resample_from_sync(received, intercept, total_ratio)

        for iteration in range(self.config.sfo_refinement_iterations):
            data_start, _ = self._refine_data_start(corrected, self.config.preamble_length)
            spectra, _ = self._extract_spectra(corrected, data_start)
            golay_channel = self._golay_channel(corrected)
            pilot_channels = [
                self._pilot_channel(spectrum, golay_channel)
                for index, spectrum in enumerate(spectra, start=1)
                if self.scheduler.is_pilot(index)
            ]
            if not pilot_channels:
                break
            reference_channel = pilot_channels[0]
            observations, _ = self._timing_observations(
                spectra, reference_channel
            )
            residual_ppm, _ = self._estimate_residual_sfo(observations)
            if residual_ppm == 0.0:
                break
            residual_ratio = 1.0 + residual_ppm * 1e-6
            total_ratio *= residual_ratio
            residual_ppm_accumulated = (total_ratio / initial_ratio - 1.0) * 1e6
            corrected = self._resample_from_sync(received, intercept, total_ratio)

        golay_channel = self._golay_channel(corrected)
        data_start, adjustment = self._refine_data_start(
            corrected, self.config.preamble_length
        )
        spectra, _ = self._extract_spectra(corrected, data_start)
        channel_model = self._build_channel_model(spectra, golay_channel)
        timing_after, _ = self._timing_observations(
            spectra,
            channel_model.base_channel
            if len(channel_model.pilot_positions)
            else golay_channel,
        )
        rows, weights, evm, data_positions, raw_final = self._extract_data_rows(
            spectra, channel_model
        )

        complete_groups = len(rows) // self.config.data_symbols_per_group
        if complete_groups == 0:
            raise ValueError("fewer than 30 data OFDM symbols were recovered")

        recovered_bits: list[NDArray[np.uint8]] = []
        successful_blocks = 0
        total_blocks = 0
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

        decoded_bytes = bits_to_bytes(np.concatenate(recovered_bits))
        total_ppm = (total_ratio - 1.0) * 1e6
        info = ReceptionInfo(
            sync_sample=int(round(intercept)),
            sample_rate_offset_ppm=float(total_ppm),
            received_ofdm_blocks=len(spectra),
            data_ofdm_blocks=len(rows),
            decoded_ldpc_groups=complete_groups,
            successful_ldpc_blocks=successful_blocks,
            total_ldpc_blocks=total_blocks,
            data_start_adjustment=adjustment,
            initial_sample_rate_offset_ppm=float(initial_ppm),
            residual_sample_rate_offset_ppm=float(residual_ppm_accumulated),
            pilot_ofdm_blocks=len(channel_model.pilot_positions),
            median_evm_percent=float(np.median(evm)),
        )

        diagnostics: ReceptionDiagnostics | None = None
        if collect_diagnostics:
            fitted = intercept + chirp_period * chirp_numbers
            after_constellation = self._limit_constellation(
                rows.reshape(-1), self.config.diagnostic_constellation_points
            )
            before_constellation = self._limit_constellation(
                constellation_before, self.config.diagnostic_constellation_points
            )
            diagnostics = ReceptionDiagnostics(
                chirp_numbers=chirp_numbers,
                chirp_positions=chirp_positions,
                chirp_fitted_positions=fitted,
                timing_before=timing_before,
                timing_after=timing_after,
                constellation_before=before_constellation,
                constellation_after=after_constellation,
                channel=channel_model.base_channel,
                golay_channel=golay_channel,
                evm_percent=evm,
                data_block_positions=data_positions,
                initial_sfo_ppm=float(initial_ppm),
                residual_sfo_ppm=float(residual_ppm_accumulated),
                total_sfo_ppm=float(total_ppm),
                data_start=data_start,
            )
        return RawReception(decoded_bytes, info, diagnostics)
    def _plausible_recovered_filename(filename: str) -> bool:
        """Conservative filename filter used only for automatic header repair."""
        if not filename or len(filename.encode("utf-8")) > 255:
            return False
        if filename != Path(filename).name or "/" in filename or "\\" in filename:
            return False
        return all(character.isprintable() and character not in "\r\n\x00" for character in filename)

    @classmethod
    def _candidate_packet(
        cls,
        data: bytes,
        *,
        allow_replacement_filename: bool = False,
    ) -> DecodedPacket | None:
        if len(data) < PacketCodec.MIN_HEADER_BYTES:
            return None
        header_length = int.from_bytes(data[:2], "big", signed=False)
        file_size = int.from_bytes(data[2:6], "big", signed=False)
        total = header_length + file_size
        if not (PacketCodec.MIN_HEADER_BYTES <= header_length <= len(data)):
            return None
        if total > len(data):
            return None

        name_bytes = data[6:header_length]
        try:
            filename = name_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            if not allow_replacement_filename:
                return None
            filename = name_bytes.decode("utf-8", errors="replace")

        if not cls._plausible_recovered_filename(filename):
            return None
        return DecodedPacket(
            filename=filename,
            file_size=file_size,
            payload=data[header_length:total],
            header_length=header_length,
        )

    @classmethod
    def recover_packet_header(
        cls,
        decoded_bytes: bytes,
        *,
        max_fixed_header_bit_flips: int = 3,
    ) -> HeaderRecovery | None:
        """Try strict parsing, filename salvage, then constrained A/B repair.

        Only the first 48 bits are modified because JOSS-F defines those as the
        two fixed-width length fields.  A repaired candidate is accepted only
        when its UTF-8 filename and byte bounds are plausible and its best score
        is unambiguous.
        """
        try:
            return HeaderRecovery(PacketCodec.parse(decoded_bytes), "strict")
        except (ValueError, UnicodeError):
            pass

        # If A and B are valid but only filename bytes were damaged, boundaries
        # are still trustworthy. Preserve the payload and replace invalid UTF-8.
        lenient = cls._candidate_packet(
            decoded_bytes,
            allow_replacement_filename=True,
        )
        if lenient is not None and "�" in lenient.filename:
            return HeaderRecovery(lenient, "valid lengths; damaged UTF-8 filename")

        if len(decoded_bytes) < 6 or max_fixed_header_bit_flips <= 0:
            return None

        fixed = bytearray(decoded_bytes[:6])
        rest = decoded_bytes[6:]
        for flip_count in range(1, max_fixed_header_bit_flips + 1):
            candidates: list[tuple[int, HeaderRecovery]] = []
            for bit_indices in combinations(range(48), flip_count):
                repaired = bytearray(fixed)
                for bit_index in bit_indices:
                    # bytes_to_bits/np.unpackbits use MSB-first bit numbering.
                    repaired[bit_index // 8] ^= 1 << (7 - bit_index % 8)
                candidate_data = bytes(repaired) + rest
                packet = cls._candidate_packet(candidate_data)
                if packet is None:
                    continue

                score = 0
                filename_length = len(packet.filename.encode("utf-8"))
                if filename_length <= 64:
                    score += 4
                if Path(packet.filename).suffix:
                    score += 3
                if packet.file_size > 0:
                    score += 1
                if packet.header_length == 6 + filename_length:
                    score += 2
                candidates.append(
                    (
                        score,
                        HeaderRecovery(
                            packet,
                            f"repaired {flip_count} fixed-header bit(s)",
                            tuple(bit_indices),
                        ),
                    )
                )

            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_score = candidates[0][0]
            best = [item[1] for item in candidates if item[0] == best_score]
            if len(best) == 1:
                return best[0]
            # Ambiguous repairs are deliberately not guessed.
            return None
        return None

    @staticmethod
    def _text_byte_quality(byte_value: int) -> float:
        """Score how plausible one byte is for a recovered .txt payload.

        This intentionally favours ordinary UTF-8/ASCII text.  It is not used
        for formal packet recovery; it is only a human-inspection fallback when
        the JOSS-F header is too damaged to provide boundaries.
        """
        if byte_value in (9, 10, 13):  # tab/newline/carriage return
            return 1.0
        if 32 <= byte_value <= 126:    # printable ASCII
            return 1.0
        if 128 <= byte_value <= 244:   # possible UTF-8 continuation/lead byte
            return 0.45
        return -1.4

    @classmethod
    def _trim_text_payload(cls, data: bytes, start: int) -> tuple[int, float]:
        """Return the likely end of a text payload beginning at start.

        The decoded LDPC stream contains the real packet followed by padding.
        If the header is broken, the payload length is unknown, so we keep the
        readable prefix and stop at the first sustained non-text region.
        """
        if not (0 <= start < len(data)):
            return start, -1e9

        min_payload = 12
        bad_run = 0
        end = len(data)
        for index, byte in enumerate(data[start:], start=start):
            quality = cls._text_byte_quality(byte)
            if quality < 0.0:
                bad_run += 1
            else:
                bad_run = 0

            # Random LDPC padding almost immediately creates multiple control
            # bytes.  Normal text can contain occasional UTF-8 continuation
            # bytes, so only stop on a sustained bad run.
            if index - start >= min_payload and bad_run >= 4:
                end = index - bad_run + 1
                break

        # If no obvious bad run appears, use a soft sliding-window check to
        # avoid writing hundreds of random bytes after short files.
        if end == len(data):
            window = 64
            for offset in range(min_payload, max(min_payload, len(data) - start - window + 1)):
                chunk = data[start + offset: start + offset + window]
                printable_ratio = sum(cls._text_byte_quality(byte) > 0.9 for byte in chunk) / window
                if printable_ratio < 0.55:
                    end = start + offset
                    break

        while end > start and cls._text_byte_quality(data[end - 1]) < 0.9:
            end -= 1

        if end <= start:
            return start, -1e9

        payload = data[start:end]
        ascii_like = sum(cls._text_byte_quality(byte) > 0.9 for byte in payload) / len(payload)
        replacement_penalty = payload.decode("utf-8", errors="replace").count("�") / max(1, len(payload))
        newline_bonus = min(payload.count(b"\n"), 8) / 80.0
        length_bonus = min(len(payload), 4000) / 20000.0
        score = ascii_like + newline_bonus + length_bonus - 3.0 * replacement_penalty
        return end, float(score)

    @classmethod
    def salvage_text_payload(cls, decoded_bytes: bytes) -> TextSalvage | None:
        """Best-effort .txt extraction when the JOSS-F header is unrecoverable.

        Strategy:
        1. Prefer payload starts immediately after an early filename ending in
           '.txt', because the standard header stores the filename directly
           before the payload.
        2. Also try small generic header lengths, because the first filename
           bytes may be corrupted while the text itself is still intact.
        3. Pick the candidate whose following bytes look most like text.
        """
        if len(decoded_bytes) < 8:
            return None

        candidates: list[tuple[int, str]] = []
        search_limit = min(len(decoded_bytes), 512)
        lower_prefix = decoded_bytes[:search_limit].lower()
        pos = lower_prefix.find(b".txt")
        while pos != -1:
            candidates.append((pos + 4, "after recovered .txt filename marker"))
            pos = lower_prefix.find(b".txt", pos + 1)

        # Fallback: typical JOSS-F headers are [6 bytes] + short filename.
        # This also covers a corrupted extension or filename with no visible
        # '.txt' substring.
        for header_length in range(PacketCodec.MIN_HEADER_BYTES, min(96, len(decoded_bytes))):
            candidates.append((header_length, f"guessed header length {header_length}"))

        # Deduplicate while preserving order.
        seen: set[int] = set()
        unique_candidates: list[tuple[int, str]] = []
        for start, method in candidates:
            if start not in seen and 0 <= start < len(decoded_bytes):
                unique_candidates.append((start, method))
                seen.add(start)

        best: TextSalvage | None = None
        for start, method in unique_candidates:
            end, score = cls._trim_text_payload(decoded_bytes, start)
            if end - start < 12:
                continue
            raw_text = decoded_bytes[start:end].decode("utf-8", errors="replace")
            # Strip leading header debris if a guessed start landed slightly
            # early.  Keep normal spaces/tabs/newlines inside the text.
            text = raw_text.lstrip("\ufeff\x00\r\n\t ")
            if not text:
                continue
            # Prefer the standard-specific anchor when present: in JOSS-F, the
            # filename is immediately before the payload, so an early '.txt'
            # marker is much stronger evidence than a generic guessed offset.
            adjusted_score = score + (0.25 if method.startswith("after recovered .txt") else 0.0)
            candidate = TextSalvage(
                text=text,
                payload_start=start,
                payload_end=end,
                method=method,
                score=adjusted_score,
            )
            if best is None or candidate.score > best.score or (
                abs(candidate.score - best.score) < 1e-9 and len(candidate.text) > len(best.text)
            ):
                best = candidate

        # Avoid dumping nonsense when the decoded stream is mostly random.
        if best is None or best.score < 0.72:
            return None
        return best

    @staticmethod
    def packet_from_override(
        decoded_bytes: bytes,
        *,
        header_length: int,
        payload_bytes: int,
        filename: str,
    ) -> DecodedPacket:
        """Extract a payload when its boundaries are known externally."""
        if header_length < 0 or payload_bytes < 0:
            raise ValueError("override lengths must be non-negative")
        total = header_length + payload_bytes
        if total > len(decoded_bytes):
            raise ValueError(
                f"override requests {total} bytes from a {len(decoded_bytes)}-byte decoded stream"
            )
        return DecodedPacket(
            filename=filename,
            file_size=payload_bytes,
            payload=decoded_bytes[header_length:total],
            header_length=header_length,
        )

    def decode_signal(self, received: NDArray[np.float64]) -> tuple[DecodedPacket, ReceptionInfo]:
        """Strict API retained for callers that require a valid JOSS-F header."""
        raw = self.decode_raw_signal(received)
        packet = PacketCodec.parse(raw.decoded_bytes)
        return packet, raw.info


    def _save_diagnostics(
        self,
        diagnostics: ReceptionDiagnostics,
        directory: str | Path,
        *,
        show: bool,
    ) -> list[Path]:
        """Save a repeatable diagnostic dashboard and machine-readable data."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        c = self.config
        bins = c.data_bins
        frequencies = bins * c.sample_rate / c.fft_length

        timing_csv = directory / "timing_drift.csv"
        with timing_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "stage",
                    "block_index",
                    "is_pilot",
                    "timing_samples",
                    "phase_slope_rad_per_bin",
                    "phase_intercept_rad",
                    "fit_error_rad",
                    "weight",
                ]
            )
            for stage, observations in (
                ("before", diagnostics.timing_before),
                ("after", diagnostics.timing_after),
            ):
                for item in observations:
                    writer.writerow(
                        [
                            stage,
                            item.block_index,
                            int(item.is_pilot),
                            item.timing_samples,
                            item.phase_slope_rad_per_bin,
                            item.phase_intercept_rad,
                            item.fit_error_rad,
                            item.weight,
                        ]
                    )

        active_channel = diagnostics.channel[bins]
        channel_csv = directory / "channel_estimate.csv"
        with channel_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bin", "frequency_hz", "magnitude", "phase_rad"])
            for bin_index, frequency, value in zip(bins, frequencies, active_channel):
                writer.writerow(
                    [int(bin_index), float(frequency), float(abs(value)), float(np.angle(value))]
                )

        evm_csv = directory / "evm_by_data_block.csv"
        with evm_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ofdm_block_index", "evm_percent"])
            writer.writerows(
                zip(diagnostics.data_block_positions.tolist(), diagnostics.evm_percent.tolist())
            )

        summary_path = directory / "summary.json"
        summary = {
            "initial_chirp_sfo_ppm": diagnostics.initial_sfo_ppm,
            "payload_refinement_sfo_ppm": diagnostics.residual_sfo_ppm,
            "total_sfo_ppm": diagnostics.total_sfo_ppm,
            "data_start_sample_after_resampling": diagnostics.data_start,
            "median_evm_percent": float(np.median(diagnostics.evm_percent)),
            "p95_evm_percent": float(np.percentile(diagnostics.evm_percent, 95)),
            "timing_observations_before": len(diagnostics.timing_before),
            "timing_observations_after": len(diagnostics.timing_after),
            "note": (
                "All corrections are receiver-only. No JOSS-F waveform, pilot, carrier, "
                "mapping, interleaver, framing or LDPC behaviour is modified."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("diagnostic plots require: pip install matplotlib") from exc

        dashboard, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
        dashboard.suptitle("JOSS-F receiver: SFO, channel and decoding diagnostics")

        chirp_residual = diagnostics.chirp_positions - diagnostics.chirp_fitted_positions
        axes[0, 0].plot(diagnostics.chirp_numbers, chirp_residual, marker="o")
        axes[0, 0].axhline(0.0, linewidth=1)
        axes[0, 0].set_title(
            "Chirp clock fit\n"
            f"initial={diagnostics.initial_sfo_ppm:+.2f} ppm, "
            f"payload refinement={diagnostics.residual_sfo_ppm:+.2f} ppm"
        )
        axes[0, 0].set_xlabel("Chirp number")
        axes[0, 0].set_ylabel("Fit residual (samples)")
        axes[0, 0].grid(True, alpha=0.3)

        for label, observations, marker in (
            ("Without SFO correction", diagnostics.timing_before, "o"),
            ("After correction", diagnostics.timing_after, "x"),
        ):
            if observations:
                block = np.asarray([item.block_index for item in observations])
                timing = np.asarray([item.timing_samples for item in observations])
                timing -= float(np.median(timing))
                axes[0, 1].plot(block, timing, marker=marker, linestyle="none", label=label)
        axes[0, 1].set_title("Measured phase-ramp timing drift")
        axes[0, 1].set_xlabel("Absolute OFDM block")
        axes[0, 1].set_ylabel("Relative timing error (samples)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        before = diagnostics.constellation_before
        if len(before):
            axes[1, 0].scatter(before.real, before.imag, s=3, alpha=0.22)
        axes[1, 0].set_title("Constellation without SFO correction")
        axes[1, 0].set_xlabel("In-phase")
        axes[1, 0].set_ylabel("Quadrature")
        axes[1, 0].set_aspect("equal", adjustable="box")
        axes[1, 0].grid(True, alpha=0.3)

        after = diagnostics.constellation_after
        if len(after):
            axes[1, 1].scatter(after.real, after.imag, s=3, alpha=0.22)
        axes[1, 1].set_title("After SFO correction + pilot channel tracking")
        axes[1, 1].set_xlabel("In-phase")
        axes[1, 1].set_ylabel("Quadrature")
        axes[1, 1].set_aspect("equal", adjustable="box")
        axes[1, 1].grid(True, alpha=0.3)

        model_db = 20.0 * np.log10(np.abs(diagnostics.channel[bins]) + EPS)
        golay_db = 20.0 * np.log10(np.abs(diagnostics.golay_channel[bins]) + EPS)
        model_db -= float(np.median(model_db))
        golay_db -= float(np.median(golay_db))
        axes[2, 0].plot(frequencies / 1000.0, golay_db, label="Golay LS")
        axes[2, 0].plot(frequencies / 1000.0, model_db, label="Known-pilot model")
        axes[2, 0].set_title("Estimated channel frequency response")
        axes[2, 0].set_xlabel("Frequency (kHz)")
        axes[2, 0].set_ylabel("Relative magnitude (dB)")
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()

        axes[2, 1].plot(
            diagnostics.data_block_positions,
            diagnostics.evm_percent,
            marker=".",
        )
        axes[2, 1].set_title(
            f"Equalised data-symbol EVM (median {np.median(diagnostics.evm_percent):.1f}%)"
        )
        axes[2, 1].set_xlabel("Absolute OFDM block")
        axes[2, 1].set_ylabel("RMS EVM (%)")
        axes[2, 1].grid(True, alpha=0.3)

        dashboard_path = directory / "receiver_dashboard.png"
        dashboard.savefig(dashboard_path, dpi=170)

        impulse_figure, impulse_axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        impulse = np.fft.irfft(diagnostics.channel, n=c.fft_length)
        samples = np.arange(c.cyclic_prefix)
        impulse_axis.plot(samples / c.sample_rate * 1000.0, np.abs(impulse[: c.cyclic_prefix]))
        impulse_axis.set_title("Pilot-refined channel impulse response")
        impulse_axis.set_xlabel("Delay (ms)")
        impulse_axis.set_ylabel("Magnitude")
        impulse_axis.grid(True, alpha=0.3)
        impulse_path = directory / "channel_impulse_response.png"
        impulse_figure.savefig(impulse_path, dpi=170)

        if show:
            plt.show()
        else:
            plt.close(dashboard)
            plt.close(impulse_figure)

        return [dashboard_path, impulse_path, timing_csv, channel_csv, evm_csv, summary_path]

    def decode_wav(
        self,
        wav_path: str | Path,
        output_directory: str | Path = ".",
        *,
        strict_header: bool = False,
        forced_header_length: int | None = None,
        forced_payload_bytes: int | None = None,
        forced_filename: str = "recovered_payload.bin",
        diagnostics_directory: str | Path | None = None,
        save_diagnostics: bool = True,
        show_plots: bool = False,
    ) -> Path:
        wav_path = Path(wav_path)
        received = read_wav(wav_path, self.config)
        peak = float(np.max(np.abs(received))) if len(received) else 0.0
        clipped = float(np.mean(np.abs(received) >= 0.995)) if len(received) else 0.0
        raw = self.decode_raw_signal(
            received,
            collect_diagnostics=save_diagnostics or show_plots,
        )

        if (forced_header_length is None) != (forced_payload_bytes is None):
            raise ValueError(
                "forced_header_length and forced_payload_bytes must be supplied together"
            )

        if forced_header_length is not None and forced_payload_bytes is not None:
            packet = self.packet_from_override(
                raw.decoded_bytes,
                header_length=forced_header_length,
                payload_bytes=forced_payload_bytes,
                filename=forced_filename,
            )
            recovery: HeaderRecovery | None = HeaderRecovery(
                packet, "manual boundary override"
            )
        else:
            recovery = self.recover_packet_header(raw.decoded_bytes)

        print(f"Decoded physical layer from {wav_path}")
        print(f"  input peak       : {peak:.3f} ({100.0 * clipped:.3f}% clipped)")
        print(f"  sync sample      : {raw.info.sync_sample}")
        print(f"  chirp SFO        : {raw.info.initial_sample_rate_offset_ppm:+.2f} ppm")
        print(f"  payload SFO fix  : {raw.info.residual_sample_rate_offset_ppm:+.2f} ppm")
        print(f"  total SFO        : {raw.info.sample_rate_offset_ppm:+.2f} ppm")
        print(f"  OFDM adjustment  : {raw.info.data_start_adjustment:+d} samples")
        print(f"  known pilots     : {raw.info.pilot_ofdm_blocks}")
        print(f"  median EVM       : {raw.info.median_evm_percent:.2f}%")
        print(
            f"  LDPC success     : {raw.info.successful_ldpc_blocks}/"
            f"{raw.info.total_ldpc_blocks}"
        )
        print(f"  decoded stream   : {len(raw.decoded_bytes)} bytes")
        print(f"  LDPC fingerprint : {self.ldpc.compatibility_fingerprint()[:16]}")
        print(f"  pilot fingerprint: {self.known_pilot.content_sha256[:16]}")

        diagnostic_paths: list[Path] = []
        if raw.diagnostics is not None and (save_diagnostics or show_plots):
            root = (
                Path(diagnostics_directory)
                if diagnostics_directory is not None
                else Path(output_directory).parent / "diagnostics"
            )
            run_directory = root / wav_path.stem
            diagnostic_paths = self._save_diagnostics(
                raw.diagnostics,
                run_directory,
                show=show_plots,
            )
            print(f"  diagnostics      : {run_directory}")

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        if recovery is not None:
            packet = recovery.packet
            output_path = output_directory / safe_output_name(packet.filename)
            output_path.write_bytes(packet.payload)
            print(f"  header status    : {recovery.method}")
            if recovery.flipped_fixed_header_bits:
                print(f"  repaired bits    : {recovery.flipped_fixed_header_bits}")
            print(f"  original name    : {packet.filename}")
            print(f"  file size        : {packet.file_size} bytes")
            print(f"  saved            : {output_path}")
            return output_path

        if strict_header:
            raise ValueError(
                "JOSS-F header could not be decoded and strict-header mode is enabled. "
                f"The physical layer recovered {len(raw.decoded_bytes)} bytes."
            )

        wav_stem = wav_path.stem
        output_path = output_directory / f"received_{wav_stem}_raw_ldpc_stream.bin"
        output_path.write_bytes(raw.decoded_bytes)

        text_salvage = self.salvage_text_payload(raw.decoded_bytes)
        text_path: Path | None = None
        if text_salvage is not None:
            text_path = output_directory / f"received_{wav_stem}_best_effort.txt"
            text_path.write_text(text_salvage.text, encoding="utf-8", errors="replace")

        metadata_path = output_path.with_suffix(output_path.suffix + ".json")
        metadata = {
            "status": "header_unrecoverable",
            "decoded_stream_bytes": len(raw.decoded_bytes),
            "first_64_bytes_hex": raw.decoded_bytes[:64].hex(),
            "sync_sample": raw.info.sync_sample,
            "initial_sample_rate_offset_ppm": raw.info.initial_sample_rate_offset_ppm,
            "residual_sample_rate_offset_ppm": raw.info.residual_sample_rate_offset_ppm,
            "total_sample_rate_offset_ppm": raw.info.sample_rate_offset_ppm,
            "data_start_adjustment": raw.info.data_start_adjustment,
            "median_evm_percent": raw.info.median_evm_percent,
            "successful_ldpc_blocks": raw.info.successful_ldpc_blocks,
            "total_ldpc_blocks": raw.info.total_ldpc_blocks,
            "ldpc_fingerprint": self.ldpc.compatibility_fingerprint(),
            "pilot_fingerprint": self.known_pilot.content_sha256,
            "diagnostic_files": [str(path) for path in diagnostic_paths],
            "text_salvage": None
            if text_salvage is None
            else {
                "path": str(text_path),
                "payload_start_guess": text_salvage.payload_start,
                "payload_end_guess": text_salvage.payload_end,
                "method": text_salvage.method,
                "score": text_salvage.score,
            },
            "note": (
                "The JOSS-F header supplies both payload offset and payload length. "
                "Without a recoverable header, the exact payload boundary is not uniquely "
                "determined; the raw file preserves every LDPC-decoded byte, including header "
                "and post-payload padding."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print("  header status    : unrecoverable; raw decoded stream preserved")
        if text_salvage is not None and text_path is not None:
            print("  text salvage     : best-effort .txt extracted")
            print(f"  text bytes guess : {text_salvage.payload_start}..{text_salvage.payload_end}")
            print(f"  text method      : {text_salvage.method}")
            print(f"  saved text       : {text_path}")
        else:
            print("  text salvage     : no plausible text payload found")
        print(f"  saved raw stream : {output_path}")
        print(f"  saved metadata   : {metadata_path}")
        return text_path if text_path is not None else output_path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = BASE_DIR / "received"
DIAGNOSTICS_DIRECTORY = BASE_DIR / "diagnostics"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode a JOSS-F WAV recording")
    parser.add_argument("wav", nargs="?", help="input WAV path; prompts when omitted")
    parser.add_argument(
        "--strict-header",
        action="store_true",
        help="raise instead of preserving the raw LDPC stream when the header fails",
    )
    parser.add_argument("--header-bytes", type=int, help="known payload offset in decoded bytes")
    parser.add_argument("--payload-bytes", type=int, help="known payload length in bytes")
    parser.add_argument(
        "--filename",
        default="recovered_payload.bin",
        help="output filename used with manual boundary overrides",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="decoded-file directory (default: ./received)",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=DIAGNOSTICS_DIRECTORY,
        help="plot/data root directory (default: ./diagnostics)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="save diagnostics without opening plot windows",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="disable diagnostic plots and CSV/JSON output",
    )
    return parser.parse_args()


def resolve_input_wav(argument: str | None) -> Path:
    if argument is None:
        return ask_input_wav()
    path = Path(argument).expanduser()
    if path.suffix.lower() != ".wav":
        path = path.with_suffix(".wav")
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    return path


if __name__ == "__main__":
    args = parse_args()
    Receiver().decode_wav(
        resolve_input_wav(args.wav),
        args.output_dir,
        strict_header=args.strict_header,
        forced_header_length=args.header_bytes,
        forced_payload_bytes=args.payload_bytes,
        forced_filename=args.filename,
        diagnostics_directory=args.diagnostics_dir,
        save_diagnostics=not args.no_diagnostics,
        show_plots=not args.no_plots and not args.no_diagnostics,
    )
