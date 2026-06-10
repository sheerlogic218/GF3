"""Robust JOSS-F receiver.

The transmitter-facing standard is unchanged.  Receiver-only improvements:
* absolute-index chirp fitting with Golay disambiguation and robust SFO fit;
* band-limited arbitrary-ratio resampling;
* robust multi-repeat Golay estimation;
* quality-gated, blended and interpolated OFDM-pilot channel tracking;
* conservative residual phase/timing correction and noise-aware soft metrics;
* LDPC-guided rescue over QPSK rotations and small timing hypotheses;
* complete raw-stream/header/text-salvage behaviour from the previous receiver.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import argparse
import json

import numpy as np
from numpy.typing import NDArray
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

# Receiver-only numerical constants.
_SYNC_HUBER_C = 1.5
_RESAMPLE_CHUNK = 8_192


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


@dataclass(frozen=True)
class RawReception:
    decoded_bytes: bytes
    info: ReceptionInfo


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


@dataclass(frozen=True)
class PilotObservation:
    position: int
    channel: NDArray[np.complex128]
    raw_channel: NDArray[np.complex128]
    snr_db: float
    blend_weight: float
    relative_jump: float
    incremental_delay_samples: float
    incremental_phase_radians: float


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

    # ──────────────────────────────────────────────────────────────────────────
    # Synchronisation & SFO correction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalised_match_metric(
        signal: NDArray[np.float64],
        reference: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Energy-normalised matched-filter magnitude in ``valid`` coordinates."""
        signal = np.asarray(signal, dtype=float)
        reference = np.asarray(reference, dtype=float)
        numerator = fftconvolve(signal, reference[::-1], mode="valid")
        local_energy = fftconvolve(
            signal * signal,
            np.ones(len(reference), dtype=float),
            mode="valid",
        )
        denominator = np.sqrt(
            np.maximum(local_energy, 0.0) * float(np.dot(reference, reference))
        )
        return np.abs(numerator) / (denominator + EPS)

    @staticmethod
    def _robust_line_fit(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        base_weights: NDArray[np.float64] | None = None,
        *,
        iterations: int = 6,
    ) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
        """Huber IRLS fit of ``y = slope*x + intercept``."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if base_weights is None:
            base = np.ones_like(x)
        else:
            base = np.maximum(np.asarray(base_weights, dtype=float), EPS)
        design = np.column_stack([x, np.ones_like(x)])
        weights = base.copy()
        beta = np.array([0.0, float(np.median(y))])
        for _ in range(iterations):
            root = np.sqrt(weights)
            beta = np.linalg.lstsq(
                design * root[:, None], y * root, rcond=None
            )[0]
            residual = y - design @ beta
            centred = residual - float(np.median(residual))
            scale = max(0.25, 1.4826 * float(np.median(np.abs(centred))))
            huber = np.minimum(1.0, (_SYNC_HUBER_C * scale) / (np.abs(centred) + EPS))
            weights = base * huber
        residual = y - design @ beta
        return float(beta[0]), float(beta[1]), residual, weights

    @staticmethod
    def _quadratic_peak(metric: NDArray[np.float64], peak: int) -> float:
        if not (0 < peak < len(metric) - 1):
            return float(peak)
        left, centre, right = metric[peak - 1], metric[peak], metric[peak + 1]
        denominator = left - 2.0 * centre + right
        if abs(denominator) < EPS:
            return float(peak)
        delta = 0.5 * (left - right) / denominator
        return float(peak) + float(np.clip(delta, -0.5, 0.5))

    @staticmethod
    def _normalised_dot(
        signal: NDArray[np.float64],
        start: int,
        reference: NDArray[np.float64],
    ) -> float:
        if start < 0 or start + len(reference) > len(signal):
            return 0.0
        window = signal[start : start + len(reference)]
        denominator = float(np.linalg.norm(window) * np.linalg.norm(reference)) + EPS
        return float(abs(np.dot(window, reference)) / denominator)

    @staticmethod
    def _stretch_reference(
        reference: NDArray[np.float64], ratio: float
    ) -> NDArray[np.float64]:
        if abs(ratio - 1.0) < 1e-10:
            return np.asarray(reference, dtype=float)
        length = max(2, int(round(len(reference) * ratio)))
        source_positions = np.arange(length, dtype=float) / ratio
        return np.interp(
            source_positions,
            np.arange(len(reference), dtype=float),
            reference,
            left=0.0,
            right=0.0,
        )

    def _golay_alignment_score(
        self,
        received: NDArray[np.float64],
        intercept: float,
        ratio: float,
    ) -> float:
        """Use known post-chirp Golay positions to resolve repeated-chirp ambiguity."""
        c = self.config
        references = (
            self._stretch_reference(c.golay_amplitude * self.golay.a, ratio),
            self._stretch_reference(c.golay_amplitude * self.golay.b, ratio),
        )
        scores: list[float] = []
        # Score all four A/B pairs.  This is important because three chirp
        # lengths equal one complete Golay repeat; using only early pairs would
        # leave a 12,288-sample ambiguity when the first three chirps vanish.
        for a_relative, b_relative in self.golay.pulse_starts:
            for relative, reference in zip((a_relative, b_relative), references):
                expected = intercept + ratio * (c.chirp_train_length + relative)
                centre = int(round(expected))
                best = 0.0
                for shift in range(-12, 13, 3):
                    best = max(best, self._normalised_dot(received, centre + shift, reference))
                scores.append(best)
        return float(np.mean(scores)) if scores else 0.0

    def _collect_chirp_observations(
        self,
        metric: NDArray[np.float64],
        intercept: float,
        period: float,
        threshold: float,
        radius: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        indices: list[float] = []
        positions: list[float] = []
        strengths: list[float] = []
        for chirp_index in range(self.config.chirp_count):
            expected = intercept + chirp_index * period
            lo = max(0, int(np.floor(expected - radius)))
            hi = min(len(metric), int(np.ceil(expected + radius + 1)))
            if hi <= lo:
                continue
            local = metric[lo:hi]
            peak = lo + int(np.argmax(local))
            strength = float(metric[peak])
            if strength < threshold:
                continue
            indices.append(float(chirp_index))
            positions.append(self._quadratic_peak(metric, peak))
            strengths.append(strength)
        return (
            np.asarray(indices, dtype=float),
            np.asarray(positions, dtype=float),
            np.asarray(strengths, dtype=float),
        )

    def _bandlimited_resample(
        self,
        received: NDArray[np.float64],
        intercept: float,
        ratio: float,
    ) -> NDArray[np.float64]:
        """Windowed-sinc arbitrary-ratio resampling from raw to nominal samples."""
        if abs(ratio - 1.0) < 1e-10 and abs(intercept - round(intercept)) < 1e-8:
            return received[int(round(intercept)) :].copy()

        half = int(self.config.resampler_half_width)
        if half < 4:
            raise ValueError("resampler_half_width must be at least four")
        output_length = int(np.floor((len(received) - intercept - 1.0) / ratio)) + 1
        if output_length <= 0:
            raise ValueError("SFO correction leaves no output samples")

        offsets = np.arange(-half + 1, half + 1, dtype=np.int64)
        cutoff = min(0.98, 0.98 / max(1.0, ratio))
        beta = 8.6
        i0_beta = float(np.i0(beta))
        corrected = np.empty(output_length, dtype=float)

        for out_lo in range(0, output_length, _RESAMPLE_CHUNK):
            out_hi = min(output_length, out_lo + _RESAMPLE_CHUNK)
            positions = intercept + np.arange(out_lo, out_hi, dtype=float) * ratio
            centres = np.floor(positions).astype(np.int64)
            sample_indices = centres[:, None] + offsets[None, :]
            distances = positions[:, None] - sample_indices
            support = np.abs(distances) < half
            argument = np.clip(1.0 - (distances / half) ** 2, 0.0, 1.0)
            window = np.i0(beta * np.sqrt(argument)) / i0_beta
            taps = cutoff * np.sinc(cutoff * distances) * window * support

            valid = (sample_indices >= 0) & (sample_indices < len(received))
            clipped_indices = np.clip(sample_indices, 0, len(received) - 1)
            taps *= valid
            taps /= np.sum(taps, axis=1, keepdims=True) + EPS
            corrected[out_lo:out_hi] = np.sum(
                received[clipped_indices] * taps, axis=1
            )
        return corrected

    def synchronise_and_correct(
        self,
        received: NDArray[np.float64],
        *,
        _diag: dict | None = None,
    ) -> tuple[NDArray[np.float64], int, float]:
        """Locate the absolute preamble origin and remove constant SFO.

        Every chirp is assigned its true standard index 0..9.  The following
        Golay section resolves the otherwise unavoidable one-chirp ambiguity
        when the beginning of the train is missing or attenuated.
        """
        received = np.asarray(received, dtype=float)
        chirp = self.chirp.samples()
        n = self.config.chirp_length
        if len(received) < self.config.chirp_train_length:
            raise ValueError("recording is shorter than the JOSS-F chirp train")

        metric = self._normalised_match_metric(received, chirp)
        median = float(np.median(metric))
        sigma = max(EPS, 1.4826 * float(np.median(np.abs(metric - median))))
        maximum = float(np.max(metric))
        threshold = max(median + 5.0 * sigma, 0.12 * maximum)
        prominence = max(3.0 * sigma, 0.025 * maximum, EPS)
        peaks, _ = find_peaks(
            metric,
            height=threshold,
            prominence=prominence,
            distance=max(1, int(0.50 * n)),
        )
        if metric[0] >= threshold:
            peaks = np.unique(np.concatenate([np.array([0], dtype=int), peaks]))
        argmax = int(np.argmax(metric))
        peaks = np.unique(np.concatenate([peaks.astype(int), np.array([argmax])]))

        # Keep the strongest candidates.  Each observed peak can represent any
        # of the ten standard chirp indices; Golay scoring fixes the absolute one.
        ranked = peaks[np.argsort(metric[peaks])[::-1]][:40]
        candidate_starts: set[int] = set()
        for peak in ranked:
            for chirp_index in range(self.config.chirp_count):
                candidate = int(round(int(peak) - chirp_index * n))
                if -self.config.sync_peak_search_radius <= candidate < len(metric):
                    candidate_starts.add(candidate)

        best: dict[str, object] | None = None
        coarse_radius = int(self.config.sync_peak_search_radius)
        fine_radius = max(32, coarse_radius // 3)
        max_ratio_error = self.config.sync_max_sfo_ppm * 1e-6

        for initial_start in candidate_starts:
            indices, observed, strengths = self._collect_chirp_observations(
                metric,
                float(initial_start),
                float(n),
                threshold,
                coarse_radius,
            )
            if len(indices) < self.config.sync_min_chirps:
                continue

            period, intercept, residual, fit_weights = self._robust_line_fit(
                indices, observed, strengths * strengths
            )
            indices, observed, strengths = self._collect_chirp_observations(
                metric, intercept, period, threshold, fine_radius
            )
            if len(indices) < self.config.sync_min_chirps:
                continue
            period, intercept, residual, fit_weights = self._robust_line_fit(
                indices, observed, strengths * strengths
            )

            ratio = period / n
            if not (1.0 - max_ratio_error <= ratio <= 1.0 + max_ratio_error):
                continue
            centred = residual - float(np.median(residual))
            robust_scale = max(0.25, 1.4826 * float(np.median(np.abs(centred))))
            residual_limit = max(
                float(self.config.sync_max_fit_residual), 3.5 * robust_scale
            )
            inliers = np.abs(residual) <= residual_limit
            if int(np.sum(inliers)) < self.config.sync_min_chirps:
                continue

            indices = indices[inliers]
            observed = observed[inliers]
            strengths = strengths[inliers]
            period, intercept, residual, fit_weights = self._robust_line_fit(
                indices, observed, strengths * strengths
            )
            ratio = period / n
            span = float(np.max(indices) - np.min(indices))
            if span < self.config.sync_min_chirps - 1:
                continue
            rms = float(np.sqrt(np.average(residual * residual, weights=fit_weights)))
            golay_score = self._golay_alignment_score(received, intercept, ratio)
            score = (
                4.0 * len(indices)
                + 0.5 * span
                + 10.0 * float(np.mean(strengths))
                + 24.0 * golay_score
                - 0.35 * rms
            )
            candidate = dict(
                score=score,
                indices=indices,
                observed=observed,
                strengths=strengths,
                period=period,
                intercept=intercept,
                residual=residual,
                ratio=ratio,
                rms=rms,
                golay_score=golay_score,
            )
            if best is None or float(candidate["score"]) > float(best["score"]):
                best = candidate

        if best is None:
            template = self.chirp.train()
            coarse_metric = self._normalised_match_metric(received, template)
            sync = int(np.argmax(coarse_metric))
            warnings.warn(
                "[JOSS-F sync] robust indexed chirp fit failed; falling back to "
                "full-train timing with no SFO correction.",
                RuntimeWarning,
                stacklevel=2,
            )
            corrected = received[sync:].copy()
            if _diag is not None:
                _diag.update(dict(
                    received=received,
                    corrected=corrected,
                    chirp_metric=metric,
                    all_peaks=peaks,
                    best_chain=[],
                    chirp_indices=np.array([], dtype=float),
                    chirp_observed=np.array([], dtype=float),
                    chirp_fitted=np.array([], dtype=float),
                    chirp_residuals=np.array([], dtype=float),
                    sfo_ratio=1.0,
                    sfo_ppm=0.0,
                    sync_sample=sync,
                    sync_fit_rms=np.nan,
                    golay_sync_score=0.0,
                    min_chain_required=self.config.sync_min_chirps,
                ))
            return corrected, sync, 0.0

        indices = np.asarray(best["indices"], dtype=float)
        observed = np.asarray(best["observed"], dtype=float)
        period = float(best["period"])
        intercept = float(best["intercept"])
        ratio = float(best["ratio"])
        fitted = intercept + period * indices
        residual = observed - fitted
        corrected = self._bandlimited_resample(received, intercept, ratio)
        sync = int(round(intercept))
        offset_ppm = (ratio - 1.0) * 1e6

        if _diag is not None:
            _diag.update(dict(
                received=received,
                corrected=corrected,
                chirp_metric=metric,
                all_peaks=peaks,
                best_chain=np.rint(observed).astype(int).tolist(),
                chirp_indices=indices,
                chirp_observed=observed,
                chirp_fitted=fitted,
                chirp_residuals=residual,
                sfo_ratio=ratio,
                sfo_ppm=offset_ppm,
                sync_sample=sync,
                sync_fit_rms=float(best["rms"]),
                golay_sync_score=float(best["golay_score"]),
                min_chain_required=self.config.sync_min_chirps,
            ))

        return corrected, sync, offset_ppm

    # ──────────────────────────────────────────────────────────────────────────
    # Golay channel estimation
    # ──────────────────────────────────────────────────────────────────────────

    def _golay_channel(
        self,
        received: NDArray[np.float64],
        *,
        _diag: dict | None = None,
    ) -> NDArray[np.complex128]:
        """Robustly combine the four official Golay A/B channel estimates."""
        c = self.config
        transform_length = c.fft_length + c.golay_gap
        golay_start = c.chirp_train_length

        x_a = np.pad(c.golay_amplitude * self.golay.a, (0, c.golay_gap))
        x_b = np.pad(c.golay_amplitude * self.golay.b, (0, c.golay_gap))
        x_a_f = np.fft.rfft(x_a, n=transform_length)
        x_b_f = np.fft.rfft(x_b, n=transform_length)
        pair_channels: list[NDArray[np.complex128]] = []

        for a_relative, b_relative in self.golay.pulse_starts:
            numerator = np.zeros(transform_length // 2 + 1, dtype=np.complex128)
            denominator = np.zeros(transform_length // 2 + 1, dtype=float)
            for relative, reference_f in ((a_relative, x_a_f), (b_relative, x_b_f)):
                start = golay_start + relative
                window = received[start : start + transform_length]
                if len(window) < transform_length:
                    window = np.pad(window, (0, transform_length - len(window)))
                received_f = np.fft.rfft(window, n=transform_length)
                numerator += received_f * np.conj(reference_f)
                denominator += np.abs(reference_f) ** 2

            regularisation = c.golay_regularisation * float(np.max(denominator))
            pair_impulse = np.fft.irfft(
                numerator / (denominator + regularisation),
                n=transform_length,
            )
            pair_impulse = pair_impulse[: c.cyclic_prefix]
            pair_channels.append(np.fft.rfft(pair_impulse, n=c.fft_length))

        stack = np.stack(pair_channels)
        preliminary = np.median(stack.real, axis=0) + 1j * np.median(stack.imag, axis=0)
        bins = c.data_bins
        scale = float(np.median(np.abs(preliminary[bins]))) + EPS
        distances = np.array(
            [float(np.median(np.abs(channel[bins] - preliminary[bins]))) / scale
             for channel in stack],
            dtype=float,
        )
        pair_weights = 1.0 / (0.05 + distances * distances)
        pair_weights /= float(np.sum(pair_weights))
        channel = np.sum(stack * pair_weights[:, None], axis=0)

        # Project the final estimate onto a physically admissible real impulse
        # response no longer than the cyclic prefix.
        impulse = np.fft.irfft(channel, n=c.fft_length)
        impulse[c.cyclic_prefix :] = 0.0
        channel = np.fft.rfft(impulse, n=c.fft_length)

        if _diag is not None:
            _diag["golay_channel"] = channel
            _diag["golay_pair_channels"] = pair_channels
            _diag["golay_pair_weights"] = pair_weights
            _diag["golay_impulse"] = impulse

        return channel

    # ──────────────────────────────────────────────────────────────────────────
    # CP self-correlation for data-start refinement
    # ──────────────────────────────────────────────────────────────────────────

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

    def _refine_data_start(
        self,
        received: NDArray[np.float64],
        nominal: int,
        *,
        _diag: dict | None = None,
    ) -> tuple[int, int]:
        """Refine the first OFDM boundary using cyclic-prefix self-correlation."""
        radius = self.config.data_start_refine_radius

        # Evaluate all coarse offsets (step 4) and store for the diagnostic plot.
        coarse_offsets = list(range(-radius, radius + 1, 4))
        coarse_scores  = [self._cp_score(received, nominal + off)
                          for off in coarse_offsets]
        best_coarse_idx = int(np.argmax(coarse_scores))
        coarse_best     = coarse_offsets[best_coarse_idx]

        fine_offsets = range(max(-radius, coarse_best - 5),
                             min(radius,  coarse_best + 5) + 1)
        best = max(fine_offsets,
                   key=lambda off: self._cp_score(received, nominal + off))

        if _diag is not None:
            _diag["cp_offsets"]     = np.array(coarse_offsets)
            _diag["cp_scores"]      = np.array(coarse_scores)
            _diag["cp_best_offset"] = best

        return nominal + best, int(best)

    # ──────────────────────────────────────────────────────────────────────────
    # Channel tracking, equalisation and soft metrics
    # ──────────────────────────────────────────────────────────────────────────

    def _noise_power_from_unused_bins(
        self, spectrum: NDArray[np.complex128]
    ) -> float:
        c = self.config
        mask = np.ones(len(spectrum), dtype=bool)
        guard_lo = max(0, c.first_data_bin - 12)
        guard_hi = min(len(spectrum), c.last_data_bin + 13)
        mask[guard_lo:guard_hi] = False
        mask[:8] = False
        # Very high audio bins can contain device noise-shaping; a robust median
        # over the remaining null bins limits their influence.
        powers = np.abs(spectrum[mask]) ** 2
        if not len(powers):
            return EPS
        return max(EPS, float(np.median(powers)) / np.log(2.0))

    def _project_channel(
        self, channel: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Project a frequency response onto a real channel within the CP."""
        c = self.config
        impulse = np.fft.irfft(channel, n=c.fft_length)
        impulse[c.cyclic_prefix :] = 0.0
        return np.fft.rfft(impulse, n=c.fft_length)

    def _relative_phase_line(
        self,
        current: NDArray[np.complex128],
        reference: NDArray[np.complex128],
    ) -> tuple[float, float, float]:
        """Return incremental delay, centre-bin phase and robust fit RMS."""
        bins = self.config.data_bins
        current_b = current[bins]
        reference_b = reference[bins]
        magnitude = np.abs(current_b) * np.abs(reference_b)
        threshold = float(np.percentile(magnitude, 35.0))
        valid = magnitude > max(threshold, EPS)
        if int(np.sum(valid)) < 32:
            return 0.0, 0.0, np.inf
        x = bins[valid].astype(float) - float(np.mean(bins[valid]))
        phase = np.unwrap(np.angle(current_b[valid] * np.conj(reference_b[valid])))
        slope, intercept, residual, weights = self._robust_line_fit(
            x, phase, magnitude[valid]
        )
        delay = -slope * self.config.fft_length / (2.0 * np.pi)
        rms = float(np.sqrt(np.average(residual * residual, weights=weights)))
        return float(delay), float(intercept), rms

    def _equalise_with_weights(
        self,
        spectrum: NDArray[np.complex128],
        channel: NDArray[np.complex128],
        noise_power: float,
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], dict[str, float | bool]]:
        bins = self.config.data_bins
        channel_b = channel[bins]
        power = np.abs(channel_b) ** 2
        reference_power = float(np.median(power)) + EPS
        regularisation = max(
            self.config.equaliser_regularisation * reference_power,
            noise_power / max(self.config.ofdm_scale ** 2, EPS),
        )
        estimate = spectrum[bins] * np.conj(channel_b) / (power + regularisation)

        # A data-directed correction is only permitted inside a deliberately
        # small basin.  Larger QPSK rotations are ambiguous and are handled by
        # the LDPC-guided group rescue below rather than silently relabelling bits.
        hard = QPSK.hard_symbols(estimate)
        distance = np.abs(estimate - hard)
        reliable = (distance < 0.85) & (power > 0.15 * reference_power)
        phase_applied = False
        fitted_phase = 0.0
        fitted_delay = 0.0
        fit_rms = np.inf
        if int(np.sum(reliable)) >= max(60, len(bins) // 3):
            x = bins[reliable].astype(float) - float(np.mean(bins))
            phase_error = np.unwrap(
                np.angle(estimate[reliable] * np.conj(hard[reliable]))
            )
            slope, intercept, residual, fit_weights = self._robust_line_fit(
                x,
                phase_error,
                np.clip(np.abs(estimate[reliable]), 0.1, 4.0),
            )
            fitted_delay = -slope * self.config.fft_length / (2.0 * np.pi)
            fitted_phase = intercept
            fit_rms = float(
                np.sqrt(np.average(residual * residual, weights=fit_weights))
            )
            phase_limit = np.deg2rad(self.config.decision_phase_limit_deg)
            if (
                abs(fitted_phase) <= phase_limit
                and abs(fitted_delay) <= self.config.decision_delay_limit_samples
                and fit_rms <= 0.32
            ):
                x_all = bins.astype(float) - float(np.mean(bins))
                estimate *= np.exp(-1j * (slope * x_all + intercept))
                phase_applied = True

        robust_magnitude = float(np.median(np.abs(estimate)))
        gain = 1.0
        if robust_magnitude > EPS:
            gain = np.sqrt(2.0) / robust_magnitude
            estimate *= gain

        # Post-equalisation confidence.  The absolute LLR scale is regularised
        # by measured decision residual while carrier-to-carrier confidence is
        # driven by measured null-bin noise and channel gain.
        hard_after = QPSK.hard_symbols(estimate)
        decision_mse = float(np.median(np.abs(estimate - hard_after) ** 2))
        carrier_snr = power / (noise_power + regularisation + EPS)
        carrier_snr /= float(np.median(carrier_snr) + EPS)
        common = 2.0 / max(decision_mse, 0.04)
        weights = np.clip(common * carrier_snr, 0.01, 50.0)
        metadata: dict[str, float | bool] = {
            "phase_correction_applied": phase_applied,
            "fitted_phase_radians": float(fitted_phase),
            "fitted_delay_samples": float(fitted_delay),
            "fit_rms": float(fit_rms),
            "decision_mse": decision_mse,
            "noise_power": float(noise_power),
        }
        return estimate, weights.astype(float), metadata

    def _equalise(
        self,
        spectrum: NDArray[np.complex128],
        channel: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        """Compatibility wrapper used by external diagnostics/tests."""
        estimate, _, _ = self._equalise_with_weights(
            spectrum, channel, self._noise_power_from_unused_bins(spectrum)
        )
        return estimate

    def _pilot_channel(
        self,
        position: int,
        spectrum: NDArray[np.complex128],
        fallback: NDArray[np.complex128],
    ) -> PilotObservation:
        known = self.known_pilot.transmitted_frequency
        bins = self.config.data_bins
        raw = fallback.copy()
        valid = np.abs(known[bins]) > EPS
        raw_bins = spectrum[bins[valid]] / known[bins[valid]]
        raw[bins[valid]] = raw_bins
        raw = self._project_channel(raw)

        noise_power = self._noise_power_from_unused_bins(spectrum)
        signal_power = float(np.median(np.abs(spectrum[bins]) ** 2))
        snr_db = 10.0 * np.log10((signal_power + EPS) / (noise_power + EPS))

        fallback_b = fallback[bins]
        raw_b = raw[bins]
        denominator = np.vdot(fallback_b, fallback_b)
        scalar = 1.0 + 0.0j if abs(denominator) < EPS else np.vdot(fallback_b, raw_b) / denominator
        aligned = scalar * fallback_b
        relative_jump = float(np.median(np.abs(raw_b - aligned))) / (
            float(np.median(np.abs(raw_b))) + EPS
        )

        if snr_db < self.config.pilot_min_snr_db:
            blend = 0.0
        else:
            snr_quality = float(np.clip(
                (snr_db - self.config.pilot_min_snr_db) / 15.0, 0.15, 1.0
            ))
            jump_quality = min(
                1.0,
                self.config.pilot_max_relative_jump / max(relative_jump, EPS),
            )
            blend = float(self.config.pilot_update_weight * snr_quality * jump_quality)

        updated = fallback.copy()
        updated[bins] = (1.0 - blend) * fallback_b + blend * raw_b
        updated = self._project_channel(updated)
        delay, phase, _ = self._relative_phase_line(raw, fallback)
        return PilotObservation(
            position=position,
            channel=updated,
            raw_channel=raw,
            snr_db=float(snr_db),
            blend_weight=blend,
            relative_jump=relative_jump,
            incremental_delay_samples=delay,
            incremental_phase_radians=phase,
        )

    @staticmethod
    def _interpolate_channel(
        position: int,
        anchor_positions: NDArray[np.int64],
        anchor_channels: list[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        if position <= int(anchor_positions[0]):
            return anchor_channels[0]
        if position >= int(anchor_positions[-1]):
            return anchor_channels[-1]
        right_index = int(np.searchsorted(anchor_positions, position, side="right"))
        left_index = right_index - 1
        left_position = int(anchor_positions[left_index])
        right_position = int(anchor_positions[right_index])
        fraction = (position - left_position) / (right_position - left_position)
        return (
            (1.0 - fraction) * anchor_channels[left_index]
            + fraction * anchor_channels[right_index]
        )

    def _extract_data_rows(
        self,
        received: NDArray[np.float64],
        data_start: int,
        golay_channel: NDArray[np.complex128],
        *,
        _diag: dict | None = None,
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
                    data_start + index * symbol_length
                    : data_start + (index + 1) * symbol_length
                ]
            )
            for index in range(block_count)
        ]

        # Sequentially filter pilot observations, then use the offline nature of
        # the modem to interpolate every data block between bracketing anchors.
        observations: dict[int, PilotObservation] = {}
        previous = golay_channel
        for index, spectrum in enumerate(spectra, start=1):
            if not self.scheduler.is_pilot(index):
                continue
            observation = self._pilot_channel(index, spectrum, previous)
            observations[index] = observation
            previous = observation.channel

        anchor_positions = np.asarray([0, *sorted(observations)], dtype=np.int64)
        anchor_channels = [golay_channel] + [
            observations[position].channel for position in sorted(observations)
        ]

        rows: list[NDArray[np.complex128]] = []
        weights: list[NDArray[np.float64]] = []
        raw_spectra_bins: list[NDArray[np.complex128]] = []
        eq_rows_diag: list[NDArray[np.complex128]] = []
        tracked_channels: dict[int, NDArray[np.complex128]] = {}
        equaliser_metadata: list[dict[str, float | bool]] = []
        row_noise_powers: list[float] = []
        n_diag = 8

        for one_based, spectrum in enumerate(spectra, start=1):
            if self.scheduler.is_pilot(one_based):
                continue
            channel = self._interpolate_channel(
                one_based, anchor_positions, anchor_channels
            )
            noise_power = self._noise_power_from_unused_bins(spectrum)
            estimate, row_weights, metadata = self._equalise_with_weights(
                spectrum, channel, noise_power
            )
            rows.append(estimate)
            weights.append(row_weights)
            row_noise_powers.append(noise_power)
            equaliser_metadata.append(metadata)

            if _diag is not None and len(raw_spectra_bins) < n_diag:
                raw_spectra_bins.append(spectrum[c.data_bins].copy())
                eq_rows_diag.append(estimate.copy())
                tracked_channels[one_based] = channel.copy()

        if not rows:
            raise ValueError("no JOSS-F data OFDM blocks were found")

        if _diag is not None:
            _diag["pilot_channels"] = {
                position: observation.channel
                for position, observation in observations.items()
            }
            _diag["pilot_raw_channels"] = {
                position: observation.raw_channel
                for position, observation in observations.items()
            }
            _diag["pilot_snr_db"] = {
                position: observation.snr_db
                for position, observation in observations.items()
            }
            _diag["pilot_blend_weights"] = {
                position: observation.blend_weight
                for position, observation in observations.items()
            }
            _diag["pilot_relative_jumps"] = {
                position: observation.relative_jump
                for position, observation in observations.items()
            }
            _diag["pilot_incremental_delays"] = {
                position: observation.incremental_delay_samples
                for position, observation in observations.items()
            }
            _diag["pilot_incremental_phases"] = {
                position: observation.incremental_phase_radians
                for position, observation in observations.items()
            }
            _diag["pilot_spectra"] = {
                position: spectra[position - 1].copy()
                for position in observations
            }
            _diag["known_pilot_frequency"] = self.known_pilot.transmitted_frequency.copy()
            _diag["tracked_channels"] = tracked_channels
            _diag["raw_spectra_bins"] = raw_spectra_bins
            _diag["equalised_rows"] = eq_rows_diag
            _diag["equaliser_metadata"] = equaliser_metadata
            _diag["row_noise_powers"] = row_noise_powers

        return np.stack(rows), np.stack(weights), block_count

    # ──────────────────────────────────────────────────────────────────────────
    # Core decode pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_llr_blocks(
        self,
        llr_blocks: NDArray[np.float64],
        selected: NDArray[np.int64] | None = None,
    ) -> tuple[list[NDArray[np.uint8]], list[int], list[bool]]:
        if selected is None:
            selected = np.arange(len(llr_blocks), dtype=np.int64)
        information: list[NDArray[np.uint8]] = []
        iterations: list[int] = []
        successes: list[bool] = []
        for index in selected:
            decoded, used, success = self.ldpc.decode_block(llr_blocks[int(index)])
            information.append(decoded)
            iterations.append(int(used))
            successes.append(bool(success))
        return information, iterations, successes

    def _group_hypothesis(
        self,
        rows: NDArray[np.complex128],
        rotation_quadrants: int,
        timing_samples: float,
    ) -> NDArray[np.complex128]:
        bins = self.config.data_bins.astype(float)
        phase = (
            2.0 * np.pi * bins * timing_samples / self.config.fft_length
            - rotation_quadrants * np.pi / 2.0
        )
        return rows * np.exp(1j * phase)[None, :]

    def _decode_group_with_rescue(
        self,
        rows: NDArray[np.complex128],
        weights: NDArray[np.float64],
    ) -> tuple[
        list[NDArray[np.uint8]], list[int], list[bool], dict[str, object]
    ]:
        baseline_llrs = self.interleaver.deinterleave_llrs(rows, weights)
        baseline_info, baseline_iters, baseline_success = self._decode_llr_blocks(
            baseline_llrs
        )
        best = (
            baseline_info,
            baseline_iters,
            baseline_success,
            0,
            0.0,
        )
        if all(baseline_success):
            return baseline_info, baseline_iters, baseline_success, {
                "rotation_quadrants": 0,
                "timing_samples": 0.0,
                "hypotheses_tested": 1,
                "baseline_successes": len(baseline_success),
            }

        probe = np.unique(np.linspace(
            0,
            self.config.ldpc_blocks_per_group - 1,
            6,
            dtype=np.int64,
        ))
        candidates: list[tuple[tuple[int, int, float], int, float]] = []
        # Baseline probe score is available without another decode.
        baseline_probe_success = sum(baseline_success[int(index)] for index in probe)
        baseline_probe_iters = sum(baseline_iters[int(index)] for index in probe)
        candidates.append((
            (baseline_probe_success, -baseline_probe_iters, 0.0), 0, 0.0
        ))

        def probe_candidate(rotation: int, timing: float) -> None:
            nonlocal tested
            transformed = self._group_hypothesis(rows, rotation, timing)
            llrs = self.interleaver.deinterleave_llrs(transformed, weights)
            _, probe_iters, probe_success = self._decode_llr_blocks(llrs, probe)
            tested += 1
            correction_penalty = -0.01 * (
                abs(timing) + 0.25 * min(rotation, 4 - rotation)
            )
            candidates.append((
                (
                    int(sum(probe_success)),
                    -int(sum(probe_iters)),
                    correction_penalty,
                ),
                rotation,
                timing,
            ))

        tested = 1
        # Most ambiguity failures are pure 90-degree QPSK rotations.  Test
        # those first; only open the timing grid when no rotation explains all
        # probe codewords.
        for rotation in (1, 2, 3):
            probe_candidate(rotation, 0.0)

        if max(score[0] for score, _, _ in candidates) < len(probe):
            for timing in self.config.ldpc_rescue_timing_offsets:
                if abs(timing) < EPS:
                    continue
                for rotation in range(4):
                    probe_candidate(rotation, float(timing))

        # Fully decode only the most promising distinct candidates.
        candidates.sort(key=lambda item: item[0], reverse=True)
        finalists: list[tuple[int, float]] = []
        for _, rotation, timing in candidates:
            key = (rotation, timing)
            if key not in finalists:
                finalists.append(key)
            if len(finalists) >= 3:
                break

        best_score = (
            int(sum(baseline_success)),
            -int(sum(baseline_iters)),
            0.0,
        )
        for rotation, timing in finalists:
            if rotation == 0 and abs(timing) < EPS:
                continue
            transformed = self._group_hypothesis(rows, rotation, timing)
            llrs = self.interleaver.deinterleave_llrs(transformed, weights)
            info, iters, success = self._decode_llr_blocks(llrs)
            score = (
                int(sum(success)),
                -int(sum(iters)),
                -0.01 * (abs(timing) + 0.25 * min(rotation, 4 - rotation)),
            )
            if score > best_score:
                best_score = score
                best = (info, iters, success, rotation, timing)

        info, iters, success, rotation, timing = best
        return info, iters, success, {
            "rotation_quadrants": int(rotation),
            "timing_samples": float(timing),
            "hypotheses_tested": tested,
            "baseline_successes": int(sum(baseline_success)),
            "selected_successes": int(sum(success)),
        }

    def decode_raw_signal(
        self,
        received: NDArray[np.float64],
        *,
        _diag: dict | None = None,
    ) -> RawReception:
        """Decode all complete LDPC groups without trusting the packet header."""
        received = np.asarray(received, dtype=float)
        corrected, raw_sync_start, offset_ppm = self.synchronise_and_correct(
            received, _diag=_diag
        )
        golay_channel = self._golay_channel(corrected, _diag=_diag)
        data_start, adjustment = self._refine_data_start(
            corrected, self.config.preamble_length, _diag=_diag
        )
        rows, weights, received_blocks = self._extract_data_rows(
            corrected, data_start, golay_channel, _diag=_diag
        )

        complete_groups = len(rows) // self.config.data_symbols_per_group
        if complete_groups == 0:
            raise ValueError("fewer than 30 data OFDM symbols were recovered")

        recovered_bits: list[NDArray[np.uint8]] = []
        successful_blocks = 0
        total_blocks = 0
        ldpc_iterations: list[int] = []
        ldpc_successes: list[bool] = []
        rescue_metadata: list[dict[str, object]] = []

        for group_index in range(complete_groups):
            lo = group_index * self.config.data_symbols_per_group
            hi = lo + self.config.data_symbols_per_group
            information_blocks, iterations, successes, rescue = (
                self._decode_group_with_rescue(rows[lo:hi], weights[lo:hi])
            )
            recovered_bits.append(np.concatenate(information_blocks))
            successful_blocks += int(sum(successes))
            total_blocks += len(successes)
            ldpc_iterations.extend(iterations)
            ldpc_successes.extend(successes)
            rescue_metadata.append(rescue)

        if _diag is not None:
            _diag["ldpc_iterations"] = ldpc_iterations
            _diag["ldpc_successes"] = ldpc_successes
            _diag["ldpc_rescue"] = rescue_metadata

        decoded_bytes = bits_to_bytes(np.concatenate(recovered_bits))
        info = ReceptionInfo(
            raw_sync_start,
            offset_ppm,
            received_blocks,
            len(rows),
            complete_groups,
            successful_blocks,
            total_blocks,
            adjustment,
        )
        return RawReception(decoded_bytes, info)

    # ──────────────────────────────────────────────────────────────────────────
    # Header recovery helpers (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _plausible_recovered_filename(filename: str) -> bool:
        """Conservative filename filter used only for automatic header repair."""
        if not filename or len(filename.encode("utf-8")) > 255:
            return False
        if filename != Path(filename).name or "/" in filename or "\\" in filename:
            return False
        return all(character.isprintable() and character not in "\r\n\x00"
                   for character in filename)

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
            return None
        return None

    @staticmethod
    def _text_byte_quality(byte_value: int) -> float:
        if byte_value in (9, 10, 13):
            return 1.0
        if 32 <= byte_value <= 126:
            return 1.0
        if 128 <= byte_value <= 244:
            return 0.45
        return -1.4

    @classmethod
    def _trim_text_payload(cls, data: bytes, start: int) -> tuple[int, float]:
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
            if index - start >= min_payload and bad_run >= 4:
                end = index - bad_run + 1
                break

        if end == len(data):
            window = 64
            for offset in range(min_payload, max(min_payload, len(data) - start - window + 1)):
                chunk = data[start + offset: start + offset + window]
                printable_ratio = (
                    sum(cls._text_byte_quality(byte) > 0.9 for byte in chunk) / window
                )
                if printable_ratio < 0.55:
                    end = start + offset
                    break

        while end > start and cls._text_byte_quality(data[end - 1]) < 0.9:
            end -= 1

        if end <= start:
            return start, -1e9

        payload = data[start:end]
        ascii_like = (
            sum(cls._text_byte_quality(byte) > 0.9 for byte in payload) / len(payload)
        )
        replacement_penalty = (
            payload.decode("utf-8", errors="replace").count("�") / max(1, len(payload))
        )
        newline_bonus = min(payload.count(b"\n"), 8) / 80.0
        length_bonus = min(len(payload), 4000) / 20000.0
        score = ascii_like + newline_bonus + length_bonus - 3.0 * replacement_penalty
        return end, float(score)

    @classmethod
    def salvage_text_payload(cls, decoded_bytes: bytes) -> TextSalvage | None:
        if len(decoded_bytes) < 8:
            return None

        candidates: list[tuple[int, str]] = []
        search_limit = min(len(decoded_bytes), 512)
        lower_prefix = decoded_bytes[:search_limit].lower()
        pos = lower_prefix.find(b".txt")
        while pos != -1:
            candidates.append((pos + 4, "after recovered .txt filename marker"))
            pos = lower_prefix.find(b".txt", pos + 1)

        for header_length in range(PacketCodec.MIN_HEADER_BYTES,
                                   min(96, len(decoded_bytes))):
            candidates.append((header_length, f"guessed header length {header_length}"))

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
            text = raw_text.lstrip("\ufeff\x00\r\n\t ")
            if not text:
                continue
            adjusted_score = score + (
                0.25 if method.startswith("after recovered .txt") else 0.0
            )
            candidate = TextSalvage(
                text=text,
                payload_start=start,
                payload_end=end,
                method=method,
                score=adjusted_score,
            )
            if best is None or candidate.score > best.score or (
                abs(candidate.score - best.score) < 1e-9
                and len(candidate.text) > len(best.text)
            ):
                best = candidate

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
        if header_length < 0 or payload_bytes < 0:
            raise ValueError("override lengths must be non-negative")
        total = header_length + payload_bytes
        if total > len(decoded_bytes):
            raise ValueError(
                f"override requests {total} bytes from a "
                f"{len(decoded_bytes)}-byte decoded stream"
            )
        return DecodedPacket(
            filename=filename,
            file_size=payload_bytes,
            payload=decoded_bytes[header_length:total],
            header_length=header_length,
        )

    def decode_signal(
        self, received: NDArray[np.float64]
    ) -> tuple[DecodedPacket, ReceptionInfo]:
        """Strict API retained for callers that require a valid JOSS-F header."""
        raw = self.decode_raw_signal(received)
        packet = PacketCodec.parse(raw.decoded_bytes)
        return packet, raw.info

    # ──────────────────────────────────────────────────────────────────────────
    # Top-level WAV entry point
    # ──────────────────────────────────────────────────────────────────────────

    def decode_wav(
        self,
        wav_path: str | Path,
        output_directory: str | Path = ".",
        *,
        strict_header: bool = False,
        forced_header_length: int | None = None,
        forced_payload_bytes: int | None = None,
        forced_filename: str = "recovered_payload.bin",
        diagnostics: bool = True,
    ) -> Path:
        received = read_wav(wav_path, self.config)
        peak    = float(np.max(np.abs(received))) if len(received) else 0.0
        clipped = float(np.mean(np.abs(received) >= 0.995)) if len(received) else 0.0

        # Collect diagnostic data unless explicitly disabled.
        _diag: dict | None = {} if diagnostics else None

        raw = self.decode_raw_signal(received, _diag=_diag)

        if (forced_header_length is None) != (forced_payload_bytes is None):
            raise ValueError(
                "forced_header_length and forced_payload_bytes must be supplied together"
            )

        recovery: HeaderRecovery | None
        if forced_header_length is not None and forced_payload_bytes is not None:
            packet = self.packet_from_override(
                raw.decoded_bytes,
                header_length=forced_header_length,
                payload_bytes=forced_payload_bytes,
                filename=forced_filename,
            )
            recovery = HeaderRecovery(packet, "manual boundary override")
        else:
            recovery = self.recover_packet_header(raw.decoded_bytes)

        print(f"Decoded physical layer from {wav_path}")
        print(f"  input peak       : {peak:.3f} ({100.0 * clipped:.3f}% clipped)")
        print(f"  sync sample      : {raw.info.sync_sample}")
        print(f"  sample offset    : {raw.info.sample_rate_offset_ppm:+.1f} ppm")
        print(f"  OFDM adjustment  : {raw.info.data_start_adjustment:+d} samples")
        print(f"  LDPC success     : {raw.info.successful_ldpc_blocks}/{raw.info.total_ldpc_blocks}")
        print(f"  decoded stream   : {len(raw.decoded_bytes)} bytes")
        print(f"  LDPC fingerprint : {self.ldpc.compatibility_fingerprint()[:16]}")
        print(f"  pilot fingerprint: {self.known_pilot.content_sha256[:16]}")

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        # ── Save the decoded output ──
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
        else:
            if strict_header:
                # Still run diagnostics before raising.
                pass

            wav_stem = Path(wav_path).stem
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
                "sample_rate_offset_ppm": raw.info.sample_rate_offset_ppm,
                "data_start_adjustment": raw.info.data_start_adjustment,
                "successful_ldpc_blocks": raw.info.successful_ldpc_blocks,
                "total_ldpc_blocks": raw.info.total_ldpc_blocks,
                "ldpc_fingerprint": self.ldpc.compatibility_fingerprint(),
                "pilot_fingerprint": self.known_pilot.content_sha256,
                "text_salvage": None if text_salvage is None else {
                    "path": str(text_path),
                    "payload_start_guess": text_salvage.payload_start,
                    "payload_end_guess": text_salvage.payload_end,
                    "method": text_salvage.method,
                    "score": text_salvage.score,
                },
                "note": (
                    "The JOSS-F header supplies both payload offset and payload length. "
                    "Without a recoverable header, the exact payload boundary is not "
                    "uniquely determined; the raw file preserves every LDPC-decoded byte, "
                    "including header and post-payload padding.  If text_salvage is present, "
                    "the .txt file is only a best-effort human-readable reconstruction."
                ),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
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

            if strict_header:
                # Raise after diagnostics are generated below.
                _run_diagnostics(_diag, output_directory, wav_path, self.config)
                raise ValueError(
                    "JOSS-F header could not be decoded and strict-header mode is enabled. "
                    f"The physical layer recovered {len(raw.decoded_bytes)} bytes."
                )

        # ── Diagnostics ──
        _run_diagnostics(_diag, output_directory, wav_path, self.config)

        return text_path if (recovery is None and text_path is not None) else output_path


# ── Diagnostics helper (isolated so import failures are non-fatal) ────────────

def _run_diagnostics(
    _diag: dict | None,
    output_directory: Path,
    wav_path: str | Path,
    config: ModemConfig,
) -> None:
    if _diag is None:
        return
    try:
        from diagnostics import plot_all  # type: ignore[import]
    except ImportError:
        print("  [note] install matplotlib for diagnostic plots (pip install matplotlib)")
        return
    try:
        diag_dir = Path(output_directory) / "diagnostic_plots"
        wav_stem = Path(wav_path).stem
        print(f"\n  Generating diagnostic plots → {diag_dir}/")
        plot_all(_diag, diag_dir, wav_stem=wav_stem, config=config)
    except Exception as exc:  # noqa: BLE001
        print(f"  [note] diagnostics failed: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode a JOSS-F WAV recording")
    parser.add_argument("wav", nargs="?", help="input WAV path; prompts when omitted")
    parser.add_argument(
        "--strict-header",
        action="store_true",
        help="raise instead of preserving the raw LDPC stream when the header fails",
    )
    parser.add_argument(
        "--header-bytes",
        type=int,
        help="known payload offset in decoded bytes for manual recovery",
    )
    parser.add_argument(
        "--payload-bytes",
        type=int,
        help="known payload length in bytes for manual recovery",
    )
    parser.add_argument(
        "--filename",
        default="recovered_payload.bin",
        help="output filename used with manual boundary overrides",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="skip generating diagnostic plots",
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
        OUTPUT_DIRECTORY,
        strict_header=args.strict_header,
        forced_header_length=args.header_bytes,
        forced_payload_bytes=args.payload_bytes,
        forced_filename=args.filename,
        diagnostics=not args.no_diagnostics,
    )
