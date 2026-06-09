from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import struct

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve, find_peaks

from diagnostics import (
    DiagnosticSession,
    complex_gain_and_nmse,
    normalised_correlation,
    qpsk_evm,
    rms,
    sha256_array,
)
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
    """Original JOSS-F receiver with non-invasive diagnostics.

    The synchronisation, channel estimate, timing selection, equalisation,
    deinterleaving, LDPC decoding and header parsing are unchanged. Diagnostic
    calculations are observational only and are never fed back into decoding.
    """

    def __init__(
        self,
        config: ModemConfig = CONFIG,
        diagnostics: bool = True,
        diagnostics_root: str | Path | None = None,
        make_plots: bool = True,
        deep_diagnostics: bool = True,
    ):
        self.config = config
        self.ldpc = WiMaxLDPC(config)
        self.interleaver = StandardInterleaver(config)
        self.ofdm = OFDM(config)
        self.scheduler = PilotScheduler(config.pilot_interval)
        self.chirp = LinearChirp(config)
        self.golay = GolayPilot(config)
        self.known_pilot = KnownOFDMPilot(config)
        self.diagnostics_enabled = diagnostics
        self.diagnostics_root = None if diagnostics_root is None else Path(diagnostics_root)
        self.make_plots = make_plots
        self.deep_diagnostics = deep_diagnostics
        self.diag: DiagnosticSession | None = None
        self._diagnostic_spectra: list[NDArray[np.complex128]] = []
        self._diagnostic_equalised_rows: list[NDArray[np.complex128]] = []

    # ------------------------------------------------------------------
    # Small diagnostics-only helpers
    # ------------------------------------------------------------------
    def _start_diagnostics(self, wav_path: str | Path) -> DiagnosticSession | None:
        if not self.diagnostics_enabled:
            return None
        source = Path(wav_path)
        if self.diagnostics_root is not None:
            directory = self.diagnostics_root
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = source.parent / "diagnostics" / f"{source.stem}_{stamp}"
        session = DiagnosticSession(directory, source)
        c = self.config
        session.stage(
            "configuration",
            sample_rate=c.sample_rate,
            chirp_length=c.chirp_length,
            chirp_count=c.chirp_count,
            chirp_start_hz=c.chirp_start_hz,
            chirp_stop_hz=c.chirp_stop_hz,
            chirp_hash=sha256_array(self.chirp.samples()),
            golay_a_hash=sha256_array(self.golay.a),
            golay_b_hash=sha256_array(self.golay.b),
            known_pilot_symbol_hash=sha256_array(self.known_pilot.symbols),
            known_pilot_waveform_hash=sha256_array(self.known_pilot.waveform()),
            fft_length=c.fft_length,
            cyclic_prefix=c.cyclic_prefix,
            first_data_bin=c.first_data_bin,
            last_data_bin=c.last_data_bin,
            carriers_per_symbol=c.carriers_per_symbol,
            pilot_interval=c.pilot_interval,
            preamble_length=c.preamble_length,
            ofdm_symbol_length=c.ofdm_symbol_length,
            interleaver_stride=self.interleaver.STRIDE,
            interleaver_hash=sha256_array(self.interleaver.permutation),
        )
        session.log(f"Diagnostic directory: {directory}")
        return session

    @staticmethod
    def _ascii_preview(data: bytes, limit: int = 64) -> str:
        return "".join(chr(value) if 32 <= value < 127 else "." for value in data[:limit])

    def _record_header_preview(self, decoded_bytes: bytes, group_index: int) -> None:
        if self.diag is None:
            return
        preview = decoded_bytes[:96]
        entry: dict[str, object] = {
            "after_group": group_index + 1,
            "available_bytes": len(decoded_bytes),
            "first_96_hex": preview.hex(" "),
            "first_96_ascii": self._ascii_preview(preview, 96),
        }
        if len(decoded_bytes) >= 6:
            header_be = struct.unpack(">H", decoded_bytes[:2])[0]
            file_be = struct.unpack(">I", decoded_bytes[2:6])[0]
            header_le = struct.unpack("<H", decoded_bytes[:2])[0]
            file_le = struct.unpack("<I", decoded_bytes[2:6])[0]
            entry.update(
                header_length_big_endian=header_be,
                file_size_big_endian=file_be,
                total_big_endian=header_be + file_be,
                header_length_little_endian=header_le,
                file_size_little_endian=file_le,
                total_little_endian=header_le + file_le,
            )
        self.diag.event("decoded_byte_preview", **entry)
        self.diag.write_text(
            f"decoded_bytes_after_group_{group_index + 1}.txt",
            f"bytes available: {len(decoded_bytes)}\n"
            f"hex:   {preview.hex(' ')}\n"
            f"ascii: {self._ascii_preview(preview, 96)}\n",
        )

    def _diagnose_qpsk_hypotheses(
        self,
        rows: NDArray[np.complex128],
        weights: NDArray[np.float64],
    ) -> None:
        """Decode the first LDPC block under common QPSK convention changes.

        Results are diagnostic only. The actual decoder continues to use the
        unmodified identity convention.
        """
        if self.diag is None or not self.deep_diagnostics:
            return
        if len(rows) < self.config.data_symbols_per_group:
            return

        base_rows = rows[: self.config.data_symbols_per_group]
        base_weights = weights[: self.config.data_symbols_per_group]
        transforms: list[tuple[str, object]] = []
        for conjugate in (False, True):
            for quarter_turns in range(4):
                name = ("conjugate_" if conjugate else "") + f"rotate_{90 * quarter_turns:+d}_deg"

                def transform(
                    values: NDArray[np.complex128],
                    conjugate: bool = conjugate,
                    quarter_turns: int = quarter_turns,
                ) -> NDArray[np.complex128]:
                    output = np.conj(values) if conjugate else values
                    return output * (1j ** quarter_turns)

                transforms.append((name, transform))

        results: list[dict[str, object]] = []
        flat_weights = base_weights.reshape(-1)
        inverse = np.empty_like(self.interleaver.permutation)
        inverse[self.interleaver.permutation] = np.arange(len(inverse), dtype=np.int64)

        for name, transform in transforms:
            transformed = transform(base_rows)
            # Standard deinterleaver hypothesis.
            llr_blocks = self.interleaver.deinterleave_llrs(transformed, base_weights)
            info, iterations, success = self.ldpc.decode_block(llr_blocks[0])
            decoded = bits_to_bytes(info)
            results.append(
                {
                    "hypothesis": name,
                    "deinterleaver": "standard",
                    "ldpc_success": bool(success),
                    "iterations": iterations,
                    "first_bytes_hex": decoded[:16].hex(" "),
                    "first_bytes_ascii": self._ascii_preview(decoded, 16),
                }
            )

            # Frequent interoperability bug: applying the inverse permutation.
            gathered = transformed.reshape(-1)[inverse]
            gathered_weights = flat_weights[inverse]
            alternate_llrs = QPSK.llr_components(gathered, gathered_weights).reshape(
                self.config.ldpc_blocks_per_group,
                self.config.ldpc_code_bits,
            )
            info_alt, iterations_alt, success_alt = self.ldpc.decode_block(alternate_llrs[0])
            decoded_alt = bits_to_bytes(info_alt)
            results.append(
                {
                    "hypothesis": name,
                    "deinterleaver": "inverse_permutation",
                    "ldpc_success": bool(success_alt),
                    "iterations": iterations_alt,
                    "first_bytes_hex": decoded_alt[:16].hex(" "),
                    "first_bytes_ascii": self._ascii_preview(decoded_alt, 16),
                }
            )

        self.diag.rows("qpsk_interleaver_hypotheses", results)
        successful = [row for row in results if row["ldpc_success"]]
        identity_success = any(
            row["hypothesis"] == "rotate_+0_deg"
            and row["deinterleaver"] == "standard"
            and row["ldpc_success"]
            for row in results
        )
        alternate_successful = [
            row
            for row in successful
            if not (
                row["hypothesis"] == "rotate_+0_deg"
                and row["deinterleaver"] == "standard"
            )
        ]
        self.diag.stage(
            "hypothesis_scan",
            tested=len(results),
            identity_success=identity_success,
            alternate_successful_count=len(alternate_successful),
            alternate_successful_hypotheses=alternate_successful,
        )
        if not identity_success and alternate_successful:
            self.diag.warning(
                "The standard QPSK/interleaver convention fails, but an alternate hypothesis "
                "passes the first LDPC block. This strongly indicates a convention mismatch."
            )

    # ------------------------------------------------------------------
    # Original receiver implementation, with observations added
    # ------------------------------------------------------------------
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

        if self.diag is not None:
            step = max(1, len(metric) // 40_000)
            self.diag.array("chirp_metric_decimated", metric[::step])
            self.diag.array("chirp_metric_decimated_samples", np.arange(0, len(metric), step))
            self.diag.array("chirp_candidate_peaks", peaks)
            top_count = min(30, len(peaks))
            top = peaks[np.argsort(metric[peaks])[-top_count:]] if top_count else np.array([], dtype=int)
            self.diag.rows(
                "chirp_top_peaks",
                [
                    {
                        "sample": int(index),
                        "metric": float(metric[index]),
                        "relative_to_max": float(metric[index] / (np.max(metric) + EPS)),
                    }
                    for index in top[np.argsort(top)]
                ],
            )

        if best_chain is None:
            template = self.chirp.train()
            coarse_metric = np.abs(fftconvolve(received, template[::-1], mode="valid"))
            sync = int(np.argmax(coarse_metric))
            if self.diag is not None:
                self.diag.stage(
                    "chirp_sync",
                    method="full_train_fallback",
                    detected_peak_count=len(peaks),
                    best_chain_found=False,
                    sync_sample=sync,
                    single_chirp_metric_max=float(np.max(metric)),
                    train_metric_max=float(np.max(coarse_metric)),
                )
                self.diag.warning(
                    "No ten-chirp chain was found; the receiver used its coarse full-train fallback."
                )
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

        if self.diag is not None:
            spacings = np.diff(observed_values)
            chain_metrics = [float(metric[index]) for index in best_chain]
            self.diag.array("chirp_chain_samples", np.asarray(observed_values))
            self.diag.stage(
                "chirp_sync",
                method="ten_chirp_chain",
                detected_peak_count=len(peaks),
                best_chain_found=True,
                integer_peak_samples=best_chain,
                refined_peak_samples=observed_values,
                spacings=spacings,
                spacing_mean=float(np.mean(spacings)),
                spacing_std=float(np.std(spacings)),
                spacing_min=float(np.min(spacings)),
                spacing_max=float(np.max(spacings)),
                chain_metric_relative=[value / (float(np.max(metric)) + EPS) for value in chain_metrics],
                fitted_period=float(period),
                fitted_intercept=float(intercept),
                sampling_ratio=ratio,
                sample_rate_offset_ppm=offset_ppm,
                sync_sample=sync,
                corrected_samples=len(corrected),
            )
            self.diag.log(
                f"Chirps: chain={best_chain}, spacing mean/std={np.mean(spacings):.3f}/"
                f"{np.std(spacings):.3f}, SFO={offset_ppm:+.2f} ppm"
            )
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
        pulse_rows: list[dict[str, object]] = []
        individual_channels: list[NDArray[np.complex128]] = []

        for repeat_index, (a_relative, b_relative) in enumerate(self.golay.pulse_starts, start=1):
            for label, relative, reference, reference_f in (
                ("A", a_relative, x_a, x_a_f),
                ("B", b_relative, x_b, x_b_f),
            ):
                start = golay_start + relative
                window = received[start : start + transform_length]
                if len(window) < transform_length:
                    window = np.pad(window, (0, transform_length - len(window)))
                received_f = np.fft.rfft(window, n=transform_length)
                numerator += received_f * np.conj(reference_f)
                denominator += np.abs(reference_f) ** 2

                if self.diag is not None:
                    local_denominator = np.abs(reference_f) ** 2
                    local_reg = c.golay_regularisation * float(np.max(local_denominator))
                    local_h = np.fft.irfft(
                        received_f * np.conj(reference_f) / (local_denominator + local_reg),
                        n=transform_length,
                    )[: c.cyclic_prefix]
                    individual_channels.append(np.fft.rfft(local_h, n=c.fft_length))
                    pulse = window[: c.fft_length]
                    pulse_rows.append(
                        {
                            "repeat": repeat_index,
                            "sequence": label,
                            "start_sample_after_sync": start,
                            "window_rms": rms(window),
                            "pulse_peak": float(np.max(np.abs(pulse))),
                            "zero_gap_rms": rms(window[c.fft_length :]),
                            "direct_normalised_correlation": normalised_correlation(
                                pulse, reference[: c.fft_length]
                            ),
                        }
                    )

        regularisation = c.golay_regularisation * float(np.max(denominator))
        h = np.fft.irfft(
            numerator / (denominator + regularisation),
            n=transform_length,
        )
        h = h[: c.cyclic_prefix]
        channel = np.fft.rfft(h, n=c.fft_length)

        if self.diag is not None:
            bins = c.data_bins
            band = np.abs(channel[bins])
            energy = np.abs(h) ** 2
            total_energy = float(np.sum(energy)) + EPS
            cumulative = np.cumsum(energy) / total_energy
            energy_90 = int(np.searchsorted(cumulative, 0.90))
            energy_99 = int(np.searchsorted(cumulative, 0.99))
            consistency: list[float] = []
            if individual_channels:
                stack = np.stack(individual_channels)
                reference_channel = np.mean(stack, axis=0)
                consistency = [
                    normalised_correlation(item[bins], reference_channel[bins]) for item in stack
                ]
                self.diag.array("golay_individual_channels", stack)
            self.diag.rows("golay_pulses", pulse_rows)
            self.diag.array("golay_impulse_response", h)
            self.diag.array("golay_channel", channel)
            self.diag.stage(
                "golay_channel",
                pulse_count=len(pulse_rows),
                pulse_metrics=pulse_rows,
                individual_channel_coherences=consistency,
                individual_channel_coherence_min=min(consistency) if consistency else None,
                individual_channel_coherence_median=float(np.median(consistency)) if consistency else None,
                band_magnitude_min=float(np.min(band)),
                band_magnitude_median=float(np.median(band)),
                band_magnitude_max=float(np.max(band)),
                band_dynamic_range_db=float(
                    20.0 * np.log10((np.max(band) + EPS) / (np.min(band) + EPS))
                ),
                impulse_peak_index=int(np.argmax(np.abs(h))),
                impulse_energy_90_percent_index=energy_90,
                impulse_energy_99_percent_index=energy_99,
                regularisation=regularisation,
            )
            self.diag.log(
                f"Golay: |H| median={np.median(band):.4g}, dynamic range="
                f"{20.0 * np.log10((np.max(band) + EPS)/(np.min(band) + EPS)):.1f} dB, "
                f"99% impulse energy by sample {energy_99}"
            )
        return channel

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

        if self.diag is not None:
            fine_scan_offsets = np.arange(-radius, radius + 1, dtype=int)
            fine_scan_scores = np.asarray(
                [self._cp_score(received, nominal + int(offset)) for offset in fine_scan_offsets]
            )
            # A broader diagnostic scan can reveal a standardised preamble-layout mismatch.
            broad_radius = 2 * self.config.ofdm_symbol_length
            broad_offsets = np.arange(-broad_radius, broad_radius + 1, 32, dtype=int)
            broad_scores = np.asarray(
                [self._cp_score(received, nominal + int(offset)) for offset in broad_offsets]
            )
            broad_max = float(np.nanmax(broad_scores))
            near_max = np.flatnonzero(broad_scores >= broad_max - 1e-6)
            broad_best_index = int(near_max[np.argmin(np.abs(broad_offsets[near_max]))])
            broad_best = int(broad_offsets[broad_best_index])
            symbol_length = self.config.ofdm_symbol_length
            broad_residual = int(
                ((broad_best - int(best) + symbol_length // 2) % symbol_length)
                - symbol_length // 2
            )
            self.diag.array("cp_fine_offsets", fine_scan_offsets)
            self.diag.array("cp_fine_scores", fine_scan_scores)
            self.diag.array("cp_broad_offsets", broad_offsets)
            self.diag.array("cp_broad_scores", broad_scores)
            self.diag.stage(
                "ofdm_timing",
                nominal_start=nominal,
                selected_offset=int(best),
                selected_start=nominal + int(best),
                selected_score=float(self._cp_score(received, nominal + int(best))),
                nominal_score=float(self._cp_score(received, nominal)),
                broad_best_offset=broad_best,
                broad_best_score=float(broad_scores[broad_best_index]),
                broad_best_distance_from_selected=abs(broad_best - int(best)),
                broad_best_residual_from_selected=abs(broad_residual),
            )
            self.diag.log(
                f"OFDM boundary: nominal={nominal}, selected={nominal + best} ({best:+d}), "
                f"CP score={self._cp_score(received, nominal + best):.3f}; "
                f"broad best offset={broad_best:+d}"
            )
            if abs(broad_residual) > radius:
                self.diag.warning(
                    "The strongest broad CP-correlation boundary lies outside the receiver's "
                    "normal refinement radius. This suggests a preamble-length/layout mismatch."
                )
        return nominal + best, int(best)

    def _equalise(
        self,
        spectrum: NDArray[np.complex128],
        channel: NDArray[np.complex128],
        diagnostic_block_index: int | None = None,
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

        if self.diag is not None and diagnostic_block_index is not None:
            evm_rms, evm_median = qpsk_evm(estimate)
            quadrants = {
                "++": int(np.sum((estimate.real >= 0) & (estimate.imag >= 0))),
                "-+": int(np.sum((estimate.real < 0) & (estimate.imag >= 0))),
                "+-": int(np.sum((estimate.real >= 0) & (estimate.imag < 0))),
                "--": int(np.sum((estimate.real < 0) & (estimate.imag < 0))),
            }
            self.diag.event(
                "equalised_data_block",
                absolute_ofdm_block=diagnostic_block_index,
                phase_slope_rad_per_bin=float(slope),
                common_phase_rad=float(intercept),
                pre_normalisation_median_magnitude=robust_magnitude,
                evm_rms=evm_rms,
                evm_median=evm_median,
                quadrants=quadrants,
            )
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

        blocks = [
            received[
                data_start + index * symbol_length : data_start + (index + 1) * symbol_length
            ]
            for index in range(block_count)
        ]
        spectra = [self.ofdm.spectrum(block) for block in blocks]
        self._diagnostic_spectra = spectra

        if self.diag is not None:
            known = self.known_pilot.transmitted_frequency
            bins = c.data_bins
            pilot_scan_rows: list[dict[str, object]] = []
            cp_rows: list[dict[str, object]] = []
            for one_based, (block, spectrum) in enumerate(zip(blocks, spectra), start=1):
                prefix = block[: c.cyclic_prefix]
                tail = block[c.fft_length : c.fft_length + c.cyclic_prefix]
                cp_corr = float(
                    abs(np.vdot(prefix, tail))
                    / (np.linalg.norm(prefix) * np.linalg.norm(tail) + EPS)
                )
                gain, nmse, coherence = complex_gain_and_nmse(spectrum[bins], known[bins])
                pilot_scan_rows.append(
                    {
                        "ofdm_block": one_based,
                        "expected_pilot": self.scheduler.is_pilot(one_based),
                        "known_pilot_coherence": coherence,
                        "known_pilot_nmse_after_scalar_gain": nmse,
                        "best_scalar_gain_magnitude": abs(gain),
                        "best_scalar_gain_phase_rad": float(np.angle(gain)),
                        "band_rms": rms(spectrum[bins]),
                    }
                )
                cp_rows.append(
                    {
                        "ofdm_block": one_based,
                        "cp_correlation": cp_corr,
                        "block_rms": rms(block),
                        "block_peak": float(np.max(np.abs(block))),
                    }
                )

            expected_rows = [row for row in pilot_scan_rows if row["expected_pilot"]]
            best_pilot_like = max(
                pilot_scan_rows,
                key=lambda row: float(row["known_pilot_coherence"]),
            )
            self.diag.rows("ofdm_blocks", cp_rows)
            self.diag.rows("known_pilot_scan", pilot_scan_rows)
            self.diag.array(
                "known_pilot_coherence_by_block",
                np.asarray([row["known_pilot_coherence"] for row in pilot_scan_rows], dtype=float),
            )
            self.diag.array(
                "known_pilot_nmse_by_block",
                np.asarray(
                    [row["known_pilot_nmse_after_scalar_gain"] for row in pilot_scan_rows],
                    dtype=float,
                ),
            )

            # Test whether the agreed pilot appears shifted by one or more FFT bins.
            shift_rows: list[dict[str, object]] = []
            expected_positions = [
                index for index in range(1, block_count + 1) if self.scheduler.is_pilot(index)
            ]
            for shift in range(-4, 5):
                shifted_bins = bins + shift
                if np.any(shifted_bins < 0) or np.any(shifted_bins >= len(known)):
                    continue
                scores: list[float] = []
                nmses: list[float] = []
                for position in expected_positions:
                    _, nmse, coherence = complex_gain_and_nmse(
                        spectra[position - 1][shifted_bins], known[bins]
                    )
                    scores.append(coherence)
                    nmses.append(nmse)
                shift_rows.append(
                    {
                        "carrier_bin_shift": shift,
                        "mean_expected_pilot_coherence": float(np.mean(scores)) if scores else None,
                        "mean_expected_pilot_nmse": float(np.mean(nmses)) if nmses else None,
                    }
                )
            self.diag.rows("pilot_carrier_shift_scan", shift_rows)
            best_shift = (
                max(
                    (row for row in shift_rows if row["mean_expected_pilot_coherence"] is not None),
                    key=lambda row: float(row["mean_expected_pilot_coherence"]),
                )
                if expected_positions
                else None
            )
            self.diag.stage(
                "ofdm_blocks",
                complete_block_count=block_count,
                available_samples=available,
                leftover_samples=available - block_count * symbol_length,
                median_cp_correlation=float(np.median([row["cp_correlation"] for row in cp_rows])),
            )
            self.diag.stage(
                "known_pilots",
                expected_positions=expected_positions,
                expected_pilot_metrics=expected_rows,
                most_pilot_like_block=best_pilot_like,
                best_carrier_shift=best_shift,
            )
            self.diag.log(
                f"OFDM: {block_count} complete blocks; most pilot-like block="
                f"{best_pilot_like['ofdm_block']} with coherence "
                f"{float(best_pilot_like['known_pilot_coherence']):.3f}"
            )
            if expected_positions and int(best_pilot_like["ofdm_block"]) not in expected_positions:
                self.diag.warning(
                    "The block most similar to the known pilot is not at an expected pilot "
                    "position. Check whether the other transmitter counts 20 data blocks "
                    "before inserting a pilot rather than making absolute block 20 the pilot."
                )
            if best_shift is not None and int(best_shift["carrier_bin_shift"]) != 0:
                self.diag.warning(
                    f"Known-pilot agreement is best with a carrier-bin shift of "
                    f"{best_shift['carrier_bin_shift']:+d}. Check the exact occupied-bin range."
                )

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
        data_quality_rows: list[dict[str, object]] = []

        for one_based, spectrum in enumerate(spectra, start=1):
            if self.scheduler.is_pilot(one_based):
                continue
            if len(pilot_positions):
                nearest = int(pilot_positions[np.argmin(np.abs(pilot_positions - one_based))])
                channel = pilot_channels[nearest]
            else:
                nearest = None
                channel = golay_channel

            estimate = self._equalise(spectrum, channel, diagnostic_block_index=one_based)
            bins = c.data_bins
            reliability = np.abs(channel[bins]) ** 2
            reliability /= float(np.median(reliability) + EPS)
            reliability = np.clip(reliability, 0.10, 8.0)
            hard = QPSK.hard_symbols(estimate)
            residual = float(np.median(np.abs(estimate - hard) ** 2))
            common = 2.0 / max(residual, 0.05)
            rows.append(estimate)
            weights.append(common * reliability)

            if self.diag is not None:
                evm_rms, evm_median = qpsk_evm(estimate)
                data_quality_rows.append(
                    {
                        "absolute_ofdm_block": one_based,
                        "nearest_pilot_block": nearest,
                        "evm_rms": evm_rms,
                        "evm_median": evm_median,
                        "median_squared_residual": residual,
                        "llr_common_scale": common,
                        "median_channel_reliability": float(np.median(reliability)),
                        "minimum_channel_reliability": float(np.min(reliability)),
                    }
                )

        if not rows:
            raise ValueError("no JOSS-F data OFDM blocks were found")
        stacked_rows = np.stack(rows)
        stacked_weights = np.stack(weights)
        self._diagnostic_equalised_rows = rows

        if self.diag is not None:
            evms = [float(row["evm_rms"]) for row in data_quality_rows]
            self.diag.rows("data_constellation_quality", data_quality_rows)
            self.diag.array("equalised_constellation_first_blocks", stacked_rows[: min(5, len(stacked_rows))])
            self.diag.stage(
                "equalisation",
                data_rows=len(stacked_rows),
                evm_rms_median=float(np.median(evms)),
                evm_rms_min=float(np.min(evms)),
                evm_rms_max=float(np.max(evms)),
                first_blocks=data_quality_rows[: min(10, len(data_quality_rows))],
            )
            self.diag.log(
                f"Equalisation: {len(stacked_rows)} data rows, median RMS EVM="
                f"{np.median(evms):.3f}"
            )
        return stacked_rows, stacked_weights, block_count

    def decode_signal(self, received: NDArray[np.float64]) -> tuple[DecodedPacket, ReceptionInfo]:
        received = np.asarray(received, dtype=float)
        corrected, raw_sync_start, offset_ppm = self.synchronise_and_correct(received)
        if self.diag is not None:
            self.diag.stage(
                "corrected_signal",
                samples=len(corrected),
                duration_seconds=len(corrected) / self.config.sample_rate,
                peak=float(np.max(np.abs(corrected))) if len(corrected) else 0.0,
                rms=rms(corrected),
            )
        golay_channel = self._golay_channel(corrected)
        data_start, adjustment = self._refine_data_start(corrected, self.config.preamble_length)
        rows, weights, received_blocks = self._extract_data_rows(
            corrected, data_start, golay_channel
        )

        complete_groups = len(rows) // self.config.data_symbols_per_group
        if self.diag is not None:
            self.diag.stage(
                "grouping",
                recovered_data_rows=len(rows),
                complete_groups=complete_groups,
                discarded_incomplete_rows=len(rows) % self.config.data_symbols_per_group,
            )
        if complete_groups == 0:
            raise ValueError("fewer than 30 data OFDM symbols were recovered")

        self._diagnose_qpsk_hypotheses(rows, weights)

        recovered_bits: list[NDArray[np.uint8]] = []
        successful_blocks = 0
        total_blocks = 0
        decoded_groups = 0
        requested_bytes: int | None = None
        header_error: str | None = None
        maximum_recoverable = (
            complete_groups * self.config.ldpc_blocks_per_group * self.config.ldpc_info_bits // 8
        )
        ldpc_rows: list[dict[str, object]] = []

        for group_index in range(complete_groups):
            lo = group_index * self.config.data_symbols_per_group
            hi = lo + self.config.data_symbols_per_group
            llr_blocks = self.interleaver.deinterleave_llrs(rows[lo:hi], weights[lo:hi])
            information_blocks: list[NDArray[np.uint8]] = []
            for block_index, llr in enumerate(llr_blocks):
                information, iterations, success = self.ldpc.decode_block(llr)
                information_blocks.append(information)
                successful_blocks += int(success)
                total_blocks += 1
                ldpc_rows.append(
                    {
                        "group": group_index + 1,
                        "ldpc_block": block_index + 1,
                        "success": bool(success),
                        "iterations": iterations,
                        "llr_min": float(np.min(llr)),
                        "llr_median_abs": float(np.median(np.abs(llr))),
                        "llr_mean_abs": float(np.mean(np.abs(llr))),
                        "llr_max": float(np.max(llr)),
                        "hard_one_fraction": float(np.mean(llr < 0.0)),
                    }
                )
            recovered_bits.append(np.concatenate(information_blocks))
            decoded_groups += 1

            decoded_bytes = bits_to_bytes(np.concatenate(recovered_bits))
            self._record_header_preview(decoded_bytes, group_index)
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

            if self.diag is not None:
                group_rows = [row for row in ldpc_rows if row["group"] == group_index + 1]
                self.diag.log(
                    f"LDPC group {group_index + 1}: "
                    f"{sum(bool(row['success']) for row in group_rows)}/"
                    f"{len(group_rows)} blocks passed; first bytes="
                    f"{decoded_bytes[:16].hex(' ')}"
                )

            if requested_bytes is not None and len(decoded_bytes) >= requested_bytes:
                packet = PacketCodec.parse(decoded_bytes[:requested_bytes])
                if self.diag is not None:
                    self.diag.rows("ldpc_blocks", ldpc_rows)
                    self.diag.stage(
                        "ldpc",
                        successful_blocks=successful_blocks,
                        total_blocks=total_blocks,
                        success_fraction=successful_blocks / max(total_blocks, 1),
                        decoded_groups=decoded_groups,
                    )
                    self.diag.stage(
                        "header",
                        valid=True,
                        header_length=packet.header_length,
                        file_size=packet.file_size,
                        filename=packet.filename,
                        requested_total_bytes=requested_bytes,
                    )
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
        if self.diag is not None:
            self.diag.rows("ldpc_blocks", ldpc_rows)
            self.diag.stage(
                "ldpc",
                successful_blocks=successful_blocks,
                total_blocks=total_blocks,
                success_fraction=successful_blocks / max(total_blocks, 1),
                decoded_groups=decoded_groups,
                median_iterations=float(np.median([row["iterations"] for row in ldpc_rows])),
            )
        try:
            packet = PacketCodec.parse(decoded_bytes)
        except (ValueError, UnicodeError) as exc:
            detail = header_error or str(exc)
            if self.diag is not None:
                self.diag.stage(
                    "header",
                    valid=False,
                    error=detail,
                    decoded_bytes_available=len(decoded_bytes),
                    first_96_hex=decoded_bytes[:96].hex(" "),
                    first_96_ascii=self._ascii_preview(decoded_bytes, 96),
                )
            raise ValueError(
                "JOSS-F header could not be decoded. "
                f"LDPC parity checks passed for {successful_blocks}/{total_blocks} blocks; "
                f"data-start adjustment was {adjustment:+d} samples. {detail}"
            ) from exc

        if self.diag is not None:
            self.diag.stage(
                "header",
                valid=True,
                header_length=packet.header_length,
                file_size=packet.file_size,
                filename=packet.filename,
            )
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

    # ------------------------------------------------------------------
    # Report/plot generation
    # ------------------------------------------------------------------
    def _diagnostic_assessment(self) -> str:
        if self.diag is None:
            return "Diagnostics disabled."
        stages = self.diag.report.get("stages", {})
        lines = ["JOSS-F RECEIVE DIAGNOSTIC ASSESSMENT", "=" * 38, ""]

        chirp = stages.get("chirp_sync", {})
        if not chirp.get("best_chain_found", False):
            lines.append("FIRST LIKELY FAILURE: chirp synchronisation/reference mismatch.")
            lines.append("The ten expected chirps were not detected as a coherent chain.")
            return "\n".join(lines) + "\n"
        lines.append(
            f"Chirp synchronisation: chain found; spacing std="
            f"{float(chirp.get('spacing_std', float('nan'))):.3f} samples; "
            f"SFO={float(chirp.get('sample_rate_offset_ppm', float('nan'))):+.2f} ppm."
        )

        timing = stages.get("ofdm_timing", {})
        selected_score = float(timing.get("selected_score", 0.0))
        broad_distance = int(timing.get("broad_best_residual_from_selected", 0))
        lines.append(
            f"OFDM timing: selected CP score={selected_score:.3f}; broad-best distance="
            f"{broad_distance} samples."
        )
        if broad_distance > self.config.data_start_refine_radius:
            lines.append("FIRST LIKELY FAILURE: preamble length/layout or OFDM-start convention mismatch.")
            return "\n".join(lines) + "\n"
        if selected_score < 0.12:
            lines.append("FIRST LIKELY FAILURE: weak/incorrect OFDM symbol boundary or CP convention.")
            return "\n".join(lines) + "\n"

        pilots = stages.get("known_pilots", {})
        expected_metrics = pilots.get("expected_pilot_metrics", [])
        expected_coherence = [float(row["known_pilot_coherence"]) for row in expected_metrics]
        most_pilot_like = pilots.get("most_pilot_like_block", {})
        if expected_coherence:
            lines.append(
                f"Known pilots: expected-position median coherence="
                f"{np.median(expected_coherence):.3f}; most pilot-like block="
                f"{most_pilot_like.get('ofdm_block')}."
            )
            expected_positions = set(pilots.get("expected_positions", []))
            if int(most_pilot_like.get("ofdm_block", -1)) not in expected_positions:
                lines.append("FIRST LIKELY FAILURE: pilot scheduling/counting convention mismatch.")
                return "\n".join(lines) + "\n"
            if float(np.median(expected_coherence)) < 0.35:
                lines.append(
                    "FIRST LIKELY FAILURE: known-pilot file, occupied-bin range, FFT convention, "
                    "or pilot waveform construction mismatch."
                )
                return "\n".join(lines) + "\n"

        equalisation = stages.get("equalisation", {})
        median_evm = float(equalisation.get("evm_rms_median", float("inf")))
        lines.append(f"Equalised constellation: median RMS EVM={median_evm:.3f}.")
        if median_evm > 0.65:
            lines.append(
                "FIRST LIKELY FAILURE: channel estimation/equalisation, residual timing, "
                "or carrier mapping mismatch."
            )
            return "\n".join(lines) + "\n"

        ldpc = stages.get("ldpc", {})
        success_fraction = float(ldpc.get("success_fraction", 0.0))
        lines.append(
            f"LDPC: {ldpc.get('successful_blocks', 0)}/{ldpc.get('total_blocks', 0)} "
            f"blocks passed ({100.0 * success_fraction:.1f}%)."
        )
        hypotheses = stages.get("hypothesis_scan", {})
        if success_fraction == 0.0:
            if int(hypotheses.get("alternate_successful_count", 0)) > 0:
                lines.append(
                    "FIRST LIKELY FAILURE: QPSK rotation/reflection or interleaver-direction mismatch; "
                    "see qpsk_interleaver_hypotheses.csv."
                )
            else:
                lines.append(
                    "FIRST LIKELY FAILURE: LDPC matrix/scaling, interleaver, QPSK bit mapping, "
                    "or earlier symbol-quality failure."
                )
            return "\n".join(lines) + "\n"

        header = stages.get("header", {})
        if not header.get("valid", False):
            lines.append(
                "FIRST LIKELY FAILURE: header field order/endianness/definition mismatch. "
                "Physical-layer and at least some LDPC decoding are working."
            )
            return "\n".join(lines) + "\n"

        lines.append("No failure boundary detected: the packet decoded successfully.")
        return "\n".join(lines) + "\n"

    def _write_plots(self) -> None:
        if self.diag is None or not self.make_plots:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.diag.warning("matplotlib is unavailable; diagnostic plots were skipped")
            return

        arrays = self.diag.arrays
        if "input_envelope_time" in arrays:
            fig = plt.figure(figsize=(11, 4))
            plt.plot(arrays["input_envelope_time"], arrays["input_envelope_peak"])
            plt.xlabel("Time (s)")
            plt.ylabel("Peak |amplitude|")
            plt.title("Recording amplitude envelope")
            plt.grid(True, alpha=0.25)
            self.diag.save_figure("01_recording_envelope.png", fig)
            plt.close(fig)

        if "chirp_metric_decimated" in arrays:
            fig = plt.figure(figsize=(11, 4))
            plt.plot(
                arrays["chirp_metric_decimated_samples"],
                arrays["chirp_metric_decimated"],
            )
            if "chirp_chain_samples" in arrays:
                samples = arrays["chirp_chain_samples"]
                plt.scatter(samples, np.interp(samples, arrays["chirp_metric_decimated_samples"], arrays["chirp_metric_decimated"]))
            plt.xlabel("Recording sample")
            plt.ylabel("Matched-filter magnitude")
            plt.title("Chirp synchronisation metric")
            plt.grid(True, alpha=0.25)
            self.diag.save_figure("02_chirp_metric.png", fig)
            plt.close(fig)

        if "golay_impulse_response" in arrays:
            fig = plt.figure(figsize=(11, 4))
            plt.plot(np.abs(arrays["golay_impulse_response"]))
            plt.xlabel("Impulse-response sample")
            plt.ylabel("|h[n]|")
            plt.title("Golay channel impulse response")
            plt.grid(True, alpha=0.25)
            self.diag.save_figure("03_golay_impulse_response.png", fig)
            plt.close(fig)

        if "golay_channel" in arrays:
            channel = arrays["golay_channel"]
            frequencies = np.arange(len(channel)) * self.config.sample_rate / self.config.fft_length
            fig = plt.figure(figsize=(11, 4))
            plt.plot(frequencies, 20.0 * np.log10(np.abs(channel) + EPS))
            plt.xlim(0, self.config.sample_rate / 2)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.title("Golay channel frequency response")
            plt.grid(True, alpha=0.25)
            self.diag.save_figure("04_golay_frequency_response.png", fig)
            plt.close(fig)

        if "cp_fine_offsets" in arrays:
            fig = plt.figure(figsize=(11, 4))
            plt.plot(arrays["cp_fine_offsets"], arrays["cp_fine_scores"])
            plt.xlabel("Offset from nominal data start (samples)")
            plt.ylabel("Median CP correlation")
            plt.title("Fine OFDM-boundary scan")
            plt.grid(True, alpha=0.25)
            self.diag.save_figure("05_cp_timing_scan.png", fig)
            plt.close(fig)

        if "cp_broad_offsets" in arrays:
            fig = plt.figure(figsize=(11, 4))
            plt.plot(arrays["cp_broad_offsets"], arrays["cp_broad_scores"])
            plt.xlabel("Offset from nominal data start (samples)")
            plt.ylabel("Median CP correlation")
            plt.title("Broad OFDM-boundary scan")
            plt.grid(True, alpha=0.25)
            self.diag.save_figure("06_cp_broad_scan.png", fig)
            plt.close(fig)

        if "known_pilot_coherence_by_block" in arrays:
            values = arrays["known_pilot_coherence_by_block"]
            blocks = np.arange(1, len(values) + 1)
            fig = plt.figure(figsize=(11, 4))
            plt.plot(blocks, values, marker=".")
            expected = blocks[blocks % self.config.pilot_interval == 0]
            for position in expected:
                plt.axvline(position, linestyle="--", alpha=0.25)
            plt.xlabel("Absolute OFDM block")
            plt.ylabel("Known-pilot coherence")
            plt.title("Known-pilot similarity across every OFDM block")
            plt.grid(True, alpha=0.25)
            self.diag.save_figure("07_known_pilot_scan.png", fig)
            plt.close(fig)

        if "equalised_constellation_first_blocks" in arrays:
            symbols = arrays["equalised_constellation_first_blocks"].reshape(-1)
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(symbols.real, symbols.imag, s=5, alpha=0.35)
            plt.axhline(0.0, linewidth=0.8)
            plt.axvline(0.0, linewidth=0.8)
            plt.xlabel("In-phase")
            plt.ylabel("Quadrature")
            plt.title("Equalised constellation: first data blocks")
            plt.axis("equal")
            plt.grid(True, alpha=0.25)
            self.diag.save_figure("08_equalised_constellation.png", fig)
            plt.close(fig)

    def decode_wav(self, wav_path: str | Path, output_directory: str | Path = ".") -> Path:
        self.diag = self._start_diagnostics(wav_path)
        error: BaseException | None = None
        output_path: Path | None = None
        try:
            received = read_wav(wav_path, self.config)
            peak = float(np.max(np.abs(received))) if len(received) else 0.0
            clipped = float(np.mean(np.abs(received) >= 0.995)) if len(received) else 0.0

            if self.diag is not None:
                window = 512
                count = (len(received) + window - 1) // window
                padded = np.pad(received, (0, count * window - len(received)))
                reshaped = padded.reshape(count, window)
                envelope = np.max(np.abs(reshaped), axis=1)
                times = np.arange(count) * window / self.config.sample_rate
                spectrum = np.fft.rfft(received)
                frequencies = np.fft.rfftfreq(len(received), d=1.0 / self.config.sample_rate)
                total_power = float(np.sum(np.abs(spectrum) ** 2)) + EPS
                band = (frequencies >= self.config.data_low_hz) & (
                    frequencies <= self.config.data_high_hz
                )
                self.diag.array("input_envelope_time", times)
                self.diag.array("input_envelope_peak", envelope)
                self.diag.stage(
                    "input_wav",
                    samples=len(received),
                    duration_seconds=len(received) / self.config.sample_rate,
                    peak=peak,
                    rms=rms(received),
                    median=float(np.median(received)),
                    clipped_fraction=clipped,
                    clipped_percent=100.0 * clipped,
                    zero_fraction=float(np.mean(received == 0.0)),
                    data_band_power_fraction=float(
                        np.sum(np.abs(spectrum[band]) ** 2) / total_power
                    ),
                )
                self.diag.log(
                    f"Input: {len(received)} samples ({len(received)/self.config.sample_rate:.2f} s), "
                    f"peak={peak:.3f}, RMS={rms(received):.4f}, clipped={100*clipped:.4f}%"
                )

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
            if self.diag is not None:
                self.diag.stage("result", decoded=True, saved_path=output_path, reception_info=info)
            return output_path
        except BaseException as exc:
            error = exc
            if self.diag is not None:
                self.diag.stage("result", decoded=False)
                self.diag.log(f"Decode failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            if self.diag is not None:
                try:
                    assessment = self._diagnostic_assessment()
                    self.diag.write_text("ASSESSMENT.txt", assessment)
                    self.diag.log("\n" + assessment.rstrip())
                    self._write_plots()
                except BaseException as report_error:
                    self.diag.warning(
                        f"An error occurred while producing diagnostics: {report_error}"
                    )
                report_path = self.diag.finalise(
                    "success" if error is None and output_path is not None else "failed",
                    error,
                )
                print(f"Diagnostic report: {report_path}")


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
    parser = argparse.ArgumentParser(
        description="Decode a JOSS-F WAV and write detailed non-invasive diagnostics."
    )
    parser.add_argument("wav", nargs="?", help="WAV file to decode; prompts when omitted")
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="Directory for this diagnostic run instead of an automatic timestamped directory",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots")
    parser.add_argument(
        "--shallow",
        action="store_true",
        help="Skip alternate QPSK/interleaver hypothesis decoding",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory for successfully decoded payloads",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    input_path = Path(arguments.wav).expanduser() if arguments.wav else ask_input_wav()
    if arguments.wav and input_path.suffix.lower() != ".wav":
        input_path = input_path.with_suffix(".wav")
    if arguments.wav and not input_path.is_absolute():
        input_path = BASE_DIR / input_path
    Receiver(
        diagnostics=True,
        diagnostics_root=arguments.diagnostics_dir,
        make_plots=not arguments.no_plots,
        deep_diagnostics=not arguments.shallow,
    ).decode_wav(input_path, arguments.output_directory)
