"""JOSS-F receiver.

Changes from baseline
─────────────────────
* ``synchronise_and_correct``:  SFO estimation now accepts partial chirp chains
  (≥ ``_MIN_CHAIN_LEN`` peaks instead of requiring all ten).  When the full
  train is found all behaviour is identical; the relaxed threshold only fires
  when speaker/multipath or a quiet room drops one or more chirp peaks below the
  detection threshold.  The ``ValueError`` for implausible ratios is softened to
  a warning + graceful fallback so a single bad recording cannot crash the
  pipeline.

* ``_refine_data_start``:  all coarse CP-correlation scores are now stored in
  ``_diag`` so the diagnostic plot can show the full correlation landscape.

* ``_equalise``:  the phase-slope correction now uses only the most reliable
  symbols (those whose distance from the nearest QPSK point is below a
  threshold).  On a well-equalised block this is all symbols; on a badly
  distorted block the pruning prevents a handful of wrong hard decisions from
  rotating the whole constellation.

* ``decode_raw_signal``:  accepts an optional ``_diag`` keyword dict; when
  supplied the receiver fills it with all intermediate data needed by
  ``diagnostics.plot_all``.

* ``decode_wav``:  after decoding it calls ``diagnostics.plot_all`` (saved to
  ``<output_directory>/diagnostic_plots/``) and shows the figures.  Pass
  ``diagnostics=False`` to suppress this.

Everything else – packet framing, LDPC codec, interleaver, Golay channel
estimator, output filenames, text-salvage, header repair – is unchanged.
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

# Minimum number of chirps required to form a valid chain for SFO estimation.
# 6-out-of-10 still gives 4 DOF for the 2-parameter linear fit and keeps the
# maximum period-estimation error below ~4 ppm for typical hardware.
_MIN_CHAIN_LEN = 6


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

    def synchronise_and_correct(
        self,
        received: NDArray[np.float64],
        *,
        _diag: dict | None = None,
    ) -> tuple[NDArray[np.float64], int, float]:
        """Find the chirp train and correct transmitter/receiver clock mismatch.

        Accepts chains of length ≥ _MIN_CHAIN_LEN (default 6).  If even that
        cannot be met the coarse matched-filter fallback is used with zero SFO
        correction.
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
            prominence=max(float(np.max(metric)) * 0.035, EPS),
        )

        # scipy.signal.find_peaks does not return index zero, but compliant WAV
        # files may begin immediately with the first chirp.
        seed_candidates = np.unique(
            np.concatenate([
                peaks.astype(int),
                np.array([0, int(np.argmax(metric))], dtype=int),
            ])
        )

        best_chain: list[int] | None = None
        best_score = -np.inf
        tolerance = int(0.08 * n)
        for first in seed_candidates:
            chain = [int(first)]
            score = float(metric[first])
            current = int(first)
            for _ in range(1, self.config.chirp_count):
                target = current + n
                # Use a local variable to avoid shadowing seed_candidates.
                next_cands = peaks[np.abs(peaks - target) <= tolerance]
                if not len(next_cands):
                    break
                chosen = int(next_cands[np.argmax(metric[next_cands])])
                chain.append(chosen)
                score += float(metric[chosen])
                current = chosen

            # Accept chains that reach the minimum required length.
            if len(chain) >= _MIN_CHAIN_LEN:
                spacing = np.diff(chain)
                score /= 1.0 + float(np.std(spacing))
                # Prefer longer chains for equal score-per-chirp.
                length_bonus = len(chain) * 1e-6
                if score + length_bonus > best_score:
                    best_score = score + length_bonus
                    best_chain = chain

        # ── Fallback: coarse cross-correlation with the full chirp train ──
        if best_chain is None:
            template = self.chirp.train()
            coarse_metric = np.abs(fftconvolve(received, template[::-1], mode="valid"))
            sync = int(np.argmax(coarse_metric))
            warnings.warn(
                f"[JOSS-F sync] could not form a chirp chain of ≥ {_MIN_CHAIN_LEN}; "
                "falling back to coarse sync with NO SFO correction.  "
                "LDPC decoding may fail on live recordings.",
                RuntimeWarning, stacklevel=2,
            )
            corrected = received[sync:]
            if _diag is not None:
                _diag.update(dict(
                    received=received, corrected=corrected,
                    chirp_metric=metric, all_peaks=peaks,
                    best_chain=[], sfo_ratio=1.0, sfo_ppm=0.0,
                    sync_sample=sync, min_chain_required=_MIN_CHAIN_LEN,
                ))
            return corrected, sync, 0.0

        # ── Sub-sample peak refinement ──
        observed_values: list[float] = []
        for peak in best_chain:
            if 0 < peak < len(metric) - 1:
                left, centre, right = metric[peak - 1], metric[peak], metric[peak + 1]
                denominator = left - 2.0 * centre + right
                delta = 0.0 if abs(denominator) < EPS else 0.5 * (left - right) / denominator
                observed_values.append(float(peak) + float(np.clip(delta, -0.5, 0.5)))
            else:
                observed_values.append(float(peak))

        # ── SFO estimation via linear fit on the detected chain ──
        chain_len = len(best_chain)
        chirp_number = np.arange(chain_len, dtype=float)
        period, intercept = np.polyfit(chirp_number, np.asarray(observed_values), 1)
        ratio = float(period / n)

        if not (0.995 <= ratio <= 1.005):
            warnings.warn(
                f"[JOSS-F SFO] implausible ratio estimated ({ratio:.7f}, "
                f"{(ratio - 1.0) * 1e6:+.0f} ppm); "
                "skipping SFO correction.",
                RuntimeWarning, stacklevel=2,
            )
            sync = int(round(intercept))
            corrected = received[sync:]
            if _diag is not None:
                _diag.update(dict(
                    received=received, corrected=corrected,
                    chirp_metric=metric, all_peaks=peaks,
                    best_chain=best_chain, sfo_ratio=1.0, sfo_ppm=0.0,
                    sync_sample=sync, min_chain_required=_MIN_CHAIN_LEN,
                ))
            return corrected, sync, 0.0

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

        if _diag is not None:
            _diag.update(dict(
                received=received,
                corrected=corrected,
                chirp_metric=metric,
                all_peaks=peaks,
                best_chain=best_chain,
                sfo_ratio=ratio,
                sfo_ppm=offset_ppm,
                sync_sample=sync,
                min_chain_required=_MIN_CHAIN_LEN,
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
        channel = np.fft.rfft(h, n=c.fft_length)

        if _diag is not None:
            _diag["golay_channel"] = channel

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
    # Equalisation
    # ──────────────────────────────────────────────────────────────────────────

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

        # ── Improved phase-slope correction ──
        # Use only symbols that are close to a valid QPSK point so that a few
        # badly distorted bins (e.g. deep channel nulls) cannot corrupt the fit.
        hard = QPSK.hard_symbols(estimate)
        residuals = np.abs(estimate - hard)

        _RELIABILITY_THRESHOLD = 1.2   # max distance from nearest QPSK point
        reliable = residuals < _RELIABILITY_THRESHOLD
        n_reliable = int(np.sum(reliable))

        # Fall back to all symbols when most are already close to ideal.
        if n_reliable >= max(30, len(bins) // 4):
            est_fit   = estimate[reliable]
            hard_fit  = hard[reliable]
            bins_fit  = bins[reliable]
        else:
            est_fit   = estimate
            hard_fit  = hard
            bins_fit  = bins

        phase_error = np.unwrap(np.angle(est_fit * np.conj(hard_fit)))
        x_fit = bins_fit.astype(float) - float(np.mean(bins_fit))
        fit_weights = np.clip(np.abs(est_fit), 0.10, 3.0)
        design = np.column_stack([x_fit, np.ones_like(x_fit)])
        slope, intercept_ph = np.linalg.lstsq(
            design * fit_weights[:, None],
            phase_error * fit_weights,
            rcond=None,
        )[0]
        x_all = bins.astype(float) - float(np.mean(bins.astype(float)))
        estimate *= np.exp(-1j * (slope * x_all + intercept_ph))

        # AGC normalisation: makes decoding insensitive to slow gain recovery.
        robust_magnitude = float(np.median(np.abs(estimate)))
        if robust_magnitude > EPS:
            estimate *= np.sqrt(2.0) / robust_magnitude
        return estimate

    # ──────────────────────────────────────────────────────────────────────────
    # Pilot channel update
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # Data extraction and equalisation
    # ──────────────────────────────────────────────────────────────────────────

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

        # Diagnostic storage (first N raw spectra at data bins)
        _N_DIAG = 8
        raw_spectra_bins: list[NDArray[np.complex128]] = []
        eq_rows_diag: list[NDArray[np.complex128]] = []

        for one_based, spectrum in enumerate(spectra, start=1):
            if self.scheduler.is_pilot(one_based):
                continue
            if len(pilot_positions):
                nearest = int(pilot_positions[np.argmin(np.abs(pilot_positions - one_based))])
                channel = pilot_channels[nearest]
            else:
                channel = golay_channel

            # Store raw spectrum at data bins (before equalization) for diagnostics.
            if _diag is not None and len(raw_spectra_bins) < _N_DIAG:
                raw_spectra_bins.append(spectrum[c.data_bins].copy())

            estimate = self._equalise(spectrum, channel)

            if _diag is not None and len(eq_rows_diag) < _N_DIAG:
                eq_rows_diag.append(estimate.copy())

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

        if _diag is not None:
            _diag["pilot_channels"]  = pilot_channels
            _diag["raw_spectra_bins"] = raw_spectra_bins
            _diag["equalised_rows"]  = eq_rows_diag

        return np.stack(rows), np.stack(weights), block_count

    # ──────────────────────────────────────────────────────────────────────────
    # Core decode pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def decode_raw_signal(
        self,
        received: NDArray[np.float64],
        *,
        _diag: dict | None = None,
    ) -> RawReception:
        """Decode every complete LDPC group without trusting the packet header.

        This is the key interoperability fallback: a corrupt six-byte header no
        longer discards the rest of the successfully LDPC-decoded stream.
        """
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
        ldpc_successes:  list[bool] = []

        for group_index in range(complete_groups):
            lo = group_index * self.config.data_symbols_per_group
            hi = lo + self.config.data_symbols_per_group
            llr_blocks = self.interleaver.deinterleave_llrs(rows[lo:hi], weights[lo:hi])
            information_blocks: list[NDArray[np.uint8]] = []
            for llr in llr_blocks:
                information, iters, success = self.ldpc.decode_block(llr)
                information_blocks.append(information)
                successful_blocks += int(success)
                total_blocks += 1
                ldpc_iterations.append(int(iters))
                ldpc_successes.append(bool(success))
            recovered_bits.append(np.concatenate(information_blocks))

        if _diag is not None:
            _diag["ldpc_iterations"] = ldpc_iterations
            _diag["ldpc_successes"]  = ldpc_successes

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
