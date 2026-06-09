from __future__ import annotations

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

    def decode_raw_signal(self, received: NDArray[np.float64]) -> RawReception:
        """Decode every complete LDPC group without trusting the packet header.

        This is the key interoperability fallback: a corrupt six-byte header no
        longer discards the rest of the successfully LDPC-decoded stream.
        """
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

    @staticmethod
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

    def decode_wav(
        self,
        wav_path: str | Path,
        output_directory: str | Path = ".",
        *,
        strict_header: bool = False,
        forced_header_length: int | None = None,
        forced_payload_bytes: int | None = None,
        forced_filename: str = "recovered_payload.bin",
    ) -> Path:
        received = read_wav(wav_path, self.config)
        peak = float(np.max(np.abs(received))) if len(received) else 0.0
        clipped = float(np.mean(np.abs(received) >= 0.995)) if len(received) else 0.0
        raw = self.decode_raw_signal(received)

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

        wav_stem = Path(wav_path).stem
        output_path = output_directory / f"received_{wav_stem}_raw_ldpc_stream.bin"
        output_path.write_bytes(raw.decoded_bytes)
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
            "note": (
                "The JOSS-F header supplies both payload offset and payload length. "
                "Without a recoverable header, the exact payload boundary is not uniquely "
                "determined; this file preserves every LDPC-decoded byte, including header "
                "and post-payload padding."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print("  header status    : unrecoverable; raw decoded stream preserved")
        print(f"  saved raw stream : {output_path}")
        print(f"  saved metadata   : {metadata_path}")
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
    )
