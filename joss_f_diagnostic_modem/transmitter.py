from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import json

import numpy as np
from numpy.typing import NDArray

from diagnostics import DiagnosticSession, rms, sha256_array, sha256_bytes
from modem import (
    CONFIG, KnownOFDMPilot, ModemConfig, OFDM, PacketCodec, PilotScheduler,
    Preamble, StandardInterleaver, WiMaxLDPC, bytes_to_bits, normalise_peak,
    write_wav,
)


@dataclass(frozen=True)
class TransmissionInfo:
    payload_bytes: int
    framed_bytes: int
    ldpc_groups: int
    data_ofdm_blocks: int
    pilot_ofdm_blocks: int
    duration_seconds: float


class Transmitter:
    """Original JOSS-F transmitter with reference fingerprints and diagnostics."""

    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        self.ldpc = WiMaxLDPC(config)
        self.interleaver = StandardInterleaver(config)
        self.ofdm = OFDM(config)
        self.scheduler = PilotScheduler(config.pilot_interval)
        self.preamble = Preamble(config)
        self.known_pilot = KnownOFDMPilot(config)
        self._last_packet = b""
        self._last_packet_bits = np.empty(0, dtype=np.uint8)
        self._last_data_rows = np.empty((0, config.carriers_per_symbol), dtype=np.complex128)
        self._last_block_schedule: list[dict[str, object]] = []
        self._last_preamble = np.empty(0, dtype=float)
        self._last_ofdm_stream = np.empty(0, dtype=float)
        self._last_pre_normalisation_peak = 0.0
        self._last_normalisation_scale = 1.0

    def _coded_rows(self, packet_bits: NDArray[np.uint8]) -> NDArray[np.complex128]:
        c = self.config
        information_per_group = c.ldpc_blocks_per_group * c.ldpc_info_bits
        group_count = max(1, (len(packet_bits) + information_per_group - 1) // information_per_group)
        padded = np.empty(group_count * information_per_group, dtype=np.uint8)
        padded[: len(packet_bits)] = packet_bits
        if len(packet_bits) < len(padded):
            # JOSS-F does not prescribe post-payload padding. Deterministic
            # pseudo-random padding avoids long all-zero LDPC codewords, which
            # otherwise produce extreme OFDM peaks and trigger phone/OS limiters.
            rng = np.random.default_rng(c.padding_seed)
            padded[len(packet_bits) :] = rng.integers(
                0, 2, len(padded) - len(packet_bits), dtype=np.uint8
            )
        groups = padded.reshape(-1, information_per_group)
        rows: list[NDArray[np.complex128]] = []
        for group in groups:
            information_blocks = group.reshape(c.ldpc_blocks_per_group, c.ldpc_info_bits)
            coded = np.stack([self.ldpc.encode_block(block) for block in information_blocks])
            rows.append(self.interleaver.interleave(coded))
        return np.concatenate(rows, axis=0)

    def _ofdm_stream(self, data_rows: NDArray[np.complex128]) -> tuple[NDArray[np.float64], int]:
        blocks: list[NDArray[np.float64]] = []
        row_index = 0
        block_index = 1
        pilot_blocks = 0
        known = self.known_pilot.waveform()
        schedule: list[dict[str, object]] = []
        while row_index < len(data_rows):
            if self.scheduler.is_pilot(block_index):
                block = known
                blocks.append(block)
                pilot_blocks += 1
                schedule.append(
                    {
                        "absolute_ofdm_block": block_index,
                        "kind": "known_pilot",
                        "data_row": None,
                        "peak": float(np.max(np.abs(block))),
                        "rms": rms(block),
                        "waveform_hash": sha256_array(block),
                    }
                )
            else:
                block = self.ofdm.modulate_data_row(data_rows[row_index])
                blocks.append(block)
                schedule.append(
                    {
                        "absolute_ofdm_block": block_index,
                        "kind": "data",
                        "data_row": row_index,
                        "peak": float(np.max(np.abs(block))),
                        "rms": rms(block),
                        "waveform_hash": sha256_array(block),
                    }
                )
                row_index += 1
            block_index += 1
        self._last_block_schedule = schedule
        return np.concatenate(blocks), pilot_blocks

    def create_signal(self, file_bytes: bytes, filename: str) -> tuple[NDArray[np.float64], TransmissionInfo]:
        c = self.config
        packet = PacketCodec.build(file_bytes, filename)
        bits = bytes_to_bits(packet)
        data_rows = self._coded_rows(bits)
        ofdm_stream, pilot_blocks = self._ofdm_stream(data_rows)

        preamble = self.preamble.waveform()
        unnormalised_signal = np.concatenate(
            [
                np.zeros(c.leading_silence),
                preamble,
                ofdm_stream,
                np.zeros(c.trailing_silence),
            ]
        )
        signal = normalise_peak(unnormalised_signal, c.output_peak)
        groups = len(data_rows) // c.data_symbols_per_group
        info = TransmissionInfo(
            payload_bytes=len(file_bytes),
            framed_bytes=len(packet),
            ldpc_groups=groups,
            data_ofdm_blocks=len(data_rows),
            pilot_ofdm_blocks=pilot_blocks,
            duration_seconds=len(signal) / c.sample_rate,
        )

        # Diagnostic observations only.
        self._last_packet = packet
        self._last_packet_bits = bits.copy()
        self._last_data_rows = data_rows.copy()
        self._last_preamble = preamble.copy()
        self._last_ofdm_stream = ofdm_stream.copy()
        self._last_pre_normalisation_peak = (
            float(np.max(np.abs(unnormalised_signal))) if len(unnormalised_signal) else 0.0
        )
        self._last_normalisation_scale = (
            1.0
            if self._last_pre_normalisation_peak <= c.output_peak
            or self._last_pre_normalisation_peak == 0.0
            else c.output_peak / self._last_pre_normalisation_peak
        )
        return signal, info

    def _write_diagnostics(
        self,
        payload_path: Path,
        wav_path: Path,
        file_bytes: bytes,
        signal: NDArray[np.float64],
        info: TransmissionInfo,
    ) -> Path:
        directory = wav_path.parent / "diagnostics" / f"{wav_path.stem}_tx"
        diag = DiagnosticSession(directory, payload_path)
        c = self.config
        pilot_path = getattr(self.known_pilot, "path", None)
        pilot_file_hash = None
        if pilot_path is not None and Path(pilot_path).is_file():
            pilot_file_hash = sha256_bytes(Path(pilot_path).read_bytes())

        zero_codeword = self.ldpc.encode_block(np.zeros(c.ldpc_info_bits, dtype=np.uint8))
        header_length = int.from_bytes(self._last_packet[:2], "big")
        payload_length = int.from_bytes(self._last_packet[2:6], "big")
        diag.stage(
            "configuration",
            sample_rate=c.sample_rate,
            chirp_length=c.chirp_length,
            chirp_count=c.chirp_count,
            chirp_start_hz=c.chirp_start_hz,
            chirp_stop_hz=c.chirp_stop_hz,
            chirp_hash=sha256_array(self.preamble.chirp.samples()),
            chirp_train_hash=sha256_array(self.preamble.chirp.train()),
            golay_a_hash=sha256_array(self.preamble.golay.a),
            golay_b_hash=sha256_array(self.preamble.golay.b),
            golay_waveform_hash=sha256_array(self.preamble.golay.waveform()),
            known_pilot_file=str(pilot_path) if pilot_path is not None else None,
            known_pilot_file_sha256=pilot_file_hash,
            known_pilot_symbol_hash=sha256_array(self.known_pilot.symbols),
            known_pilot_frequency_hash=sha256_array(self.known_pilot.frequency),
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
            zero_information_ldpc_codeword_hash=sha256_array(zero_codeword),
        )
        diag.stage(
            "packet",
            filename=payload_path.name,
            payload_bytes=len(file_bytes),
            payload_sha256=sha256_bytes(file_bytes),
            header_length=header_length,
            payload_length_field=payload_length,
            framed_bytes=len(self._last_packet),
            first_96_packet_bytes_hex=self._last_packet[:96].hex(" "),
            first_96_packet_bytes_ascii="".join(
                chr(value) if 32 <= value < 127 else "." for value in self._last_packet[:96]
            ),
            packet_sha256=sha256_bytes(self._last_packet),
            packet_bits=len(self._last_packet_bits),
        )
        diag.stage(
            "coding",
            ldpc_groups=info.ldpc_groups,
            information_bits_per_group=c.ldpc_blocks_per_group * c.ldpc_info_bits,
            coded_bits_per_group=c.ldpc_blocks_per_group * c.ldpc_code_bits,
            data_rows=info.data_ofdm_blocks,
            data_rows_hash=sha256_array(self._last_data_rows),
        )
        diag.rows("ofdm_block_schedule", self._last_block_schedule)
        diag.stage(
            "waveform",
            leading_silence=c.leading_silence,
            preamble_samples=len(self._last_preamble),
            ofdm_samples=len(self._last_ofdm_stream),
            trailing_silence=c.trailing_silence,
            total_samples=len(signal),
            duration_seconds=len(signal) / c.sample_rate,
            preamble_peak=float(np.max(np.abs(self._last_preamble))),
            preamble_rms=rms(self._last_preamble),
            ofdm_peak=float(np.max(np.abs(self._last_ofdm_stream))),
            ofdm_rms=rms(self._last_ofdm_stream),
            pre_normalisation_peak=self._last_pre_normalisation_peak,
            normalisation_scale=self._last_normalisation_scale,
            final_peak=float(np.max(np.abs(signal))),
            final_rms=rms(signal),
            signal_hash_float64=sha256_array(signal),
            data_blocks=info.data_ofdm_blocks,
            pilot_blocks=info.pilot_ofdm_blocks,
            pilot_positions=[
                row["absolute_ofdm_block"]
                for row in self._last_block_schedule
                if row["kind"] == "known_pilot"
            ],
        )
        diag.array("transmitted_signal", signal)
        diag.array("preamble", self._last_preamble)
        diag.array("known_pilot_waveform", self.known_pilot.waveform())
        diag.write_text(
            "REFERENCE_FINGERPRINTS.txt",
            "\n".join(
                [
                    "JOSS-F TRANSMITTER REFERENCE FINGERPRINTS",
                    "=" * 42,
                    f"chirp:             {sha256_array(self.preamble.chirp.samples())}",
                    f"chirp train:       {sha256_array(self.preamble.chirp.train())}",
                    f"Golay A:           {sha256_array(self.preamble.golay.a)}",
                    f"Golay B:           {sha256_array(self.preamble.golay.b)}",
                    f"Golay waveform:    {sha256_array(self.preamble.golay.waveform())}",
                    f"pilot .npy file:   {pilot_file_hash}",
                    f"pilot symbols:     {sha256_array(self.known_pilot.symbols)}",
                    f"pilot waveform:    {sha256_array(self.known_pilot.waveform())}",
                    f"interleaver:       {sha256_array(self.interleaver.permutation)}",
                    f"zero LDPC vector:  {sha256_array(zero_codeword)}",
                    f"packet:             {sha256_bytes(self._last_packet)}",
                    f"final float signal: {sha256_array(signal)}",
                    "",
                    "Two groups using identical standard components should match all component",
                    "fingerprints above the packet/final-signal lines.",
                ]
            )
            + "\n",
        )
        diag.log(f"Header bytes: {self._last_packet[:min(32, len(self._last_packet))].hex(' ')}")
        diag.log(
            f"Normalisation: pre-peak={self._last_pre_normalisation_peak:.6f}, "
            f"scale={self._last_normalisation_scale:.8f}, final peak={np.max(np.abs(signal)):.6f}"
        )
        report = diag.finalise("success")
        return report

    def transmit_file(self, payload_path: str | Path, wav_path: str | Path = "tx.wav") -> TransmissionInfo:
        payload_path = Path(payload_path)
        wav_path = Path(wav_path)
        file_bytes = payload_path.read_bytes()
        signal, info = self.create_signal(file_bytes, payload_path.name)
        write_wav(wav_path, signal, self.config)
        print(f"Wrote {wav_path}")
        print(f"  file             : {payload_path.name} ({info.payload_bytes} bytes)")
        print(f"  framed bytes     : {info.framed_bytes}")
        print(f"  LDPC groups      : {info.ldpc_groups}")
        print(f"  Golay pulses     : {2 * self.config.golay_repeats} "
              f"({self.config.golay_repeats} A/B pairs)")
        print(f"  OFDM data starts : sample {self.config.preamble_length} after first chirp")
        print(f"  data/pilot OFDM  : {info.data_ofdm_blocks}/{info.pilot_ofdm_blocks}")
        print(f"  duration         : {info.duration_seconds:.2f} s")
        report_path = self._write_diagnostics(payload_path, wav_path, file_bytes, signal, info)
        print(f"  diagnostics      : {report_path}")
        return info


BASE_DIR = Path(__file__).resolve().parent
PAYLOAD_FILE = BASE_DIR / "payload.txt"


def ask_output_wav() -> Path:
    """Ask for an output name, accepting either ``tx`` or ``tx.wav``."""
    name = input("Output WAV name [tx]: ").strip() or "tx"
    path = Path(name).expanduser()
    if path.suffix.lower() != ".wav":
        path = path.with_suffix(".wav")
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transmit a JOSS-F packet and save reference fingerprints."
    )
    parser.add_argument("payload", nargs="?", type=Path, default=PAYLOAD_FILE)
    parser.add_argument("--wav", type=Path, help="Output WAV path; prompts when omitted")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    payload = arguments.payload.expanduser()
    if not payload.is_absolute():
        payload = BASE_DIR / payload
    if not payload.is_file():
        raise FileNotFoundError(f"Payload file not found: {payload}")
    output = arguments.wav.expanduser() if arguments.wav else ask_output_wav()
    if not output.is_absolute():
        output = BASE_DIR / output
    Transmitter().transmit_file(payload, output)
