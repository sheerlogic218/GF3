from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

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
    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        self.ldpc = WiMaxLDPC(config)
        self.interleaver = StandardInterleaver(config)
        self.ofdm = OFDM(config)
        self.scheduler = PilotScheduler(config.pilot_interval)
        self.preamble = Preamble(config)
        self.known_pilot = KnownOFDMPilot(config)

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
        while row_index < len(data_rows):
            if self.scheduler.is_pilot(block_index):
                blocks.append(known)
                pilot_blocks += 1
            else:
                blocks.append(self.ofdm.modulate_data_row(data_rows[row_index]))
                row_index += 1
            block_index += 1
        return np.concatenate(blocks), pilot_blocks

    def create_signal(self, file_bytes: bytes, filename: str) -> tuple[NDArray[np.float64], TransmissionInfo]:
        c = self.config
        packet = PacketCodec.build(file_bytes, filename)
        bits = bytes_to_bits(packet)
        data_rows = self._coded_rows(bits)
        ofdm_stream, pilot_blocks = self._ofdm_stream(data_rows)

        signal = np.concatenate(
            [
                np.zeros(c.leading_silence),
                self.preamble.waveform(),
                ofdm_stream,
                np.zeros(c.trailing_silence),
            ]
        )
        signal = normalise_peak(signal, c.output_peak)
        groups = len(data_rows) // c.data_symbols_per_group
        info = TransmissionInfo(
            payload_bytes=len(file_bytes),
            framed_bytes=len(packet),
            ldpc_groups=groups,
            data_ofdm_blocks=len(data_rows),
            pilot_ofdm_blocks=pilot_blocks,
            duration_seconds=len(signal) / c.sample_rate,
        )
        return signal, info

    def transmit_file(self, payload_path: str | Path, wav_path: str | Path = "tx.wav") -> TransmissionInfo:
        payload_path = Path(payload_path)
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


if __name__ == "__main__":
    if not PAYLOAD_FILE.is_file():
        raise FileNotFoundError(f"Payload file not found: {PAYLOAD_FILE}")
    Transmitter().transmit_file(PAYLOAD_FILE, ask_output_wav())
