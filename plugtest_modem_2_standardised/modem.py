"""Shared JOSS-F modem components.

This file contains the standard parameters, QPSK/OFDM blocks, pilot generation,
packet framing, WiMAX LDPC coding, interleaving and WAV helpers.  Normal users
only need to run transmitter.py and receiver.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

# -----------------------------------------------------------------------------
# config.py
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModemConfig:
    """All JOSS-F parameters and a few receiver-only tuning constants."""

    sample_rate: int = 48_000

    # Initial chirp train
    chirp_length: int = 4_096
    chirp_start_hz: float = 750.0
    chirp_stop_hz: float = 18_000.0
    chirp_count: int = 10
    chirp_amplitude: float = 0.20

    # Official Group-1 Golay pilot generator referenced by JOSS-F.
    golay_order: int = 12
    golay_gap: int = 2_048
    golay_repeats: int = 4
    golay_seed: tuple[int, int] = (1, 1)
    golay_amplitude: float = 0.20

    # Golay pilot and OFDM
    fft_length: int = 4_096
    cyclic_prefix: int = 2_048
    data_low_hz: float = 2_000.0
    data_high_hz: float = 12_000.0

    # The written clause says frequencies are strictly inside 2--12 kHz,
    # which gives 853 bins. Appendix B requires 854 bins. The cohort's
    # interleaver therefore forces bins 171..1024 inclusive.
    first_data_bin: int = 171
    last_data_bin: int = 1_024

    pilot_interval: int = 20

    # WiMAX LDPC: IEEE 802.16, rate 1/2, Z=61
    ldpc_z: int = 61
    ldpc_info_bits: int = 732
    ldpc_code_bits: int = 1_464
    ldpc_blocks_per_group: int = 35
    data_symbols_per_group: int = 30
    carriers_per_symbol: int = 854

    # Practical waveform settings. These are not standardised.
    leading_silence: int = 4_800
    trailing_silence: int = 4_800
    ofdm_scale: float = 12.0
    output_peak: float = 0.80
    padding_seed: int = 0x4A4F5353

    # Receiver-side tuning only.
    sync_refine_radius: int = 512
    data_start_refine_radius: int = 384
    golay_regularisation: float = 1e-4
    equaliser_regularisation: float = 2e-3
    pilot_update_weight: float = 0.75
    ldpc_max_iterations: int = 60
    ldpc_min_sum_scale: float = 0.80

    @property
    def data_bins(self) -> NDArray[np.int64]:
        bins = np.arange(self.first_data_bin, self.last_data_bin + 1, dtype=np.int64)
        if len(bins) != self.carriers_per_symbol:
            raise RuntimeError("JOSS-F interleaver requires exactly 854 data carriers")
        return bins

    @property
    def strict_written_standard_bins(self) -> NDArray[np.int64]:
        """Bins obtained by interpreting both frequency limits strictly."""
        df = self.sample_rate / self.fft_length
        lo = int(np.floor(self.data_low_hz / df)) + 1
        hi = int(np.ceil(self.data_high_hz / df)) - 1
        return np.arange(lo, hi + 1, dtype=np.int64)

    @property
    def ofdm_symbol_length(self) -> int:
        return self.fft_length + self.cyclic_prefix

    @property
    def chirp_train_length(self) -> int:
        return self.chirp_count * self.chirp_length

    @property
    def golay_block_length(self) -> int:
        # One official block is [A, gap, B, gap].
        return 2 * self.fft_length + 2 * self.golay_gap

    @property
    def golay_section_length(self) -> int:
        # Official golay_pilot.py: [prefix gap] + 4*[A, gap, B, gap].
        return self.golay_gap + self.golay_repeats * self.golay_block_length

    @property
    def preamble_length(self) -> int:
        # 10*4096 chirps + 51200-sample official Golay section = 92160.
        return self.chirp_train_length + self.golay_section_length


CONFIG = ModemConfig()

# -----------------------------------------------------------------------------
# utilities.py
# -----------------------------------------------------------------------------
EPS = 1e-12


def bytes_to_bits(data: bytes) -> NDArray[np.uint8]:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: NDArray[np.uint8]) -> bytes:
    usable = len(bits) - (len(bits) % 8)
    return np.packbits(bits[:usable]).tobytes()


def pad_bits(bits: NDArray[np.uint8], multiple: int) -> NDArray[np.uint8]:
    pad = (-len(bits)) % multiple
    if pad == 0:
        return bits.astype(np.uint8, copy=False)
    return np.pad(bits.astype(np.uint8, copy=False), (0, pad))


def normalise_peak(signal: NDArray[np.float64], peak: float) -> NDArray[np.float64]:
    maximum = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if maximum <= peak or maximum < EPS:
        return signal
    return signal * (peak / maximum)


def safe_output_name(filename: str) -> str:
    name = Path(filename).name or "payload.bin"
    return f"received_{name}"

# -----------------------------------------------------------------------------
# modulation.py
# -----------------------------------------------------------------------------
class QPSK:
    """JOSS-F Gray QPSK mapping, with no 1/sqrt(2) normalisation.

    00 -> +1 + j
    01 -> -1 + j
    10 -> +1 - j
    11 -> -1 - j
    """

    @staticmethod
    def map(bits: NDArray[np.uint8]) -> NDArray[np.complex128]:
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) % 2:
            bits = np.pad(bits, (0, 1))
        pairs = bits.reshape(-1, 2)
        real = 1.0 - 2.0 * pairs[:, 1]
        imag = 1.0 - 2.0 * pairs[:, 0]
        return real + 1j * imag

    @staticmethod
    def hard_symbols(symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        real = np.where(np.real(symbols) >= 0.0, 1.0, -1.0)
        imag = np.where(np.imag(symbols) >= 0.0, 1.0, -1.0)
        return real + 1j * imag

    @staticmethod
    def demap(symbols: NDArray[np.complex128]) -> NDArray[np.uint8]:
        symbols = np.asarray(symbols)
        bits = np.empty(2 * symbols.size, dtype=np.uint8)
        bits[0::2] = (symbols.imag < 0).astype(np.uint8)
        bits[1::2] = (symbols.real < 0).astype(np.uint8)
        return bits

    @staticmethod
    def llr_components(
        symbols: NDArray[np.complex128],
        weights: NDArray[np.float64] | float = 1.0,
    ) -> NDArray[np.float64]:
        """Return [LLR(b0), LLR(b1)] pairs; positive favours bit zero."""
        symbols = np.asarray(symbols)
        weights = np.asarray(weights, dtype=float)
        out = np.empty(2 * symbols.size, dtype=float)
        out[0::2] = np.broadcast_to(weights, symbols.shape).reshape(-1) * symbols.imag.reshape(-1)
        out[1::2] = np.broadcast_to(weights, symbols.shape).reshape(-1) * symbols.real.reshape(-1)
        return out

# -----------------------------------------------------------------------------
# framing.py
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DecodedPacket:
    filename: str
    file_size: int
    payload: bytes
    header_length: int


class PacketCodec:
    """JOSS-F fixed header: [2-byte A][4-byte B][UTF-8 filename C]."""

    MIN_HEADER_BYTES = 6
    MAX_HEADER_BYTES = 65_535

    @staticmethod
    def build(file_bytes: bytes, filename: str) -> bytes:
        name = filename.encode("utf-8")
        header_length = 2 + 4 + len(name)
        if header_length > PacketCodec.MAX_HEADER_BYTES:
            raise ValueError("UTF-8 filename is too long for the 2-byte header length")
        header = struct.pack(">H", header_length) + struct.pack(">I", len(file_bytes)) + name
        return header + file_bytes

    @staticmethod
    def expected_total_bytes(data: bytes) -> int | None:
        if len(data) < PacketCodec.MIN_HEADER_BYTES:
            return None
        header_length = struct.unpack(">H", data[:2])[0]
        file_size = struct.unpack(">I", data[2:6])[0]
        if not (PacketCodec.MIN_HEADER_BYTES <= header_length <= PacketCodec.MAX_HEADER_BYTES):
            raise ValueError(f"invalid decoded header length: {header_length}")
        return header_length + file_size

    @staticmethod
    def parse(data: bytes) -> DecodedPacket:
        total = PacketCodec.expected_total_bytes(data)
        if total is None:
            raise ValueError("not enough decoded bytes for the JOSS-F header")
        if len(data) < total:
            raise ValueError(f"decoded {len(data)} bytes, but the header requests {total}")
        header_length = struct.unpack(">H", data[:2])[0]
        file_size = struct.unpack(">I", data[2:6])[0]
        filename = data[6:header_length].decode("utf-8", errors="strict")
        payload = data[header_length:total]
        return DecodedPacket(filename, file_size, payload, header_length)

# -----------------------------------------------------------------------------
# pilot_data.py
# -----------------------------------------------------------------------------
"""Appendix-A known pilot data from JOSS-F, packed MSB-first."""

PILOT_BITS_HEX = "05fece0d5f219b5937c6513689da58b463ee58afba184f9788f4ec03d78a05a04c6fd81b93f9330dc5b876cd5ca87165e20d3cbb3e1adbcbf9e332c758b940a4f5ec4de36e0b1cb51796d2bdd7bb1ab94ac248e26fd31b6ef42828b4f68601023f136de47a076bf51f64e0d082f2a37673746dd117141dd324f42ca7837eae219567ac1cb7960ce4a68e7bffae3d759f38b7141bd43500d7deb1b377cc57f17b40995bc347bec748dadf7c48ce90be1ba62457754bca274580f9ba145ac2c2467182e5ea095a48a333543726fd432dbdd75f7976452bcfddecca251f368ba8ac22023ecb91e200d654f6cd2f769ec0996b082a030f07b0c293fbdf5caf426a7038f8677f1e3719178702dcbee1e01079d6078620c41720fd4498b15b4097b3245ceb803d894be963f438b63e967fdebdc1de4e0dcba8aa47c4e60d8cbdcef05c756984e25eba23cc7fc364e5df72dda9ea7751a5317bf99ab172ad82b56e3506742c90ca53f0c66d2a0612583f9ce9355f734292d0bd9b727185b13973555a02bec0685beef8227379bde859b17ed44af1b6f129f0a33f884f3e625142ea0c46b0886069ec41d34584c37040fcba4b4b061dfe7863d6a979119bb7d8bd8746d93883007d4ac10e7d1b4bad7db415706de6d57274e1491b61542d4a7507caaa35fc0a3e27c66f39790603ef0972d01a95f830365e4f771ec9bfb9e946ca0adab4"

# -----------------------------------------------------------------------------
# pilots.py
# -----------------------------------------------------------------------------
class LinearChirp:
    """Gold-reference JOSS-F linear chirp used by the interoperable groups.

    The final sample reaches 18 kHz and the chirp starts at cosine phase.
    """

    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config

    def samples(self) -> NDArray[np.float64]:
        c = self.config
        t = np.arange(c.chirp_length, dtype=float) / c.sample_rate
        duration = (c.chirp_length - 1) / c.sample_rate
        rate = (c.chirp_stop_hz - c.chirp_start_hz) / duration
        phase = 2.0 * np.pi * (c.chirp_start_hz * t + 0.5 * rate * t * t)
        return c.chirp_amplitude * np.cos(phase)

    def train(self) -> NDArray[np.float64]:
        return np.tile(self.samples(), self.config.chirp_count)


class GolayPilot:
    """Official Group-1 JOSS-F Golay pilot, reproduced exactly.

    Layout after the ten chirps:
        [2048 zeros] + 4 * [A, 2048 zeros, B, 2048 zeros]

    This contains eight Golay pulses and is exactly 51,200 samples long.
    """

    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        expected_length = 1 << config.golay_order
        if config.fft_length != expected_length:
            raise ValueError(
                f"Golay order {config.golay_order} requires length {expected_length}, "
                f"not {config.fft_length}"
            )
        self.a, self.b = self._generate(config.golay_order, config.golay_seed)

    @staticmethod
    def _generate(
        order: int,
        seed: tuple[int, int] = (1, 1),
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        a = np.array([seed[0]], dtype=float)
        b = np.array([seed[1]], dtype=float)
        for _ in range(order):
            a, b = np.concatenate([a, b]), np.concatenate([a, -b])
        return a, b

    @property
    def pulse_starts(self) -> list[tuple[int, int]]:
        """Return (A start, B start) pairs relative to the Golay section."""
        c = self.config
        starts: list[tuple[int, int]] = []
        for repeat in range(c.golay_repeats):
            a_start = c.golay_gap + repeat * c.golay_block_length
            b_start = a_start + c.fft_length + c.golay_gap
            starts.append((a_start, b_start))
        return starts

    def waveform(self) -> NDArray[np.float64]:
        c = self.config
        gap = np.zeros(c.golay_gap, dtype=float)
        block = np.concatenate([self.a, gap, self.b, gap])
        signal = np.concatenate([gap, np.tile(block, c.golay_repeats)])
        if len(signal) != c.golay_section_length:
            raise RuntimeError("official Golay pilot must contain 51,200 samples")
        return c.golay_amplitude * signal


class KnownOFDMPilot:
    """JOSS-F Appendix-A pilot OFDM block.

    Appendix A contains 4096 bits, which are Gray-QPSK mapped to 2048 symbols
    and placed on rFFT bins 1..2048. The Nyquist bin cannot carry an imaginary
    component in real audio; numpy's irfft therefore transmits its real part.
    All data-band bins 171..1024 remain unambiguous.
    """

    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        packed = bytes.fromhex(PILOT_BITS_HEX)
        self.bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8)).astype(np.uint8)
        if len(self.bits) != 4096:
            raise RuntimeError("Appendix-A pilot must contain 4096 bits")
        n_bins = len(self.config.data_bins)        # 854
        self.symbols = QPSK.map(self.bits[:n_bins * 2])  # first 1708 bits only
        self.frequency = self._frequency_vector()
        self.time = np.fft.irfft(self.frequency, n=config.fft_length) * config.ofdm_scale
        # Use the actually realisable spectrum, including the real Nyquist bin.
        self.transmitted_frequency = np.fft.rfft(self.time, n=config.fft_length)

    def _frequency_vector(self) -> NDArray[np.complex128]:
        spectrum = np.zeros(self.config.fft_length // 2 + 1, dtype=np.complex128)
        spectrum[self.config.data_bins] = self.symbols   # data bins only, all else zero
        return spectrum

    def waveform(self) -> NDArray[np.float64]:
        cp = self.time[-self.config.cyclic_prefix :]
        return np.concatenate([cp, self.time])


class Preamble:
    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        self.chirp = LinearChirp(config)
        self.golay = GolayPilot(config)

    def waveform(self) -> NDArray[np.float64]:
        out = np.concatenate([self.chirp.train(), self.golay.waveform()])
        if len(out) != self.config.preamble_length:
            raise RuntimeError("JOSS-F chirp-plus-Golay preamble must contain 92,160 samples")
        return out

# -----------------------------------------------------------------------------
# ofdm.py
# -----------------------------------------------------------------------------
class OFDM:
    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        self.data_bins = config.data_bins

    def modulate_data_row(self, symbols: NDArray[np.complex128]) -> NDArray[np.float64]:
        symbols = np.asarray(symbols, dtype=np.complex128)
        if symbols.shape != (len(self.data_bins),):
            raise ValueError(f"expected {len(self.data_bins)} QPSK symbols")
        spectrum = np.zeros(self.config.fft_length // 2 + 1, dtype=np.complex128)
        spectrum[self.data_bins] = symbols
        useful = np.fft.irfft(spectrum, n=self.config.fft_length) * self.config.ofdm_scale
        return self.add_prefix(useful)

    def add_prefix(self, useful: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(useful) != self.config.fft_length:
            raise ValueError("OFDM useful section has the wrong length")
        return np.concatenate([useful[-self.config.cyclic_prefix :], useful])

    def remove_prefix(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(block) != self.config.ofdm_symbol_length:
            raise ValueError("OFDM block has the wrong length")
        return block[self.config.cyclic_prefix :]

    def spectrum(self, block: NDArray[np.float64]) -> NDArray[np.complex128]:
        useful = self.remove_prefix(block)
        return np.fft.rfft(useful, n=self.config.fft_length)


class PilotScheduler:
    """Marks absolute OFDM positions 20, 40, 60, ... as pilot blocks."""

    def __init__(self, interval: int = CONFIG.pilot_interval):
        if interval < 2:
            raise ValueError("pilot interval must be at least two")
        self.interval = interval

    def is_pilot(self, one_based_block_index: int) -> bool:
        return one_based_block_index % self.interval == 0

    def total_blocks_for_data_rows(self, data_rows: int) -> int:
        position = 1
        used = 0
        while used < data_rows:
            if not self.is_pilot(position):
                used += 1
            position += 1
        return position - 1

# -----------------------------------------------------------------------------
# coding.py
# -----------------------------------------------------------------------------
# IEEE 802.16 rate-1/2 base matrix, defined at the largest lifting Z0=96.
_WIMAX_RATE_HALF_BASE = np.array(
    [
        [-1,94,73,-1,-1,-1,-1,-1,55,83,-1,-1,7,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,27,-1,-1,-1,22,79,9,-1,-1,-1,12,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,24,22,81,-1,33,-1,-1,-1,0,-1,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1],
        [61,-1,47,-1,-1,-1,-1,-1,65,25,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,39,-1,-1,-1,84,-1,-1,41,72,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,46,40,-1,82,-1,-1,-1,79,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1],
        [-1,-1,95,53,-1,-1,-1,-1,-1,14,18,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1],
        [-1,11,73,-1,-1,-1,2,-1,-1,47,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1,-1],
        [12,-1,-1,-1,83,24,-1,43,-1,-1,-1,51,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1],
        [-1,-1,-1,-1,-1,94,-1,59,-1,-1,70,72,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1],
        [-1,-1,7,65,-1,-1,-1,-1,39,49,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0],
        [43,-1,-1,-1,-1,66,-1,41,-1,-1,-1,26,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0],
    ],
    dtype=np.int16,
)


class WiMaxLDPC:
    """Pure-Python WiMAX QC-LDPC encoder and normalised min-sum decoder."""

    _cache: dict[int, tuple[list[NDArray[np.int64]], list[int], list[int]]] = {}

    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        self.z = config.ldpc_z
        self.k = config.ldpc_info_bits
        self.n = config.ldpc_code_bits
        self.m = self.n - self.k
        if (self.k, self.n, self.m) != (12 * self.z, 24 * self.z, 12 * self.z):
            raise ValueError("rate-1/2 WiMAX dimensions must be K=12Z and N=24Z")
        if self.z not in self._cache:
            checks = self._build_checks()
            info_masks = self._build_info_masks(checks)
            inverse_rows = self._invert_parity_matrix(checks)
            self._cache[self.z] = (checks, info_masks, inverse_rows)
        self.checks, self.info_masks, self.inverse_rows = self._cache[self.z]
        self.check_groups = [
            np.stack(self.checks[row * self.z : (row + 1) * self.z])
            for row in range(12)
        ]
        self.edge_variables = np.concatenate([group.reshape(-1) for group in self.check_groups])

    def _scaled_base(self) -> NDArray[np.int16]:
        scaled = np.full_like(_WIMAX_RATE_HALF_BASE, -1)
        mask = _WIMAX_RATE_HALF_BASE >= 0
        scaled[mask] = (_WIMAX_RATE_HALF_BASE[mask] * self.z // 96).astype(np.int16)
        return scaled

    def _build_checks(self) -> list[NDArray[np.int64]]:
        base = self._scaled_base()
        checks: list[NDArray[np.int64]] = []
        for block_row in range(12):
            for local_row in range(self.z):
                variables: list[int] = []
                for block_col in range(24):
                    shift = int(base[block_row, block_col])
                    if shift >= 0:
                        # A right-circulant shift by p places the one at r+p.
                        variables.append(block_col * self.z + (local_row + shift) % self.z)
                checks.append(np.asarray(variables, dtype=np.int64))
        return checks

    def _build_info_masks(self, checks: list[NDArray[np.int64]]) -> list[int]:
        masks: list[int] = []
        for variables in checks:
            mask = 0
            for variable in variables:
                value = int(variable)
                if value < self.k:
                    mask |= 1 << value
            masks.append(mask)
        return masks

    def _invert_parity_matrix(self, checks: list[NDArray[np.int64]]) -> list[int]:
        augmented: list[int] = []
        for row_index, variables in enumerate(checks):
            parity_row = 0
            for variable in variables:
                value = int(variable)
                if value >= self.k:
                    parity_row |= 1 << (value - self.k)
            augmented.append(parity_row | (1 << (self.m + row_index)))

        for column in range(self.m):
            pivot = next(
                (row for row in range(column, self.m) if (augmented[row] >> column) & 1),
                None,
            )
            if pivot is None:
                raise RuntimeError("WiMAX LDPC parity matrix is singular")
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            pivot_row = augmented[column]
            for row in range(self.m):
                if row != column and ((augmented[row] >> column) & 1):
                    augmented[row] ^= pivot_row

        identity_mask = (1 << self.m) - 1
        for row, value in enumerate(augmented):
            if value & identity_mask != 1 << row:
                raise RuntimeError("failed to invert WiMAX LDPC parity matrix")
        return [value >> self.m for value in augmented]

    @staticmethod
    def _bits_to_int(bits: NDArray[np.uint8]) -> int:
        value = 0
        for index, bit in enumerate(np.asarray(bits, dtype=np.uint8)):
            value |= int(bit) << index
        return value

    def encode_block(self, information: NDArray[np.uint8]) -> NDArray[np.uint8]:
        information = np.asarray(information, dtype=np.uint8)
        if information.shape != (self.k,):
            raise ValueError(f"LDPC information block must contain {self.k} bits")
        information_int = self._bits_to_int(information)

        syndrome = 0
        for row, mask in enumerate(self.info_masks):
            syndrome |= ((mask & information_int).bit_count() & 1) << row

        parity_int = 0
        for bit, inverse_row in enumerate(self.inverse_rows):
            parity_int |= ((inverse_row & syndrome).bit_count() & 1) << bit

        parity = np.fromiter(
            ((parity_int >> bit) & 1 for bit in range(self.m)),
            count=self.m,
            dtype=np.uint8,
        )
        codeword = np.concatenate([information, parity])
        if not self.is_codeword(codeword):
            raise RuntimeError("internal LDPC encoder parity check failed")
        return codeword

    def encode_blocks(self, information_bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
        information_bits = pad_bits(np.asarray(information_bits, dtype=np.uint8), self.k)
        blocks = information_bits.reshape(-1, self.k)
        return np.stack([self.encode_block(block) for block in blocks])

    def syndrome(self, codeword: NDArray[np.uint8]) -> NDArray[np.uint8]:
        codeword = np.asarray(codeword, dtype=np.uint8)
        return np.fromiter(
            (int(np.bitwise_xor.reduce(codeword[variables])) for variables in self.checks),
            count=self.m,
            dtype=np.uint8,
        )

    def is_codeword(self, codeword: NDArray[np.uint8]) -> bool:
        return not np.any(self.syndrome(codeword))

    def decode_block(self, llr: NDArray[np.float64]) -> tuple[NDArray[np.uint8], int, bool]:
        """Decode one codeword with vectorised flooding normalised min-sum."""
        channel_llr = np.asarray(llr, dtype=float)
        if channel_llr.shape != (self.n,):
            raise ValueError(f"LDPC LLR block must contain {self.n} values")

        beliefs = channel_llr.copy()
        messages = [np.zeros(group.shape, dtype=float) for group in self.check_groups]
        scale = self.config.ldpc_min_sum_scale

        for iteration in range(1, self.config.ldpc_max_iterations + 1):
            new_messages: list[NDArray[np.float64]] = []
            for variables, old in zip(self.check_groups, messages):
                extrinsic = beliefs[variables] - old
                signs = np.where(extrinsic >= 0.0, 1.0, -1.0)
                magnitudes = np.abs(extrinsic)
                smallest_index = np.argmin(magnitudes, axis=1)
                two_smallest = np.partition(magnitudes, 1, axis=1)[:, :2]
                smallest = two_smallest[:, 0]
                second = two_smallest[:, 1]
                total_sign = np.prod(signs, axis=1)
                new = scale * total_sign[:, None] * signs * smallest[:, None]
                new[np.arange(self.z), smallest_index] = (
                    scale
                    * total_sign
                    * signs[np.arange(self.z), smallest_index]
                    * second
                )
                new_messages.append(new)

            flat_messages = np.concatenate([message.reshape(-1) for message in new_messages])
            beliefs = channel_llr + np.bincount(
                self.edge_variables,
                weights=flat_messages,
                minlength=self.n,
            )
            messages = new_messages

            hard = (beliefs < 0.0).astype(np.uint8)
            if self.is_codeword(hard):
                return hard[: self.k], iteration, True

        hard = (beliefs < 0.0).astype(np.uint8)
        return hard[: self.k], self.config.ldpc_max_iterations, self.is_codeword(hard)



class StandardInterleaver:
    """Appendix-B 35-codeword to 30-OFDM-symbol stride interleaver."""

    STRIDE = 15_839

    def __init__(self, config: ModemConfig = CONFIG):
        self.config = config
        self.cells = config.data_symbols_per_group * config.carriers_per_symbol
        expected = config.ldpc_blocks_per_group * (config.ldpc_code_bits // 2)
        if self.cells != expected:
            raise ValueError("JOSS-F interleaver dimensions are inconsistent")
        j = np.arange(self.cells, dtype=np.int64)
        self.permutation = (self.STRIDE * j) % self.cells
        if len(np.unique(self.permutation)) != self.cells:
            raise RuntimeError("Appendix-B stride is not a permutation")

    def interleave(self, coded_blocks: NDArray[np.uint8]) -> NDArray[np.complex128]:
        expected = (self.config.ldpc_blocks_per_group, self.config.ldpc_code_bits)
        coded_blocks = np.asarray(coded_blocks, dtype=np.uint8)
        if coded_blocks.shape != expected:
            raise ValueError(f"coded block array must have shape {expected}")
        source_symbols = QPSK.map(coded_blocks.reshape(-1)).reshape(-1)
        destination = np.empty(self.cells, dtype=np.complex128)
        destination[self.permutation] = source_symbols
        return destination.reshape(
            self.config.data_symbols_per_group,
            self.config.carriers_per_symbol,
        )

    def deinterleave_llrs(
        self,
        equalised_symbols: NDArray[np.complex128],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        shape = (self.config.data_symbols_per_group, self.config.carriers_per_symbol)
        equalised_symbols = np.asarray(equalised_symbols, dtype=np.complex128)
        weights = np.asarray(weights, dtype=float)
        if equalised_symbols.shape != shape or weights.shape != shape:
            raise ValueError(f"equalised symbols and weights must have shape {shape}")
        gathered = equalised_symbols.reshape(-1)[self.permutation]
        gathered_weights = weights.reshape(-1)[self.permutation]
        llrs = QPSK.llr_components(gathered, gathered_weights)
        return llrs.reshape(self.config.ldpc_blocks_per_group, self.config.ldpc_code_bits)

# -----------------------------------------------------------------------------
# audio.py
# -----------------------------------------------------------------------------
def write_wav(path: str | Path, signal: NDArray[np.float64], config: ModemConfig = CONFIG) -> None:
    signal = np.asarray(signal, dtype=float)
    pcm = np.clip(signal, -1.0, 1.0)
    wavfile.write(str(path), config.sample_rate, np.round(pcm * 32767.0).astype(np.int16))


def read_wav(path: str | Path, config: ModemConfig = CONFIG) -> NDArray[np.float64]:
    sample_rate, signal = wavfile.read(str(path))
    if sample_rate != config.sample_rate:
        raise ValueError(f"expected a {config.sample_rate} Hz WAV, received {sample_rate} Hz")
    original_dtype = signal.dtype
    signal = signal.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        scale = max(abs(np.iinfo(original_dtype).min), np.iinfo(original_dtype).max)
        signal /= float(scale)
    if signal.ndim == 2:
        # Do not average microphone-array channels: phase differences can cause
        # comb cancellation. Select the channel with the largest robust AC level.
        centred = signal - np.median(signal, axis=0, keepdims=True)
        levels = np.percentile(np.abs(centred), 95, axis=0)
        signal = signal[:, int(np.argmax(levels))]
    signal = signal - float(np.median(signal))
    return signal


def record_wav(
    path: str | Path,
    seconds: float,
    config: ModemConfig = CONFIG,
) -> NDArray[np.float64]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("recording requires: pip install sounddevice") from exc
    samples = int(round(seconds * config.sample_rate))
    print(f"Recording {seconds:.1f} s at {config.sample_rate} Hz ...")
    signal = sd.rec(samples, samplerate=config.sample_rate, channels=1, dtype="float64")
    sd.wait()
    mono = signal[:, 0]
    write_wav(path, mono, config)
    return mono
