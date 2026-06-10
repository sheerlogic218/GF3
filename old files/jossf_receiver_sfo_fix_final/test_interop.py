from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from modem import (
    CONFIG,
    KNOWN_PILOT_FILENAME,
    PacketCodec,
    QPSK,
    StandardInterleaver,
    WiMaxLDPC,
)
from receiver import Receiver, TimingObservation
from transmitter import Transmitter


BASE_DIR = Path(__file__).resolve().parent


def test_ldpc_reference_fingerprint() -> None:
    ldpc = WiMaxLDPC(CONFIG)
    assert ldpc.compatibility_fingerprint() == ldpc.JOSSY_REFERENCE_CODEWORD_SHA256


def test_interleaver_and_qpsk_round_trip() -> None:
    rng = np.random.default_rng(12345)
    coded = rng.integers(
        0,
        2,
        size=(CONFIG.ldpc_blocks_per_group, CONFIG.ldpc_code_bits),
        dtype=np.uint8,
    )
    interleaver = StandardInterleaver(CONFIG)
    rows = interleaver.interleave(coded)
    llrs = interleaver.deinterleave_llrs(rows, np.ones(rows.shape, dtype=float))
    hard = (llrs < 0.0).astype(np.uint8)
    assert np.array_equal(hard, coded)

    expected = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j])
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    assert np.array_equal(QPSK.map(bits), expected)


def test_one_bit_header_repair() -> None:
    original = PacketCodec.build(b"payload bytes", "example.txt")
    damaged = bytearray(original)
    damaged[3] ^= 0x04
    recovery = Receiver.recover_packet_header(bytes(damaged))
    assert recovery is not None
    assert recovery.packet.filename == "example.txt"
    assert recovery.packet.payload == b"payload bytes"
    assert recovery.flipped_fixed_header_bits


def test_text_salvage_after_txt_marker() -> None:
    stream = (
        b"\xff\xff\xff\xff\xff\xffmessage.txt"
        + b"Hello world!\nThis is readable text.\nLine 3.\n"
        + bytes([0, 1, 2, 3, 255]) * 50
    )
    salvage = Receiver.salvage_text_payload(stream)
    assert salvage is not None
    assert salvage.method.startswith("after recovered .txt")
    assert salvage.text.startswith("Hello world!")
    assert "readable text" in salvage.text
    assert "message.txt" not in salvage.text


def test_residual_sfo_line_fit() -> None:
    if not (BASE_DIR / KNOWN_PILOT_FILENAME).is_file():
        print(f"SFO line-fit test skipped: place {KNOWN_PILOT_FILENAME} beside modem.py")
        return
    receiver = Receiver(CONFIG)
    target_ppm = 175.0
    observations = tuple(
        TimingObservation(
            block_index=block,
            timing_samples=(target_ppm * 1e-6) * block * CONFIG.ofdm_symbol_length,
            phase_slope_rad_per_bin=0.0,
            phase_intercept_rad=0.0,
            fit_error_rad=0.02,
            weight=50.0,
            is_pilot=(block % CONFIG.pilot_interval == 0),
        )
        for block in range(1, 32)
    )
    estimated_ppm, keep = receiver._estimate_residual_sfo(observations)
    assert np.count_nonzero(keep) >= CONFIG.sfo_min_timing_observations
    assert abs(estimated_ppm - target_ppm) < 0.01


def _apply_synthetic_channel_and_sfo(
    signal: np.ndarray,
    *,
    ppm: float,
) -> np.ndarray:
    impulse = np.zeros(500, dtype=float)
    impulse[0] = 1.0
    impulse[37] = 0.28
    impulse[143] = -0.14
    impulse[401] = 0.08
    channelled = fftconvolve(signal, impulse, mode="full")

    ratio = 1.0 + ppm * 1e-6
    output_axis = np.arange(int((len(channelled) - 1) * ratio), dtype=float)
    source_axis = output_axis / ratio
    warped = CubicSpline(
        np.arange(len(channelled), dtype=float),
        channelled,
        extrapolate=False,
    )(source_axis)
    warped = np.nan_to_num(warped)
    rng = np.random.default_rng(2026)
    warped += rng.normal(0.0, 2e-4, len(warped))
    maximum = float(np.max(np.abs(warped)))
    return warped if maximum == 0.0 else 0.70 * warped / maximum


def test_receiver_sfo_and_channel_regression() -> None:
    pilot_path = BASE_DIR / KNOWN_PILOT_FILENAME
    if not pilot_path.is_file():
        print(f"SFO regression skipped: place {KNOWN_PILOT_FILENAME} beside modem.py")
        return

    payload = b"receiver-only SFO and channel regression\n" * 3
    transmitted, _ = Transmitter(CONFIG).create_signal(payload, "regression.txt")
    received = _apply_synthetic_channel_and_sfo(transmitted, ppm=180.0)
    raw = Receiver(CONFIG).decode_raw_signal(received)
    packet = PacketCodec.parse(raw.decoded_bytes)

    assert packet.filename == "regression.txt"
    assert packet.payload == payload
    assert raw.info.successful_ldpc_blocks == raw.info.total_ldpc_blocks == 35
    assert abs(raw.info.sample_rate_offset_ppm - 180.0) < 5.0


def main() -> None:
    test_ldpc_reference_fingerprint()
    test_interleaver_and_qpsk_round_trip()
    test_one_bit_header_repair()
    test_text_salvage_after_txt_marker()

    pilot_path = BASE_DIR / KNOWN_PILOT_FILENAME
    if pilot_path.is_file():
        receiver = Receiver(CONFIG)
        print(f"pilot fingerprint: {receiver.known_pilot.content_sha256}")
        test_residual_sfo_line_fit()
        test_receiver_sfo_and_channel_regression()
    else:
        print(f"pilot check skipped: place {KNOWN_PILOT_FILENAME} beside modem.py")

    print("all interoperability and receiver DSP checks passed")


if __name__ == "__main__":
    main()
