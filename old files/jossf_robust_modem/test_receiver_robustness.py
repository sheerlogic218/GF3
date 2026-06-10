"""Deterministic receiver-only regression tests.

Run directly with:
    python test_receiver_robustness.py

These tests do not replace real speaker/microphone recordings, but they guard
against the timing, SFO, multipath and QPSK-ambiguity regressions addressed by
this receiver.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.signal import fftconvolve

from modem import CONFIG
from receiver import Receiver
from transmitter import Transmitter


def _clock_warp(signal: np.ndarray, ppm: float) -> np.ndarray:
    ratio = 1.0 + ppm * 1e-6
    length = int(np.floor((len(signal) - 1) * ratio)) + 1
    source = np.arange(length, dtype=float) / ratio
    return np.interp(source, np.arange(len(signal), dtype=float), signal)


def _recover_payload(signal: np.ndarray, payload: bytes) -> tuple[Receiver, dict]:
    receiver = Receiver()
    diag: dict = {}
    raw = receiver.decode_raw_signal(signal, _diag=diag)
    recovery = receiver.recover_packet_header(raw.decoded_bytes)
    assert recovery is not None
    assert recovery.packet.payload == payload
    assert raw.info.successful_ldpc_blocks == raw.info.total_ldpc_blocks
    return receiver, diag


def test_clean_end_to_end() -> None:
    payload = b"clean robust receiver test\n"
    signal, _ = Transmitter().create_signal(payload, "payload.txt")
    _recover_payload(signal, payload)


def test_missing_initial_chirps_keep_absolute_origin() -> None:
    payload = b"partial chirp train test\n"
    signal, _ = Transmitter().create_signal(payload, "payload.txt")
    for missing in (1, 3, 4):
        damaged = signal.copy()
        start = CONFIG.leading_silence
        damaged[start : start + missing * CONFIG.chirp_length] = 0.0
        _, diag = _recover_payload(damaged, payload)
        assert abs(diag["sync_sample"] - CONFIG.leading_silence) <= 1
        assert int(diag["chirp_indices"][0]) == missing


def test_sfo_estimation() -> None:
    signal, _ = Transmitter().create_signal(b"x", "x.txt")
    receiver = Receiver()
    for true_ppm in (-500.0, -200.0, 200.0, 500.0):
        warped = _clock_warp(signal, true_ppm)
        _, _, estimate = receiver.synchronise_and_correct(warped)
        assert abs(estimate - true_ppm) < 10.0


def test_multipath_sfo_noise_end_to_end() -> None:
    payload = b"multipath, SFO and noise test\n"
    signal, _ = Transmitter().create_signal(payload, "payload.txt")
    impulse = np.zeros(181)
    impulse[0] = 1.0
    impulse[23] = 0.32
    impulse[79] = -0.18
    impulse[151] = 0.10
    channelled = fftconvolve(signal, impulse)[: len(signal)]
    received = _clock_warp(channelled, 350.0)
    rng = np.random.default_rng(7)
    rms = float(np.sqrt(np.mean(received * received)))
    received += rng.normal(0.0, rms * 10.0 ** (-30.0 / 20.0), len(received))
    _, diag = _recover_payload(received, payload)
    assert abs(diag["sfo_ppm"] - 350.0) < 10.0


def test_ldpc_guided_qpsk_rotation_rescue() -> None:
    receiver = Receiver()
    c = receiver.config
    rng = np.random.default_rng(11)
    information = rng.integers(
        0, 2,
        size=(c.ldpc_blocks_per_group, c.ldpc_info_bits),
        dtype=np.uint8,
    )
    coded = np.stack([receiver.ldpc.encode_block(block) for block in information])
    rows = receiver.interleaver.interleave(coded)
    rotated = rows * 1j
    decoded, _, success, metadata = receiver._decode_group_with_rescue(
        rotated, np.ones(rotated.shape, dtype=float)
    )
    assert all(success)
    assert metadata["rotation_quadrants"] == 1
    assert np.array_equal(np.stack(decoded), information)


def main() -> None:
    tests = [
        test_clean_end_to_end,
        test_missing_initial_chirps_keep_absolute_origin,
        test_sfo_estimation,
        test_multipath_sfo_noise_end_to_end,
        test_ldpc_guided_qpsk_rotation_rescue,
    ]
    started = time.perf_counter()
    for test in tests:
        t0 = time.perf_counter()
        test()
        print(f"PASS {test.__name__} ({time.perf_counter() - t0:.2f} s)")
    print(f"all robustness checks passed ({time.perf_counter() - started:.2f} s)")


if __name__ == "__main__":
    main()
