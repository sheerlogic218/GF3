from pathlib import Path
import numpy as np

from modem import (
    CONFIG,
    KNOWN_PILOT_FILENAME,
    KNOWN_PILOT_SHA256,
    PacketCodec,
    QPSK,
    StandardInterleaver,
    WiMaxLDPC,
)
from receiver import Receiver


def test_ldpc_reference_fingerprint() -> None:
    ldpc = WiMaxLDPC(CONFIG)
    assert ldpc.compatibility_fingerprint() == ldpc.JOSSY_REFERENCE_CODEWORD_SHA256


def test_interleaver_and_qpsk_round_trip() -> None:
    rng = np.random.default_rng(12345)
    coded = rng.integers(
        0, 2,
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


def test_pilot_fingerprint() -> None:
    receiver = Receiver(CONFIG)
    assert receiver.known_pilot.content_sha256 == KNOWN_PILOT_SHA256
    assert (Path(__file__).resolve().parent / KNOWN_PILOT_FILENAME).is_file()


def main() -> None:
    test_ldpc_reference_fingerprint()
    test_interleaver_and_qpsk_round_trip()
    test_one_bit_header_repair()
    test_text_salvage_after_txt_marker()
    test_pilot_fingerprint()
    print("all interoperability checks passed")


if __name__ == "__main__":
    main()
