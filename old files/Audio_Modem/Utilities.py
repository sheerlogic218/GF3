import struct

import numpy as np
import numpy.typing as npt


def bytes_to_bits(data_bytes: bytes) -> npt.NDArray[np.uint8]:
    return np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))


def bits_to_bytes(data_bits: npt.NDArray[np.uint8]) -> bytes:
    return np.packbits(data_bits[: len(data_bits) // 8 * 8]).tobytes()


class Header:
    @staticmethod
    def form_header(payload_bytes: bytes, filename: str) -> npt.NDArray[np.uint8]:
        """Forms the header according to JOSS-F Clause 8.0."""
        # C
        c_bytes = filename.encode("utf-8")
        # A
        header_length = 2 + 4 + len(c_bytes)
        # B
        data_length = len(payload_bytes)
        # A+B+C
        header_bytes = struct.pack(">HI", header_length, data_length) + c_bytes

        return bytes_to_bits(header_bytes)

    @staticmethod
    def decode_header(bits: npt.NDArray[np.uint8]) -> tuple[int, int, str]:
        """Decodes the header bits, returning (header_length, data_length, filename)."""
        header_bytes = bits_to_bytes(bits)
        if len(header_bytes) < 6:
            raise ValueError("Insufficient bytes to parse header fields A and B.")

        # A, B
        header_length, data_length = struct.unpack(">HI", header_bytes[0:6])

        if len(header_bytes) < header_length:
            raise ValueError(
                f"Expected {header_length} bytes for header, but got {len(header_bytes)}."
            )

        c_bytes = header_bytes[6:header_length]
        filename = c_bytes.decode("utf-8")

        return header_length, data_length, filename
