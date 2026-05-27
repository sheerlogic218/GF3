import zlib

import numpy as np
import numpy.typing as npt


def bytes_to_bits(data_bytes: bytes) -> npt.NDArray[np.uint8]:
    return np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))


def bits_to_bytes(data_bits: npt.NDArray[np.uint8]) -> bytes:
    return np.packbits(data_bits[: len(data_bits) // 8 * 8]).tobytes()


class Header:
    def __init__(
        self,
        header_repetition: int = 5,
        version: int = 1,
    ):
        self.repetitions = header_repetition
        self.version = version.to_bytes(1)

    def form_header(self, payload_bytes: bytes) -> npt.NDArray[np.uint8]:
        ## Change to spec
        # magic 2 bytes | version 1 byte | payload length 2 bytes | crc32 4 bytes
        magic_bytes = 0xBEEF.to_bytes(2)  # idk why this is here

        length = len(payload_bytes).to_bytes(2)
        crc = zlib.crc32(payload_bytes).to_bytes(4)

        return bytes_to_bits(magic_bytes + self.version + length + crc)

    def make_header(self, payload_bytes: bytes) -> npt.NDArray[np.uint8]:
        """Returns a header with every bit repeated N times."""
        return np.repeat(self.form_header(payload_bytes), self.repetitions)

    def decode_header(self, bits: npt.NDArray[np.uint8]) -> tuple[int, int, int, int]:
        """Decodes header bits into a tuple, undoing bit repetitions."""
        header_bits = bits[:: self.repetitions]
        header_bytes = bits_to_bytes(header_bits)

        magic = int.from_bytes(header_bytes[0:2])
        version = int.from_bytes(header_bytes[2:3])
        length = int.from_bytes(header_bytes[3:5])
        crc = int.from_bytes(header_bytes[5:9])

        return magic, version, length, crc
