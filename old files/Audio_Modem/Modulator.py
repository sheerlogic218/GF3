import numpy as np
import numpy.typing as npt


class Modulator:
    def __init__(self):
        pass

    def to_bpsk(self, bits: npt.NDArray[np.uint8]) -> npt.NDArray[int]:
        return 2 * bits.astype("int") - 1

    def from_bpsk(self, symbols: npt.NDArray[int]) -> npt.NDArray[np.uint8]:
        return (symbols.real > 0).astype("int")

    def to_qpsk(self, bits: npt.NDArray[np.uint8]) -> npt.NDArray[complex]:
        return np.array(
            [(-1) ** int(b1) + (-1) ** int(b0) * 1j for b0, b1 in bits.reshape(-1, 2)]
        ) / np.sqrt(2)

    def from_qpsk(self, symbols: npt.NDArray[complex]) -> npt.NDArray[np.uint8]:
        return (
            np.array([(symbol.imag < 0, symbol.real < 0) for symbol in symbols])
            .reshape(-1)
            .astype("int")
        )
