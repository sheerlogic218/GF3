import sys
import os
from pathlib import Path
import numpy as np
import numpy.typing as npt

# Safely resolve the path to the nested C-library wrapper
current_dir = Path(__file__).resolve().parent
ldpc_root_path = current_dir / "LDPC" / "new_ldpc"
nested_ldpc_path = ldpc_root_path / "py"

if str(nested_ldpc_path) not in sys.path:
    sys.path.append(str(nested_ldpc_path))

from ldpc import code


class LdpcWrapper:
    def __init__(
        self,
        z: int = 61,
    ):
        self.z = z
        self.k = 12 * z  # Information bits (732)
        self.n = 24 * z  # Total codeword bits (1464)
        # Initialize the provided C-backed LDPC core

        original_cwd = os.getcwd()
        os.chdir(ldpc_root_path)

        try:
            # Initialize the provided C-backed LDPC core
            self.core = code(
                standard="802.16",
                rate="1/2",
                z=self.z,
                ptype="A",
            )
        finally:
            # Always safely return to the original working directory
            os.chdir(original_cwd)

    def encode_block(
        self,
        information: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        if len(information) != self.k:
            raise ValueError(f"Information block must be exactly {self.k} bits.")
        return self.core.encode(information).astype(np.uint8)

    def encode_blocks(
        self,
        information_bits: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        pad = (-len(information_bits)) % self.k
        if pad != 0:
            information_bits = np.pad(information_bits, (0, pad))

        blocks = information_bits.reshape(-1, self.k)
        encoded = np.stack([self.encode_block(block) for block in blocks])

        pad_blocks = (-len(encoded)) % 35
        if pad_blocks != 0:
            dummy_info = np.zeros(self.k, dtype=np.uint8)
            dummy_encoded = self.encode_block(dummy_info)
            encoded = np.vstack([encoded, np.tile(dummy_encoded, (pad_blocks, 1))])

        return encoded

    def decode_block(
        self,
        llr: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.uint8], int, bool]:
        llr = np.asarray(llr, dtype=np.double)
        app, iterations = self.core.decode(llr, dectype="sumprod2")
        info_bits = (app < 0.0).astype(np.uint8)[: self.k]
        success = iterations < 350
        return info_bits, iterations, success


class StandardInterleaver:
    STRIDE = 15839

    def __init__(self):
        self.cells = 30 * 854  # 25,620 QPSK symbols total
        j = np.arange(self.cells, dtype=np.int64)
        self.permutation = (self.STRIDE * j) % self.cells

    def interleave(
        self,
        coded_blocks: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.complex128]:
        """Maps (35, 1464) LDPC bits into (30, 854) QPSK symbols."""
        if coded_blocks.shape != (35, 1464):
            raise ValueError(
                f"Expected 35 blocks of 1464 bits, got {coded_blocks.shape}"
            )

        # Gray-QPSK map
        b0 = coded_blocks[:, 0::2]
        b1 = coded_blocks[:, 1::2]
        sym = (1 - 2 * b1) + 1j * (1 - 2 * b0)

        # Scatter by stride permutation
        destination = np.empty(self.cells, dtype=np.complex128)
        destination[self.permutation] = sym.reshape(-1)

        return destination.reshape((30, 854))

    def deinterleave_llrs(
        self,
        equalised_symbols: npt.NDArray[np.complex128],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Maps (30, 854) QPSK symbols back into (35, 1464) LLR values."""
        gathered_sym = equalised_symbols.reshape(-1)[self.permutation]
        gathered_weights = weights.reshape(-1)[self.permutation]

        llrs = np.empty(2 * self.cells, dtype=float)
        llrs[0::2] = gathered_weights * gathered_sym.imag  # LLR for b0
        llrs[1::2] = gathered_weights * gathered_sym.real  # LLR for b1

        return llrs.reshape((35, 1464))
