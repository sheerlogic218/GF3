from typing import Literal

import numpy as np
import numpy.typing as npt

from Audio_Modem.Modulator import Modulator
from Audio_Modem.Utilities import bytes_to_bits, bits_to_bytes, Header

accepted_chirp_types = Literal[
    "linear",
    "quadratic",
    "exponential",
    "logarithmic",
]


class Chirp:
    def __init__(
        self,
        duration: float = 1.0,
        amplitude: float = 0.6,
        fade_duration: float = 0.02,
        repeats: int = 5,
        chirp_type: accepted_chirp_types = "linear",
        f0: int = 100,
        f1: int = 5000,
    ):
        if f0 <= 0 or f1 <= 0:
            raise ValueError("f0,f1 must be > 0")

        self.duration: float = duration
        self.amplitude: float = amplitude
        self.fade_duration: float = fade_duration
        self.repeats: int = repeats
        self.type: accepted_chirp_types = chirp_type
        self.f0: int = f0
        self.f1: int = f1
        self.sampling_freq: int = 48000

    def form_chirp(self) -> npt.NDArray[float]:
        t = np.linspace(
            0,
            self.duration,
            int(self.duration * self.sampling_freq),
            endpoint=False,
        )
        match self.type:
            case "linear":
                k = (self.f1 - self.f0) / self.duration
                return 2 * np.pi * (self.f0 * t + 0.5 * k * t**2)
            case "quadratic":
                k = (self.f1 - self.f0) / self.duration**2
                return 2 * np.pi * (self.f0 * t + (k / 3) * t**3)
            case "exponential" | "logarithmic":
                ratio = self.f1 / self.f0
                return (
                    2
                    * np.pi
                    * self.f0
                    * self.duration
                    / np.log(ratio)
                    * (ratio ** (t / self.duration) - 1)
                )
            case "":
                raise ValueError("Unknown chirp type")

    def make_chirp(self):
        phase = self.form_chirp()

        fade_len = min(int(self.fade_duration * self.sampling_freq), len(phase) // 2)
        fade = np.ones_like(phase)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)

        chirp_signal = self.amplitude * np.sin(phase) * fade
        return np.tile(chirp_signal, self.repeats)


class OFDM:
    def __init__(
        self,
        sampling_freq: int = 48000,
        subcarriers: int = 1024,
        prefix_length: int = 256,
        min_freq: int = 800,
        max_freq: int = 10000,
        pilot_spacing: int = 1,
    ):
        self.sampling_freq: int = sampling_freq
        self.subcarriers: int = subcarriers
        self.prefix_length: int = prefix_length
        self.min_freq: int = min_freq
        self.max_freq: int = max_freq
        self.bins: npt.NDArray[int] = self.get_ofdm_bins()
        self.pilot_bins = self.bins[:: pilot_spacing + 1]
        self.data_bins = np.array(list(set(self.bins) - set(self.pilot_bins)))

    def get_ofdm_bins(self):
        bin_res = self.sampling_freq / self.subcarriers
        start_bin = int(np.ceil(self.min_freq / bin_res))
        end_bin = min(
            int(np.floor(self.max_freq / bin_res)),
            (self.subcarriers // 2) - 1,
        )
        return np.arange(start_bin, end_bin + 1).astype(int)

    def to_OFDM_symbol(self):
        X = np.zeros(self.subcarriers, dtype=complex)


ofdm = OFDM()


header = Header()
modulator = Modulator()

chirp = Chirp(chirp_type="linear")
chirp.make_chirp()

MESSAGE: bytes = b"Hello"
message_bits = bytes_to_bits(MESSAGE)

symbols = modulator.to_qpsk(message_bits)
decoded = bits_to_bytes(modulator.from_qpsk(symbols))


print(decoded)
