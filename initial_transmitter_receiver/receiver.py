import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# params
fs = 48000
record_seconds = 10

N = 1024
CP = 128
f_min = 500
f_max = 8000

recorded_filename = "recorded_rx.wav"

# record from microphone
print("Recording...")
rx = sd.rec(
    int(record_seconds * fs),
    samplerate=fs,
    channels=1,
    dtype="float64"
)
sd.wait()
print("Recording finished.")

rx = rx[:, 0]

# Save recording
wavfile.write(recorded_filename, fs, np.int16(rx / np.max(np.abs(rx)) * 32767))
print("Saved:", recorded_filename)

# load recorded file
fs_read, rx = wavfile.read("basic_ofdm_qpsk_tx.wav")

rx = rx.astype(np.float64)

if rx.ndim > 1:
    rx = rx[:, 0]

rx = rx / np.max(np.abs(rx))

# recreate preamble
duration = 1.0
t = np.linspace(0, duration, int(fs_read * duration), endpoint=False)

f0 = 500
f1 = 8000
k = (f1 - f0) / duration

phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
preamble = 0.5 * np.sin(phase)

# synchronisation using correlation
print("Searching for preamble...")

corr = np.correlate(rx, preamble, mode="valid")
start_idx = np.argmax(np.abs(corr))

payload_start = start_idx + len(preamble)

print("Detected preamble start:", start_idx)
print("Detected payload start:", payload_start)
print("Detected preamble time:", start_idx / fs_read, "seconds")

payload = rx[payload_start:]

# ofdm bins
freqs = np.arange(N) * fs_read / N

positive_bins = np.where(
    (freqs >= f_min) &
    (freqs <= f_max) &
    (np.arange(N) < N // 2)
)[0]

num_data_bins = len(positive_bins)
symbol_len = N + CP

print("Number of active data bins:", num_data_bins)

# ofdm symbols
num_symbols = len(payload) // symbol_len

if num_symbols == 0:
    raise ValueError("No OFDM symbols found after preamble.")

payload = payload[:num_symbols * symbol_len]
symbols_cp = payload.reshape(num_symbols, symbol_len)

symbols = symbols_cp[:, CP:]

Y = np.fft.fft(symbols, axis=1)

# placeholder channel estimation

USE_CHANNEL_ESTIMATION = False

if USE_CHANNEL_ESTIMATION:
    print("Using placeholder channel estimation...")

    # Receiver assumes the first OFDM symbol is known.
    # This must match the transmitter's known training symbols.
    known_training = np.ones(num_data_bins, dtype=complex)

    Y_train = Y[0, positive_bins]

    eps = 1e-8
    H_hat = Y_train / (known_training + eps)

    # Remaining symbols are data
    Y_data = Y[1:, positive_bins]

    # Equalise
    rx_qpsk = Y_data / (H_hat + eps)

    rx_qpsk = rx_qpsk.reshape(-1)

else:
    print("No channel estimation used yet.")
    rx_qpsk = Y[:, positive_bins].reshape(-1)

# qpsk demodulation
def qpsk_demod(symbols):
    bits = []

    for s in symbols:
        real = np.real(s)
        imag = np.imag(s)

        if real >= 0 and imag >= 0:
            bits.extend([0, 0])
        elif real < 0 and imag >= 0:
            bits.extend([0, 1])
        elif real < 0 and imag < 0:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])

    return np.array(bits, dtype=np.uint8)

rx_bits = qpsk_demod(rx_qpsk)

# decode headder
header_rep_len = 32 * 3

if len(rx_bits) < header_rep_len:
    raise ValueError("Not enough bits to decode header.")

header_rep = rx_bits[:header_rep_len]

header_bits = []

for i in range(0, header_rep_len, 3):
    group = header_rep[i:i+3]
    bit = 1 if np.sum(group) >= 2 else 0
    header_bits.append(bit)

header_bits = np.array(header_bits, dtype=np.uint8)

payload_len_bytes = np.packbits(header_bits).tobytes()
payload_len = int.from_bytes(payload_len_bytes, byteorder="big")

print("Decoded payload length in bits:", payload_len)

# extract payload
data_start = header_rep_len
data_bits = rx_bits[data_start:data_start + payload_len]

num_full_bytes = len(data_bits) // 8
data_bits = data_bits[:num_full_bytes * 8]

rx_bytes = np.packbits(data_bits).tobytes()

print("Recovered bytes:")
print(rx_bytes)

try:
    print("Recovered message:")
    print(rx_bytes.decode("utf-8"))
except UnicodeDecodeError:
    print("Could not decode as UTF-8.")