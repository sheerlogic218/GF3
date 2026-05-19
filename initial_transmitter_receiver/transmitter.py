import numpy as np
from scipy.io import wavfile

# Parameters
fs = 48000          # audio sample rate
N = 1024            # FFT size
CP = 128            # cyclic prefix length
f_min = 500         # lowest used frequency
f_max = 8000        # highest used frequency

# message
message = b"this is a basic OFDM QPSK transmitter"

# Convert bytes to bits
bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))

# header
payload_len = len(bits)
header_bits = np.unpackbits(np.array([payload_len], dtype=">u4").view(np.uint8))

# repetition coding for header: repeat each bit 3 times
header_bits_rep = np.repeat(header_bits, 3)

# Full bitstream
tx_bits = np.concatenate([header_bits_rep, bits])

# Pad to even number of bits for QPSK
if len(tx_bits) % 2 != 0:
    tx_bits = np.append(tx_bits, 0)

# QPSK 
# 00 ->  1 + 1j
# 01 -> -1 + 1j
# 11 -> -1 - 1j
# 10 ->  1 - 1j

def qpsk_map(bits):
    bit_pairs = bits.reshape(-1, 2)
    symbols = []

    for b0, b1 in bit_pairs:
        if b0 == 0 and b1 == 0:
            s = 1 + 1j
        elif b0 == 0 and b1 == 1:
            s = -1 + 1j
        elif b0 == 1 and b1 == 1:
            s = -1 - 1j
        else:  # 10
            s = 1 - 1j

        symbols.append(s / np.sqrt(2))  # normalise power

    return np.array(symbols)

qpsk_symbols = qpsk_map(tx_bits)

# OFDM bins
freqs = np.arange(N) * fs / N

positive_bins = np.where((freqs >= f_min) & (freqs <= f_max) & (np.arange(N) < N//2))[0]

num_data_bins = len(positive_bins)

print("Using bins:", positive_bins[0], "to", positive_bins[-1])
print("Number of data bins:", num_data_bins)
print("Frequency range:", freqs[positive_bins[0]], "Hz to", freqs[positive_bins[-1]], "Hz")

# OFDM symbols
num_ofdm_symbols = int(np.ceil(len(qpsk_symbols) / num_data_bins))

# Pad QPSK symbols to fill whole OFDM symbols
pad_len = num_ofdm_symbols * num_data_bins - len(qpsk_symbols)
qpsk_symbols = np.concatenate([qpsk_symbols, np.zeros(pad_len, dtype=complex)])

qpsk_grid = qpsk_symbols.reshape(num_ofdm_symbols, num_data_bins)

time_signal = []

for i in range(num_ofdm_symbols):
    X = np.zeros(N, dtype=complex)

    # Put QPSK symbols into positive frequency bins
    X[positive_bins] = qpsk_grid[i]

    # Hermitian symmetry to make real-valued audio
    X[-positive_bins] = np.conj(qpsk_grid[i])

    # IFFT to get time-domain OFDM symbol
    x = np.fft.ifft(X).real

    # Add cyclic prefix
    x_cp = np.concatenate([x[-CP:], x])

    time_signal.append(x_cp)

ofdm_signal = np.concatenate(time_signal)
print(ofdm_signal)

# chirp
duration = 1.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

f0 = 500
f1 = 8000

# Linear chirp phase
k = (f1 - f0) / duration
phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
preamble = 0.5 * np.sin(phase)

# Add short silence before and after
silence = np.zeros(int(0.3 * fs))

tx_audio = np.concatenate([
    silence,
    preamble,
    ofdm_signal,
    silence
])

# Normalise safely
tx_audio = tx_audio / np.max(np.abs(tx_audio))
tx_audio = 0.8 * tx_audio

# Convert to int16 WAV
tx_int16 = np.int16(tx_audio * 32767)

wavfile.write("initial_transmitter_receiver/basic_ofdm_qpsk_tx.wav", fs, tx_int16)

print("Saved: basic_ofdm_qpsk_tx.wav")
print("Total duration:", len(tx_audio) / fs, "seconds")