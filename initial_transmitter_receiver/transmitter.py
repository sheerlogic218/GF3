import numpy as np
from scipy.io import wavfile


# config

FS = 48000

N = 1024
CP = 128
F_MIN = 500
F_MAX = 8000

PREAMBLE_DURATION = 1.0
PREAMBLE_F0 = 500
PREAMBLE_F1 = 8000

HEADER_BITS = 32
HEADER_REPETITION = 3

SILENCE_DURATION = 0.3

OUTPUT_FILENAME = "basic_ofdm_qpsk_tx.wav"

MESSAGE = b"this is a basic OFDM QPSK transmitter"


# utils

def bytes_to_bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def build_header(payload_num_bits):
    header_bits = np.unpackbits(
        np.array([payload_num_bits], dtype=">u4").view(np.uint8)
    )

    repeated_header = np.repeat(header_bits, HEADER_REPETITION)

    return repeated_header


def build_bitstream(message):
    payload_bits = bytes_to_bits(message)
    header_bits = build_header(len(payload_bits))

    tx_bits = np.concatenate([header_bits, payload_bits])

    if len(tx_bits) % 2 != 0:
        tx_bits = np.append(tx_bits, 0)

    return tx_bits


# qpsk

def qpsk_map(bits):
    symbols = [(-1)**int(b1) + (-1)**int(b0)*1j for b0, b1 in bits.reshape(-1, 2)]
    return np.array(symbols)/np.sqrt(2)


# ofdm

def get_active_bins(fs):
    freqs = np.arange(N) * fs / N

    positive_bins = np.where(
        (freqs >= F_MIN)
        & (freqs <= F_MAX)
        & (np.arange(N) < N // 2)
    )[0]

    return positive_bins, freqs


def pad_symbols_to_grid(qpsk_symbols, num_data_bins):
    num_ofdm_symbols = int(np.ceil(len(qpsk_symbols) / num_data_bins))

    pad_len = num_ofdm_symbols * num_data_bins - len(qpsk_symbols)

    padded_symbols = np.concatenate([
        qpsk_symbols,
        np.zeros(pad_len, dtype=complex),
    ])

    qpsk_grid = padded_symbols.reshape(num_ofdm_symbols, num_data_bins)

    return qpsk_grid


def build_single_ofdm_symbol(qpsk_row, active_bins):
    X = np.zeros(N, dtype=complex)

    X[active_bins] = qpsk_row

    # Hermitian symmetry makes the IFFT output real-valued
    X[-active_bins] = np.conj(qpsk_row)

    x = np.fft.ifft(X).real

    x_cp = np.concatenate([x[-CP:], x])

    return x_cp


def build_ofdm_signal(qpsk_symbols, active_bins):
    num_data_bins = len(active_bins)
    qpsk_grid = pad_symbols_to_grid(qpsk_symbols, num_data_bins)

    time_symbols = []

    for row in qpsk_grid:
        x_cp = build_single_ofdm_symbol(row, active_bins)
        time_symbols.append(x_cp)

    return np.concatenate(time_symbols)


# preamble

def generate_chirp_preamble(fs):
    t = np.linspace(
        0,
        PREAMBLE_DURATION,
        int(fs * PREAMBLE_DURATION),
        endpoint=False,
    )

    k = (PREAMBLE_F1 - PREAMBLE_F0) / PREAMBLE_DURATION
    phase = 2 * np.pi * (PREAMBLE_F0 * t + 0.5 * k * t**2)

    return 0.5 * np.sin(phase)


# audio output

def normalise_audio(audio, peak=0.8):
    max_abs = np.max(np.abs(audio))

    if max_abs == 0:
        return audio

    return peak * audio / max_abs


def save_wav(filename, fs, audio):
    audio = normalise_audio(audio)
    audio_int16 = np.int16(audio * 32767)

    wavfile.write(filename, fs, audio_int16)

    print(f"Saved: {filename}")
    print("Total duration:", len(audio) / fs, "seconds")


# transmitter main

def run_transmitter():
    print("Message:", MESSAGE)

    tx_bits = build_bitstream(MESSAGE)
    qpsk_symbols = qpsk_map(tx_bits)

    active_bins, freqs = get_active_bins(FS)

    print("Using bins:", active_bins[0], "to", active_bins[-1])
    print("Number of data bins:", len(active_bins))
    print(
        "Frequency range:",
        freqs[active_bins[0]],
        "Hz to",
        freqs[active_bins[-1]],
        "Hz",
    )

    ofdm_signal = build_ofdm_signal(qpsk_symbols, active_bins)

    preamble = generate_chirp_preamble(FS)
    silence = np.zeros(int(SILENCE_DURATION * FS))

    tx_audio = np.concatenate([
        silence,
        preamble,
        ofdm_signal,
        silence,
    ])

    save_wav(OUTPUT_FILENAME, FS, tx_audio)


if __name__ == "__main__":
    run_transmitter()