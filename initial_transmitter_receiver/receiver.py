import numpy as np
import sounddevice as sd
from scipy.io import wavfile


# config

FS = 48000
RECORD_SECONDS = 10

N = 1024
CP = 128
F_MIN = 500
F_MAX = 8000

PREAMBLE_DURATION = 1.0
PREAMBLE_F0 = 500
PREAMBLE_F1 = 8000

HEADER_BITS = 32
HEADER_REPETITION = 3

RECORDED_FILENAME = "recorded_rx.wav"
INPUT_FILENAME = "basic_ofdm_qpsk_tx.wav"

USE_MICROPHONE = False
USE_CHANNEL_ESTIMATION = False


# audio

def normalise_audio(x):
    x = x.astype(np.float64)

    if x.ndim > 1:
        x = x[:, 0]

    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x

    return x / max_abs


def record_microphone():
    print("Recording...")

    audio = sd.rec(
        int(RECORD_SECONDS * FS),
        samplerate=FS,
        channels=1,
        dtype="float64",
    )
    sd.wait()

    print("Recording finished.")
    return audio[:, 0]


def save_wav(filename, fs, audio):
    audio = normalise_audio(audio)
    wavfile.write(filename, fs, np.int16(audio * 32767))
    print(f"Saved: {filename}")


def load_wav(filename):
    fs_read, audio = wavfile.read(filename)
    audio = normalise_audio(audio)
    return fs_read, audio


# preamble, sync

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


def find_payload_start(rx, preamble, fs):
    print("Searching for preamble...")

    corr = np.correlate(rx, preamble, mode="valid")
    preamble_start = np.argmax(np.abs(corr))
    payload_start = preamble_start + len(preamble)

    print("Detected preamble start:", preamble_start)
    print("Detected payload start:", payload_start)
    print("Detected preamble time:", preamble_start / fs, "seconds")

    return payload_start


# ofdm

def get_active_bins(fs):
    freqs = np.arange(N) * fs / N

    positive_bins = np.where(
        (freqs >= F_MIN)
        & (freqs <= F_MAX)
        & (np.arange(N) < N // 2)
    )[0]

    return positive_bins


def extract_ofdm_symbols(payload):
    symbol_len = N + CP
    num_symbols = len(payload) // symbol_len

    if num_symbols == 0:
        raise ValueError("No OFDM symbols found after preamble.")

    payload = payload[:num_symbols * symbol_len]
    symbols_cp = payload.reshape(num_symbols, symbol_len)

    symbols = symbols_cp[:, CP:]

    return symbols


def fft_ofdm_symbols(symbols):
    return np.fft.fft(symbols, axis=1)


# channel estimation

def equalise_or_extract_qpsk(Y, active_bins):
    num_data_bins = len(active_bins)

    if USE_CHANNEL_ESTIMATION:
        print("Using placeholder channel estimation...")

        known_training = np.ones(num_data_bins, dtype=complex)

        Y_train = Y[0, active_bins]

        eps = 1e-8
        H_hat = Y_train / (known_training + eps)

        Y_data = Y[1:, active_bins]
        rx_qpsk = Y_data / (H_hat + eps)

        return rx_qpsk.reshape(-1)

    print("No channel estimation used yet.")
    return Y[:, active_bins].reshape(-1)


#qpsk

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


# decoding

def decode_repeated_header(rx_bits):
    header_rep_len = HEADER_BITS * HEADER_REPETITION

    if len(rx_bits) < header_rep_len:
        raise ValueError("Not enough bits to decode header.")

    header_rep = rx_bits[:header_rep_len]

    header_bits = []

    for i in range(0, header_rep_len, HEADER_REPETITION):
        group = header_rep[i:i + HEADER_REPETITION]
        bit = 1 if np.sum(group) >= (HEADER_REPETITION / 2) else 0
        header_bits.append(bit)

    header_bits = np.array(header_bits, dtype=np.uint8)

    payload_len_bytes = np.packbits(header_bits).tobytes()
    payload_len = int.from_bytes(payload_len_bytes, byteorder="big")

    return payload_len, header_rep_len


def extract_payload_bytes(rx_bits, payload_len, data_start):
    data_bits = rx_bits[data_start:data_start + payload_len]

    num_full_bytes = len(data_bits) // 8
    data_bits = data_bits[:num_full_bytes * 8]

    return np.packbits(data_bits).tobytes()


# main receiver

def run_receiver():
    if USE_MICROPHONE:
        rx = record_microphone()
        save_wav(RECORDED_FILENAME, FS, rx)
        fs_read, rx = load_wav(RECORDED_FILENAME)
    else:
        fs_read, rx = load_wav(INPUT_FILENAME)

    preamble = generate_chirp_preamble(fs_read)
    payload_start = find_payload_start(rx, preamble, fs_read)

    payload = rx[payload_start:]

    active_bins = get_active_bins(fs_read)
    print("Number of active data bins:", len(active_bins))

    symbols = extract_ofdm_symbols(payload)
    Y = fft_ofdm_symbols(symbols)

    rx_qpsk = equalise_or_extract_qpsk(Y, active_bins)
    rx_bits = qpsk_demod(rx_qpsk)

    payload_len, data_start = decode_repeated_header(rx_bits)
    print("Decoded payload length in bits:", payload_len)

    rx_bytes = extract_payload_bytes(rx_bits, payload_len, data_start)

    print("Recovered bytes:")
    print(rx_bytes)

    try:
        print("Recovered message:")
        print(rx_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        print("Could not decode as UTF-8.")


if __name__ == "__main__":
    run_receiver()