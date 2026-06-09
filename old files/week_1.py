import numpy as np
from scipy.io import wavfile


def process_signal(rx_signal, CIR) -> bytearray:
    # system params
    N = 1024
    L = 32
    symbol_length = N + L

    # cleaning input data
    rx_signal = rx_signal.astype(float).flatten()

    # truncating input to fit to whole symbols
    num_symbols = len(rx_signal)//symbol_length
    rx_signal = rx_signal[:num_symbols*symbol_length]

    # create list of symbols
    symbols = rx_signal.reshape((num_symbols, symbol_length))

    # remove prefix
    symbols = symbols[:, L:]

    # take FFT of symbols
    symbol_response = np.fft.fft(symbols, axis = 1)

    # take FFT of channel
    channel_response = np.fft.fft(CIR, N)

    # convolution in time domain is multiplication in frequency so divide frequency responses
    X = symbol_response/channel_response

    # extract data bins 1-511
    data_bins = X[:,1:512].flatten()

    bits = np.zeros(len(data_bins) * 2, dtype=np.uint8)

    # QPSK
    # 01 | 00
    # 11 | 10
    # Im neg give MSB = 1, Re neg gives LSB 1
    # MSB
    bits[0::2] = (data_bins.imag < 0).astype(np.uint8)
    # LSB
    bits[1::2] = (data_bins.real < 0).astype(np.uint8)

    # convert to 1D list of uint8 bytes
    bytes_data = bytearray(np.packbits(bits))
    return bytes_data


def extract_data(byte_data: bytearray) -> tuple[str, bytearray]:
    first_null = byte_data.find(b'\x00')
    if first_null == -1:
        raise ValueError("First null terminator not found.")

    second_null = byte_data.find(b'\x00', first_null + 1)
    if second_null == -1:
        raise ValueError("Second null terminator not found.")

    file_name = byte_data[:first_null].decode('utf-8')
    file_size = int(byte_data[first_null + 1:second_null].decode('utf-8'))
    file_data = byte_data[second_null + 1:second_null + 1 + file_size]

    return file_name, file_data

if __name__ == "__main__":
    # Channel Impulse Response
    CIR = np.loadtxt('''data/channel (1).csv''')

    # Load received audio signal
    sample_rate, rx_signal = wavfile.read('''data/file01.wav''')
    
    # process signal
    data = process_signal(rx_signal, CIR)
    # extract data
    file_name, file_data = extract_data(data)
    # save data
    with open(f'{file_name}', 'wb') as f:
        f.write(file_data)