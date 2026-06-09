import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt

from Audio_Modem.Utilities import bytes_to_bits
from plugtest_modem.modem import AudioModem

import concurrent.futures


def get_audio(input_filepath):
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Input file not found: {input_filepath}")

    sample_rate, tx_signal = wav.read(input_filepath)

    if tx_signal.ndim > 1:
        tx_signal = np.mean(tx_signal, axis=1)

    if tx_signal.dtype != np.float32 and tx_signal.dtype != np.float64:
        tx_signal = tx_signal.astype(np.float32) / np.max(np.abs(tx_signal))

    return sample_rate, tx_signal


def get_echo_impulse_response(sample_rate):
    # Define physical room parameters
    direct_length = 5.0  # Direct distance from speaker to mic (m)
    echo_length = 6  # Indirect distance from speaker to mic (m)
    speed_of_sound = 343.0  # Speed of sound in air (m/s)

    # 1. Direct Path
    # Helps test synchronisation
    t_direct = direct_length / speed_of_sound
    direct_idx = int(t_direct * sample_rate)

    # 2. Echo Path
    t_echo = echo_length / speed_of_sound
    echo_idx = int(t_echo * sample_rate)

    # 3. Define the Channel Impulse Response (h)
    h_length = max(2000, echo_idx + 500)
    h = np.zeros(h_length)

    # Direct line-of-sight path (shifted from index 0 by the mic distance)
    h[direct_idx] = 0.7

    # First major reflection
    h[echo_idx] = -0.2

    # Add secondary reverberation tails relative to the main echo
    if echo_idx + 150 < h_length:
        h[echo_idx + 150] = 0.1
    if echo_idx + 300 < h_length:
        h[echo_idx + 300] = -0.01

    return h, t_direct, t_echo


def apply_echo(tx_signal, h_echo):
    # Convolve the transmitted audio with the room's impulse response
    return np.convolve(tx_signal, h_echo, mode="full")


def generate_gaussian_noise(rx_signal, snr_db):
    signal_power = np.mean(rx_signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return np.sqrt(noise_power) * np.random.randn(len(rx_signal))


def compute_dfft(time_signal, sample_rate):
    N = len(time_signal)

    # norm='ortho' scales the transform so energy is conserved between time and frequency domains
    fft_values = np.fft.fft(time_signal, norm="ortho")

    frequencies = np.fft.fftfreq(N, d=1.0 / sample_rate)

    return frequencies, fft_values


def simulate_acoustic_channel(
    input_filepath,
    output_filepath=None,
    snr_db=20,
    custom_h=None,
    pad_start_sec=1.5,
    pad_end_sec=1.5,
):
    sample_rate, tx_signal = get_audio(input_filepath)

    # echo impulse response
    if custom_h is None:
        h, t_direct, t_echo = get_echo_impulse_response(sample_rate)
    else:
        h = custom_h
        t_direct, t_echo = 0, 0

    rx_signal = apply_echo(tx_signal, h)

    signal_power = np.mean(rx_signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    pad_start_samples = int(pad_start_sec * sample_rate)
    pad_end_samples = int(pad_end_sec * sample_rate)
    rx_signal = np.concatenate(
        [np.zeros(pad_start_samples), rx_signal, np.zeros(pad_end_samples)]
    )
    rx_signal += noise_std * np.random.randn(len(rx_signal))
    # rx_signal += generate_gaussian_noise(rx_signal, snr_db)

    t_direct += pad_start_sec
    t_echo += pad_start_sec

    # 5. Normalize and Save Output (if requested)
    # Prevent clipping by normalizing the final received signal
    max_val = np.max(np.abs(rx_signal))
    if max_val > 1.0:
        rx_signal = rx_signal / max_val

    if output_filepath:
        # Convert back to 16-bit PCM for standard wav file saving
        rx_signal_int16 = np.int16(rx_signal * 32767)
        wav.write(output_filepath, sample_rate, rx_signal_int16)

    return sample_rate, rx_signal, t_direct, t_echo


def code_test():
    # Example: Create a dummy sine wave audio file to test the simulator
    fs_test = 48000
    t = np.linspace(0, 1, fs_test, endpoint=False)
    test_tx = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    wav.write("simulated_audio/tx_test_signal.wav", fs_test, np.int16(test_tx * 32767))

    # Run the simulator
    fs, rx_simulated, t_direct, t_echo = simulate_acoustic_channel(
        input_filepath="simulated_audio/tx_test_signal.wav",
        output_filepath="simulated_audio/rx_simulated_signal.wav",
        snr_db=8,
    )

    # Assuming you already have 'rx_simulated' and 'fs' from the channel simulator
    freqs, fft_complex = compute_dfft(rx_simulated, fs)

    # Calculate the magnitude (absolute value) of the complex numbers
    fft_magnitude = np.abs(fft_complex)

    # Plotting only the positive frequencies (the second half is just a mirrored negative spectrum)
    positive_freq_indices = freqs >= 0

    plt.figure(figsize=(10, 4))
    plt.plot(
        freqs[positive_freq_indices],
        fft_magnitude[positive_freq_indices],
        color="purple",
    )
    plt.title("Frequency Spectrum of Received Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 24000)
    plt.grid(True)
    plt.show()

    plot_duration = 0.1
    num_samples_to_plot = int(plot_duration * fs)

    time_axis = np.linspace(0, plot_duration, num_samples_to_plot, endpoint=False)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot 1: The clean transmitted signal
    axs[0].plot(
        time_axis,
        test_tx[:num_samples_to_plot],
        color="blue",
        label="Transmitted (Clean 440Hz)",
    )
    axs[0].set_title("Transmitted Signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    axs[0].legend(loc="upper right")

    # Plot 2: The received noisy/echoed signal
    axs[1].plot(
        time_axis,
        rx_simulated[:num_samples_to_plot],
        color="red",
        alpha=0.8,
        label=f"Received (SNR=15dB, Echoed)",
    )
    # Add vertical lines for arrival times
    if t_direct < plot_duration:
        axs[1].axvline(
            x=t_direct,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Direct Path ({t_direct * 1000:.1f} ms)",
        )
    if t_echo < plot_duration:
        axs[1].axvline(
            x=t_echo,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Echo Path ({t_echo * 1000:.1f} ms)",
        )
    axs[1].set_title("Simulated Received Signal")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def gather_data(data, snr_db, modem_instance):
    # simulate channel
    fs, rx_simulated, t_direct, t_echo = simulate_acoustic_channel(
        input_filepath="tx.wav",
        snr_db=snr_db,
    )

    decoded_file_name, decoded_data, h, H = modem_instance.decode_signal(rx_simulated)
    data = bytes_to_bits(data)
    decoded_data = bytes_to_bits(decoded_data)
    min_len = min(len(data), len(decoded_data))
    if min_len == 0:
        return 1

    data_bits = data[:min_len]
    decoded_data_bits = decoded_data[:min_len]
    errors = np.sum(data_bits != decoded_data_bits)
    bit_error_rate = errors / min_len
    return bit_error_rate


# def process_single_snr(snr):
#     """Runs N iterations for a specific SNR and returns the average BER."""
#     ber_accumulated = 0
#     for _ in range(N):
#         ber_accumulated += gather_data(data, snr_db=snr, modem_instance=modem)
#
#     avg_ber = ber_accumulated / N
#     print(f"Completed SNR {snr:02d} dB | Avg BER: {avg_ber * 100:.4f}%")
#     return avg_ber


def process_single_snr(snr):
    """Runs N iterations and returns ALL the raw BER results."""
    raw_bers = []
    N = 5
    for _ in range(N):
        raw_bers.append(gather_data(data, snr_db=snr, modem_instance=modem))
    print(f"Completed SNR {snr:02d} dB")
    return snr, raw_bers


def estimate_frequency_response(h, sample_rate=48000, n_fft=4096):
    """
    Calculates and plots the frequency response (magnitude and phase) of a given impulse response.

    Parameters:
        h (numpy array): The time-domain channel impulse response.
        sample_rate (int): The audio sampling rate in Hz.
        n_fft (int): The size of the FFT. A larger number gives a smoother, higher-resolution plot.

    Returns:
        freqs (numpy array): The frequency bins in Hz.
        H (numpy array): The complex frequency response.
    """
    # 1. Take the FFT to get the complex frequency response
    # Using rfft since our acoustic time-domain signal is strictly real
    H = np.fft.rfft(h, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 2. Calculate Magnitude in Decibels (dB)
    # Adding a tiny epsilon (1e-12) prevents log10(0) math domain errors
    magnitude = np.abs(H)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    # 3. Calculate Phase in Degrees
    phase = np.angle(H, deg=True)

    # ==========================================
    # Plotting
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Magnitude Response
    ax1.plot(freqs, magnitude_db, color="blue", linewidth=1.5)
    ax1.set_title("Channel Frequency Response")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    # Highlight your specific OFDM data band (4 kHz to 13 kHz)
    ax1.axvspan(
        4000, 13000, color="yellow", alpha=0.2, label="JOSS-D OFDM Band (4-13 kHz)"
    )
    ax1.legend(loc="lower left")

    # Plot Phase Response
    ax2.plot(freqs, phase, color="red", linewidth=1.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (Degrees)")
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    # Limit x-axis to the audible spectrum
    ax2.set_xlim(0, 24000)

    plt.tight_layout()
    plt.show()

    return freqs, H


def gather_comparative_data(
    data,
    snr_db,
    modem_instance,
    method,
    expected_bits,
    expected_name,
    expected_size,
    tx_wav_path,
):
    # Pass the unique tx_wav_path for this specific method
    fs, rx_simulated, _, _ = simulate_acoustic_channel(
        input_filepath=tx_wav_path,
        snr_db=snr_db,
    )

    name, decoded_data, h, H = modem_instance.decode_signal(
        rx_simulated, est_method=method
    )

    if decoded_data is None:
        return 0.5, 0  # Total failure

    decoded_bits = bytes_to_bits(decoded_data)
    min_len = min(len(expected_bits), len(decoded_bits))

    if min_len == 0:
        return 0.5, 0

    errors = np.sum(expected_bits[:min_len] != decoded_bits[:min_len])

    header_success = (
        1 if (name == expected_name and len(decoded_data) == expected_size) else 0
    )

    return errors / min_len, header_success


def run_method_comparison(
    data, modem, snr_range, expected_bits, expected_name, expected_size, iterations=50
):
    methods = ["chirp", "golay", "block", "sparse"]
    results_mean = {m: [] for m in methods}
    results_std = {m: [] for m in methods}

    for method in methods:
        print(f"\n--- Testing {method.upper()} Estimation ---")

        # Pre-generate the TX file specific to this method's pilot structure
        tx_wav_path = f"tx_{method}.wav"
        modem.transmit(data, expected_name, wav_path=tx_wav_path, pilot_type=method)

        def process_snr_wrapper(snr):
            raw_results = [
                gather_comparative_data(
                    data,
                    snr,
                    modem,
                    method,
                    expected_bits,
                    expected_name,
                    expected_size,
                    tx_wav_path,
                )
                for _ in range(iterations)
            ]
            bers = [r[0] for r in raw_results]
            return snr, bers

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            raw_results = list(executor.map(process_snr_wrapper, snr_range))

        for snr, raw_bers in raw_results:
            bers = np.array(raw_bers)
            results_mean[method].append(np.mean(bers))
            results_std[method].append(np.std(bers))
            print(f"SNR {snr:02d} dB | Mean BER: {results_mean[method][-1]:.4f}")

    return results_mean, results_std


def plot_estimation_comparison(snr_range, means, stds):
    plt.figure(figsize=(12, 8))

    # Configuration for each method
    plot_styles = {
        "chirp": {"color": "blue", "marker": "o", "label": "Chirp (JOSS-D Standard)"},
        "block": {
            "color": "orange",
            "marker": "^",
            "label": "Dense Block Pilots (High Overhead)",
        },
        "sparse": {
            "color": "red",
            "marker": "x",
            "label": "Sparse Comb Pilots (Interpolated)",
        },
        "golay": {
            "color": "green",
            "marker": "s",
            "label": "Golay Pairs (Time Domain)",
        },
    }

    for method, style in plot_styles.items():
        plt.semilogy(
            snr_range,
            means[method],
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            label=style["label"],
        )
        # Optional: Add shaded variance region (can get messy with 4 lines, uncomment if desired)
        # plt.fill_between(
        #     snr_range,
        #     np.clip(np.array(means[method]) - np.array(stds[method]), 1e-5, 1.0),
        #     np.clip(np.array(means[method]) + np.array(stds[method]), 1e-5, 1.0),
        #     color=style["color"], alpha=0.1,
        # )

    plt.title("OFDM QPSK Channel Estimation Comparison (CP=2048, Echo=1500)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which="both", ls="--", color="gray", alpha=0.6)
    plt.xlim(snr_range[0], snr_range[-1])
    plt.ylim(1e-4, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_name = "sample_text.txt"
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(
                "Cambridge Engineering OFDM Plugfest Interoperability Test File Data Block."
            )

    with open(file_name, "r") as f:
        data = f.read().encode("utf-8")

    modem = AudioModem()
    # The receiver strips the header, so the target must also be the raw data
    expected_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    snr_range = list(range(2, 21, 2))
    iterations_per_snr = 50

    means, stds = run_method_comparison(
        data,
        modem,
        snr_range,
        expected_bits,
        file_name,
        len(data),
        iterations=iterations_per_snr,
    )

    plot_estimation_comparison(snr_range, means, stds)


# if __name__ == "__main__":
#     # 1. Setup Data & TX WAV
#     file_name = "sample_text.txt"
#     with open(file_name, "r") as f:
#         data = f.read().encode("utf-8")
#
#     modem = AudioModem()
#     modem.transmit(data, file_name)
#
#     expected_payload = modem.build_payload(data, file_name)
#     expected_bits = np.unpackbits(np.frombuffer(expected_payload, dtype=np.uint8))
#
#     expected_name = file_name
#     expected_size = len(data)
#
#     # 2. Run the Comparison
#     snr_range = list(range(2, 21, 2))  # Test 0 to 20 dB in steps of 2
#     iterations_per_snr = 100
#
#     means, stds, hsrs = run_method_comparison(
#         data,
#         modem,
#         snr_range,
#         expected_bits,
#         expected_name,
#         expected_size,
#         iterations=iterations_per_snr,
#     )
#
#     # 3. Plot Results
#     plot_estimation_comparison(snr_range, means, stds)

#
# if __name__ == "__main__":
#     # 1. Get the TRUE impulse response of the room (from your simulator)
#     h_true, _, _ = get_echo_impulse_response(48000)
#
#     # 2. Get the TRUNCATED impulse response (what your receiver is forced to use)
#     h_truncated = h_true[:1024]
#
#     # Plot both to see the difference
#     estimate_frequency_response(h_true, n_fft=4096)
#     estimate_frequency_response(h_truncated, n_fft=4096)
#     exit()
# get data
# file_name = "sample_text.txt"
# with open(file_name, "r") as f:
#     data = f.read().encode("utf-8")

# # create tx.wav
# modem = AudioModem()
# tx = modem.transmit(data, file_name)
#
# snr_range = range(0, 16)
# N = 100
# mean_ber = []
# median_ber = []
# std_ber = []
# sync_drop_rate = []
#
# with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#     # Get all raw results
#     results = list(executor.map(process_single_snr, snr_range))
#
# # Process the raw data for statistics
# for snr, raw_bers in results:
#     bers = np.array(raw_bers)
#
#     # 1. Identify and separate sync failures (BER > 40%)
#     valid_runs = bers[bers < 0.40]
#     drops = len(bers) - len(valid_runs)
#     sync_drop_rate.append(drops / len(bers))
#
#     # 2. Calculate stats ONLY on valid runs (preventing math errors if all runs failed)
#     if len(valid_runs) > 0:
#         mean_ber.append(np.mean(valid_runs))
#         median_ber.append(np.median(valid_runs))
#         std_ber.append(np.std(valid_runs))
#     else:
#         mean_ber.append(0.5)
#         median_ber.append(0.5)
#         std_ber.append(0.0)
# # Convert to numpy arrays for easier plotting math
# mean_ber = np.array(mean_ber)
# std_ber = np.array(std_ber)
#
# # --- Plotting the Advanced Graph ---
# plt.figure(figsize=(10, 6))
#
# # Plot the Mean line
# plt.plot(
#     # plt.semilogy(
#     snr_range,
#     mean_ber,
#     marker="o",
#     color="blue",
#     linewidth=2,
#     label="Mean BER (Valid Runs)",
# )
#
# # Plot the Median line
# plt.plot(
#     # plt.semilogy(
#     snr_range,
#     median_ber,
#     marker="s",
#     linestyle="--",
#     color="red",
#     alpha=0.8,
#     label="Median BER",
# )
#
# # Add a shaded confidence region (Mean +/- 1 Standard Deviation)
# # Using np.clip to prevent the bottom of the shaded region from trying to go below 0 on a log scale
# plt.fill_between(
#     snr_range,
#     np.clip(mean_ber - std_ber, 1e-5, 1.0),
#     np.clip(mean_ber + std_ber, 1e-5, 1.0),
#     color="blue",
#     alpha=0.2,
#     label=r"$\pm 1$ Standard Deviation",
# )
#
# plt.title("Bit Error Rate vs. SNR (with Outlier Rejection and Variance)")
# plt.xlabel("SNR (dB)")
# plt.ylabel("Bit Error Rate (BER)")
# plt.grid(True, which="both", ls="--", color="gray", alpha=0.6)
# plt.xlim(snr_range[0], snr_range[-1])
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# # semilogy
# plt.semilogy(
#     snr_range,
#     ber_list,
#     marker="o",
#     linestyle="-",
#     color="b",
#     linewidth=2,
#     markersize=6,
# )
#
# plt.title(
#     "Bit Error Rate (BER) vs. Signal-to-Noise Ratio (SNR) on simulated channel"
# )
# plt.xlabel("SNR (dB)")
# plt.ylabel("Bit Error Rate (BER)")
#
# plt.grid(True, which="both", ls="--", color="gray", alpha=0.6)
#
# plt.xlim(snr_range[0], snr_range[-1])
#
# plt.tight_layout()
# plt.show()
