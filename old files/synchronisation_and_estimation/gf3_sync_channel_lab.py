"""
GF3 sync + channel-estimation lab
=================================

Measurement harness for testing synchronisation and channel estimation in the
GF3 audio modem project.

The script can:
  1. Generate known probe WAVs for real speaker-to-microphone tests.
  2. Analyse a recorded WAV and estimate the effective acoustic channel.
  3. Run a fully simulated demo using a synthetic echo/noise channel.
  4. Optionally play and record live audio if `sounddevice` is installed.

This is not a complete file-transmitting modem. It is a diagnostic tool for
testing synchronisation methods and estimating the speaker-room-microphone
channel before integrating these ideas into an OFDM/DMT modem.

Typical workflow
----------------

0) From the repository/folder containing this script, run commands as:

    python gf3_sync_channel_lab.py <command> [options]


1) Run a simulated demo
-----------------------

This creates a fake transmitted signal, passes it through a simulated acoustic
channel with echoes/noise, then generates diagnostic plots.

    python gf3_sync_channel_lab.py demo --out results_demo

Optional step-sync demo:

    python gf3_sync_channel_lab.py demo --sync-kind step --out results_demo_step

Typical outputs:
    results_demo/
        demo_tx.wav
        demo_rx_simulated.wav
        h_true.npy
        metadata.json
        results.json
        00_recording_spectrogram.png
        01_matched_filter_sync.png
        02_repeated_half_metric.png
        03_channel_impulse_response.png
        04_channel_frequency_response_magnitude.png
        05_channel_frequency_response_phase.png
        06_ofdm_constellation_equalisation.png


2) Generate real probe WAVs
---------------------------

This creates the signal to play through a speaker, plus the templates needed by
the analyser.

Chirp synchronisation probe:

    python gf3_sync_channel_lab.py make-probes --out probes_chirp --sync-kind chirp --fs 48000

Step-frequency synchronisation probe:

    python gf3_sync_channel_lab.py make-probes --out probes_step --sync-kind step --fs 48000

Each probe directory contains:
    measurement_tx.wav      Main WAV to play through the transmitter speaker.
    sync_template.wav       Known sync signal used for matched filtering.
    repeated_template.wav   Repeated-half signal used for self-similarity timing.
    training_template.wav   Known broadband signal used for channel estimation.
    metadata.json           Timing/layout information for the generated probe.


3) Record a real audio channel
------------------------------

Play:
    probes_chirp/measurement_tx.wav

Record the received audio at 48 kHz mono WAV, then save it somewhere like:
    recordings/rx_chirp_baseline.wav

The recording may contain silence before and after the transmission. Do not trim
it manually; the script estimates the timing automatically.

Recommended first tests:
    recordings/rx_chirp_baseline.wav
    recordings/rx_chirp_far.wav
    recordings/rx_chirp_noisy.wav
    recordings/rx_step_baseline.wav
    recordings/rx_step_noisy.wav


4) Analyse a real recording
---------------------------

Analyse a chirp-probe recording:

    python gf3_sync_channel_lab.py analyse \
        --probe-dir probes_chirp \
        --rx recordings/rx_chirp_baseline.wav \
        --out results_chirp_baseline

Analyse a step-probe recording:

    python gf3_sync_channel_lab.py analyse \
        --probe-dir probes_step \
        --rx recordings/rx_step_baseline.wav \
        --out results_step_baseline

Analyse a noisier/further-away take:

    python gf3_sync_channel_lab.py analyse \
        --probe-dir probes_chirp \
        --rx recordings/rx_chirp_noisy.wav \
        --out results_chirp_noisy

Optional channel length override:

    python gf3_sync_channel_lab.py analyse \
        --probe-dir probes_chirp \
        --rx recordings/rx_chirp_baseline.wav \
        --out results_chirp_baseline_h4800 \
        --h-len 4800


5) Optional live play-record helper
-----------------------------------

This plays a generated probe and records from the default microphone.

Requires:
    python -m pip install sounddevice

Example:

    python gf3_sync_channel_lab.py live \
        --tx probes_chirp/measurement_tx.wav \
        --rx-out recordings/rx_live_chirp.wav

Then analyse the live recording:

    python gf3_sync_channel_lab.py analyse \
        --probe-dir probes_chirp \
        --rx recordings/rx_live_chirp.wav \
        --out results_live_chirp


What the generated plots mean
-----------------------------

00_recording_spectrogram.png
    Time-frequency view of the recorded signal. Useful for checking clipping,
    noise, missing playback, and whether the expected probe sections appear.

01_matched_filter_sync.png
    Matched-filter timing result using the known sync template. A clean result
    should have one dominant peak.

02_repeated_half_metric.png
    Self-similarity timing metric for the repeated-half section. Useful for
    testing Schmidl-Cox-style synchronisation ideas.

03_channel_impulse_response.png
    Estimated channel in the time domain. Early peaks correspond to direct-path
    sound; later peaks/tails correspond to echoes and reverberation.

04_channel_frequency_response_magnitude.png
    Estimated channel magnitude response. Shows which frequencies/subcarriers
    are strongly passed or attenuated.

05_channel_frequency_response_phase.png
    Estimated channel phase response. Relevant for OFDM/DMT equalisation.

06_ofdm_constellation_equalisation.png
    Demo-only plot showing how one-tap equalisation improves a simulated
    OFDM/DMT constellation when the true simulated channel is available.


Conceptual model
----------------

The channel-estimation part assumes:

    y[n] ≈ x[n] * h[n] + noise

where:
    x[n] = known transmitted training signal
    y[n] = aligned received training segment
    h[n] = effective speaker-room-microphone impulse response

In the frequency domain:

    Y(f) ≈ H(f) X(f)

The script estimates the channel using a regularised frequency-domain
least-squares form:

    H_hat(f) = Y(f) X*(f) / (|X(f)|^2 + regularisation)

Then:

    h_hat[n] = IFFT(H_hat(f))

So:
    H_hat(f) is the estimated frequency response.
    h_hat[n] is the estimated impulse response.


Repository notes
----------------

Suggested directories:
    probes_chirp/
    probes_step/
    recordings/
    results_chirp_baseline/
    results_chirp_noisy/
    results_step_baseline/
    results_step_noisy/

For a clean repository, consider committing:
    - this script
    - README / usage notes
    - selected result plots
    - small metadata/results JSON files

Avoid committing many large WAV recordings unless needed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


Array = NDArray[np.float64]
EPS = 1e-12


# -----------------------------------------------------------------------------
# Basic audio utilities
# -----------------------------------------------------------------------------

def to_float_audio(x: np.ndarray) -> Array:
    """Convert WAV data to mono float64 in approximately [-1, 1]."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        max_abs = np.iinfo(x.dtype).max
        x = x.astype(np.float64) / max_abs
    else:
        x = x.astype(np.float64)
    x = x - np.mean(x)
    return x


def normalise_peak(x: Array, peak: float = 0.8) -> Array:
    """Remove DC and scale to a chosen peak amplitude."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    m = np.max(np.abs(x)) + EPS
    return peak * x / m


def write_wav(path: Path | str, fs: int, x: Array, peak: float = 0.8) -> None:
    """Write mono 16-bit WAV with conservative peak normalisation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = normalise_peak(x, peak=peak)
    wavfile.write(path, fs, np.int16(np.clip(y, -1, 1) * 32767))


def read_wav(path: Path | str) -> Tuple[int, Array]:
    fs, x = wavfile.read(path)
    return fs, to_float_audio(x)


def seconds_to_samples(fs: int, seconds: float) -> int:
    return int(round(fs * seconds))


# -----------------------------------------------------------------------------
# Probe generation
# -----------------------------------------------------------------------------

def make_chirp_template(
    fs: int = 48_000,
    duration: float = 0.45,
    f0: float = 500.0,
    f1: float = 8_000.0,
    method: str = "logarithmic",
) -> Array:
    """Linear/log chirp used as a matched-filter synchronisation preamble."""
    n = seconds_to_samples(fs, duration)
    t = np.arange(n) / fs
    x = signal.chirp(t, f0=f0, t1=t[-1], f1=f1, method=method)
    x *= signal.windows.tukey(n, alpha=0.08)
    return normalise_peak(x, peak=1.0)


def make_step_template(
    fs: int = 48_000,
    tone_duration: float = 0.055,
    freqs: Iterable[float] = (700, 4_500, 950, 6_500, 1_200, 3_300, 1_600, 7_200),
) -> Array:
    """
    Stepped-frequency template.

    This is the concrete version of the 'jumping low-high frequency' idea:
    a deterministic sequence of short tones, each smoothly windowed. The matched
    filter sees the whole fingerprint, not just one sine frequency.
    """
    pieces = []
    for f in freqs:
        n = seconds_to_samples(fs, tone_duration)
        t = np.arange(n) / fs
        tone = np.sin(2 * np.pi * f * t)
        tone *= signal.windows.tukey(n, alpha=0.25)
        pieces.append(tone)
    x = np.concatenate(pieces)
    return normalise_peak(x, peak=1.0)


def make_bandlimited_noise(
    fs: int = 48_000,
    duration: float = 1.25,
    f_min: float = 500.0,
    f_max: float = 8_000.0,
    seed: int = 9,
) -> Array:
    """Deterministic band-limited white-noise probe for channel estimation."""
    rng = np.random.default_rng(seed)
    n = seconds_to_samples(fs, duration)
    z = rng.standard_normal(n)
    Z = np.fft.rfft(z)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mask = (freqs >= f_min) & (freqs <= f_max)
    Z *= mask
    x = np.fft.irfft(Z, n=n)
    x *= signal.windows.tukey(n, alpha=0.05)
    return normalise_peak(x, peak=1.0)


def make_repeated_random_template(
    fs: int = 48_000,
    half_duration: float = 0.18,
    f_min: float = 500.0,
    f_max: float = 8_000.0,
    seed: int = 123,
) -> Array:
    """Two identical halves for a Schmidl-Cox-like self-similarity metric."""
    half = make_bandlimited_noise(fs, half_duration, f_min, f_max, seed=seed)
    x = np.concatenate([half, half])
    x *= signal.windows.tukey(len(x), alpha=0.03)
    return normalise_peak(x, peak=1.0)


@dataclass
class ProbeMetadata:
    fs: int
    sync_kind: str
    sync_start: int
    sync_len: int
    repeated_start: int
    repeated_len: int
    repeated_half_len: int
    training_start: int
    training_len: int
    f_min: float
    f_max: float
    gap_samples: int


def make_measurement_waveform(
    fs: int = 48_000,
    sync_kind: str = "chirp",
    f_min: float = 500.0,
    f_max: float = 8_000.0,
) -> Tuple[Array, Array, Array, Array, ProbeMetadata]:
    """Return full measurement TX waveform and its component templates."""
    if sync_kind == "chirp":
        sync = make_chirp_template(fs, f0=f_min, f1=f_max, method="logarithmic")
    elif sync_kind == "step":
        sync = make_step_template(fs)
    else:
        raise ValueError("sync_kind must be 'chirp' or 'step'")

    repeated = make_repeated_random_template(fs, f_min=f_min, f_max=f_max)
    training = make_bandlimited_noise(fs, duration=1.25, f_min=f_min, f_max=f_max, seed=9)

    pre = np.zeros(seconds_to_samples(fs, 0.25))
    gap = np.zeros(seconds_to_samples(fs, 0.12))
    post = np.zeros(seconds_to_samples(fs, 0.35))

    sync_start = len(pre)
    repeated_start = sync_start + len(sync) + len(gap)
    training_start = repeated_start + len(repeated) + len(gap)

    tx = np.concatenate([pre, sync, gap, repeated, gap, training, post])
    tx = normalise_peak(tx, peak=0.75)

    meta = ProbeMetadata(
        fs=fs,
        sync_kind=sync_kind,
        sync_start=sync_start,
        sync_len=len(sync),
        repeated_start=repeated_start,
        repeated_len=len(repeated),
        repeated_half_len=len(repeated) // 2,
        training_start=training_start,
        training_len=len(training),
        f_min=f_min,
        f_max=f_max,
        gap_samples=len(gap),
    )
    return tx, sync, repeated, training, meta


def make_probes(out_dir: Path, fs: int = 48_000, sync_kind: str = "chirp") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tx, sync, repeated, training, meta = make_measurement_waveform(fs=fs, sync_kind=sync_kind)
    write_wav(out_dir / "measurement_tx.wav", fs, tx, peak=0.75)
    write_wav(out_dir / "sync_template.wav", fs, sync, peak=0.8)
    write_wav(out_dir / "repeated_template.wav", fs, repeated, peak=0.8)
    write_wav(out_dir / "training_template.wav", fs, training, peak=0.8)
    (out_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2))
    print(f"Wrote probes to {out_dir.resolve()}")


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def make_room_channel(
    fs: int = 48_000,
    max_len: int = 2_400,
    seed: int = 5,
) -> Array:
    """
    Synthetic acoustic channel with direct path + sparse echoes + weak tail.

    At 48 kHz, 1500 samples is about 31 ms, matching the order of delay spread
    expected from a first strong wall echo in a medium-sized room.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(max_len)
    echoes = [
        (0, 1.00),
        (155, 0.35),
        (430, -0.22),
        (910, 0.16),
        (1_520, -0.12),
        (2_050, 0.07),
    ]
    for delay, gain in echoes:
        if delay < max_len:
            h[delay] += gain

    # Small diffuse reverberant tail, decaying with time.
    tail_start = 250
    tail = rng.standard_normal(max_len - tail_start)
    tail *= np.exp(-np.arange(len(tail)) / (0.020 * fs))
    h[tail_start:] += 0.025 * tail
    h /= np.sqrt(np.sum(h**2) + EPS)
    return h


def apply_channel(
    tx: Array,
    h: Array,
    snr_db: float = 28.0,
    leading_delay: int = 3_000,
    clipping: Optional[float] = None,
    seed: int = 7,
) -> Array:
    """Convolve, add delay/noise, optionally clip."""
    rng = np.random.default_rng(seed)
    y = signal.fftconvolve(tx, h, mode="full")
    y = np.concatenate([np.zeros(leading_delay), y])
    sig_power = np.mean(y**2) + EPS
    noise_power = sig_power / (10 ** (snr_db / 10))
    y += rng.standard_normal(len(y)) * np.sqrt(noise_power)
    if clipping is not None:
        y = np.clip(y, -clipping, clipping)
    return y


# -----------------------------------------------------------------------------
# Synchronisation methods
# -----------------------------------------------------------------------------

def matched_filter_sync(rx: Array, template: Array) -> Tuple[int, Array]:
    """
    Normalised sliding matched-filter score.

    Returns:
        best_start: estimated index in rx where template starts
        score: normalised correlation score for each possible start
    """
    rx = np.asarray(rx, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    template = template - np.mean(template)
    rx0 = rx - np.mean(rx)
    n = len(template)

    # valid correlation: score[k] = sum rx[k:k+n] * template
    corr = signal.correlate(rx0, template, mode="valid", method="fft")
    template_energy = np.sum(template**2) + EPS
    win_energy = signal.convolve(rx0**2, np.ones(n), mode="valid", method="fft") + EPS
    score = corr / np.sqrt(template_energy * win_energy)
    best_start = int(np.argmax(np.abs(score)))
    return best_start, score


def schmidl_cox_metric(rx: Array, half_len: int) -> Array:
    """
    Self-similarity metric for two repeated halves.

    M[d] = |sum_k r[d+k] r*[d+k+L]|^2 / (sum_k |r[d+k+L]|^2)^2
    For real audio, conjugation is harmless but kept for the standard form.
    """
    r = np.asarray(rx, dtype=np.float64)
    L = int(half_len)
    if len(r) < 2 * L + 1:
        return np.array([], dtype=np.float64)

    a = r[:-L] * r[L:]
    p = signal.convolve(a, np.ones(L), mode="valid", method="fft")
    e = signal.convolve(r[L:] ** 2, np.ones(L), mode="valid", method="fft") + EPS
    return (np.abs(p) ** 2) / (e**2)


def peak_to_sidelobe_ratio(score: Array, guard: int = 2_000) -> float:
    """Simple quality metric: main peak divided by largest non-guard sidelobe."""
    a = np.abs(score).copy()
    if len(a) == 0:
        return np.nan
    peak = int(np.argmax(a))
    main = a[peak] + EPS
    lo = max(0, peak - guard)
    hi = min(len(a), peak + guard + 1)
    a[lo:hi] = 0.0
    side = np.max(a) + EPS
    return float(main / side)


# -----------------------------------------------------------------------------
# Channel estimation
# -----------------------------------------------------------------------------

def estimate_channel_freqdomain(
    x: Array,
    y: Array,
    h_len: int = 2_400,
    reg: float = 1e-4,
) -> Array:
    """
    Regularised frequency-domain LS estimate.

    Model: y ≈ x * h.
    H_hat[k] = Y[k] X*[k] / (|X[k]|^2 + λ max|X|^2)
    Then transform H_hat back into an impulse response.
    """
    n_fft = int(2 ** np.ceil(np.log2(len(x) + len(y) + h_len)))
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    lam = reg * np.max(np.abs(X) ** 2 + EPS)
    H = Y * np.conj(X) / (np.abs(X) ** 2 + lam)
    h = np.fft.irfft(H, n=n_fft)[:h_len]
    return h.astype(np.float64)


def estimate_channel_sampled_lstsq(
    x: Array,
    y: Array,
    h_len: int = 1_200,
    ridge: float = 1e-3,
    max_rows: int = 8_000,
) -> Array:
    """
    Sampled regularised linear least-squares estimate.

    Uses rows y[n] = h[0]x[n] + h[1]x[n-1] + ... + h[L-1]x[n-L+1].
    Row sampling keeps memory modest while still showing the true LS model.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    max_n = min(len(y), len(x))
    rows = np.arange(h_len - 1, max_n)
    if len(rows) > max_rows:
        rows = np.linspace(rows[0], rows[-1], max_rows).astype(int)

    Xmat = np.empty((len(rows), h_len), dtype=np.float64)
    offsets = np.arange(h_len)
    for i, n in enumerate(rows):
        Xmat[i, :] = x[n - offsets]
    yr = y[rows]

    A = Xmat.T @ Xmat
    b = Xmat.T @ yr
    A.flat[:: h_len + 1] += ridge * np.trace(A) / h_len + EPS
    h = np.linalg.solve(A, b)
    return h.astype(np.float64)


def frequency_response(h: Array, fs: int, n_fft: int = 8192) -> Tuple[Array, Array]:
    H = np.fft.rfft(h, n=n_fft)
    f = np.fft.rfftfreq(n_fft, d=1 / fs)
    return f, H


# -----------------------------------------------------------------------------
# OFDM/DMT constellation demo
# -----------------------------------------------------------------------------

def qpsk_symbols(bits: np.ndarray) -> np.ndarray:
    """Gray-ish QPSK mapping normalised to unit power."""
    b = bits.reshape(-1, 2)
    real = np.where(b[:, 0] == 0, 1.0, -1.0)
    imag = np.where(b[:, 1] == 0, 1.0, -1.0)
    return (real + 1j * imag) / np.sqrt(2)


def ofdm_equalisation_demo(
    h_true: Array,
    h_est: Array,
    fs: int,
    out_dir: Path,
    n_fft: int = 4_096,
    cp: int = 2_400,
    f_min: float = 500.0,
    f_max: float = 8_000.0,
    seed: int = 1,
) -> Dict[str, float]:
    """Create a simple real-valued DMT symbol and show before/after equalisation."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(n_fft, d=1 / fs)
    pos_bins = np.where((freqs > f_min) & (freqs < f_max))[0]
    pos_bins = pos_bins[pos_bins < n_fft // 2]

    bits = rng.integers(0, 2, size=2 * len(pos_bins))
    S = qpsk_symbols(bits)
    X = np.zeros(n_fft, dtype=np.complex128)
    X[pos_bins] = S
    X[-pos_bins] = np.conj(S)

    x_time = np.fft.ifft(X).real
    x_cp = np.concatenate([x_time[-cp:], x_time])
    y = signal.fftconvolve(x_cp, h_true, mode="full")
    y += rng.standard_normal(len(y)) * 0.01 * np.sqrt(np.mean(y**2) + EPS)

    # Ideal timing for the demo; the sync problem is handled elsewhere.
    y_block = y[cp : cp + n_fft]
    Y = np.fft.fft(y_block)
    H_est = np.fft.fft(h_est, n=n_fft)
    Z = Y[pos_bins] / (H_est[pos_bins] + 1e-3)
    raw = Y[pos_bins]

    # One complex gain ambiguity remains because h_est is not perfect. Remove it
    # only for fair visual comparison of the equalised cloud.
    alpha = np.vdot(Z, S) / (np.vdot(Z, Z) + EPS)
    Z_aligned = alpha * Z

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(raw.real, raw.imag, s=10, alpha=0.5, label="received bins before EQ")
    ax.scatter(Z_aligned.real, Z_aligned.imag, s=10, alpha=0.7, label="after one-tap EQ")
    ax.scatter(S.real, S.imag, marker="x", s=70, label="ideal QPSK")
    ax.set_title("OFDM/DMT constellation: before and after one-tap equalisation")
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_ofdm_constellation_equalisation.png", dpi=180)
    plt.close(fig)

    evm = np.sqrt(np.mean(np.abs(Z_aligned - S) ** 2) / (np.mean(np.abs(S) ** 2) + EPS))
    return {"num_used_bins": int(len(pos_bins)), "evm_after_equalisation": float(evm)}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def save_sync_plot(
    rx: Array,
    fs: int,
    score: Array,
    best_start: int,
    out_path: Path,
    title: str,
) -> None:
    t_rx = np.arange(len(rx)) / fs
    t_score = np.arange(len(score)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(t_rx, rx, linewidth=0.8)
    axes[0].axvline(best_start / fs, linestyle="--", label="estimated sync start")
    axes[0].set_title(title)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_score, np.abs(score), linewidth=0.9)
    axes[1].axvline(best_start / fs, linestyle="--")
    axes[1].set_xlabel("Time in recording / s")
    axes[1].set_ylabel("|normalised correlation|")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_schmidl_plot(metric: Array, fs: int, best_repeated_start: int, out_path: Path) -> None:
    t = np.arange(len(metric)) / fs
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, metric, linewidth=0.9)
    ax.axvline(best_repeated_start / fs, linestyle="--", label="metric peak")
    ax.set_title("Repeated-half self-similarity metric")
    ax.set_xlabel("Time in recording / s")
    ax.set_ylabel("M[d]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_channel_plots(
    h_fd: Array,
    h_ls: Array,
    fs: int,
    out_dir: Path,
    h_true: Optional[Array] = None,
    f_min: float = 500.0,
    f_max: float = 8_000.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(len(h_fd), len(h_ls), len(h_true) if h_true is not None else 0)
    t_ms = np.arange(n) / fs * 1000

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if h_true is not None:
        ax.plot(t_ms[: len(h_true)], h_true, linewidth=1.0, label="true channel")
    ax.plot(t_ms[: len(h_fd)], h_fd, linewidth=1.0, label="freq-domain estimate")
    ax.plot(t_ms[: len(h_ls)], h_ls, linewidth=1.0, alpha=0.8, label="sampled LS estimate")
    ax.set_title("Estimated impulse response")
    ax.set_xlabel("Delay / ms")
    ax.set_ylabel("h[n]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "03_channel_impulse_response.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for h, label in [(h_true, "true"), (h_fd, "freq-domain estimate"), (h_ls, "sampled LS estimate")]:
        if h is None:
            continue
        f, H = frequency_response(h, fs)
        use = (f >= f_min) & (f <= f_max)
        mag_db = 20 * np.log10(np.abs(H) + EPS)
        ax.plot(f[use], mag_db[use], linewidth=1.0, label=label)
    ax.set_title("Channel frequency response magnitude")
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("|H(f)| / dB")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "04_channel_frequency_response_magnitude.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for h, label in [(h_true, "true"), (h_fd, "freq-domain estimate"), (h_ls, "sampled LS estimate")]:
        if h is None:
            continue
        f, H = frequency_response(h, fs)
        use = (f >= f_min) & (f <= f_max)
        phase = np.unwrap(np.angle(H))
        ax.plot(f[use], phase[use], linewidth=1.0, label=label)
    ax.set_title("Channel frequency response phase")
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Unwrapped phase / rad")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_channel_frequency_response_phase.png", dpi=180)
    plt.close(fig)


def save_spectrogram(x: Array, fs: int, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.specgram(x, NFFT=1024, Fs=fs, noverlap=768)
    ax.set_ylim(0, 10_000)
    ax.set_title(title)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Frequency / Hz")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main analysis pipeline
# -----------------------------------------------------------------------------

def analyse_recording(
    rx: Array,
    fs: int,
    sync: Array,
    repeated: Array,
    training: Array,
    meta: ProbeMetadata,
    out_dir: Path,
    h_len: int = 2_400,
    h_true: Optional[Array] = None,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Matched-filter synchronisation on the external sync template.
    sync_start_rx, sync_score = matched_filter_sync(rx, sync)
    save_sync_plot(
        rx,
        fs,
        sync_score,
        sync_start_rx,
        out_dir / "01_matched_filter_sync.png",
        f"Matched-filter synchronisation using {meta.sync_kind} template",
    )

    # 2) Repeated-half self-similarity metric.
    metric = schmidl_cox_metric(rx, meta.repeated_half_len)
    repeated_peak = int(np.argmax(metric)) if len(metric) else -1
    if len(metric):
        save_schmidl_plot(metric, fs, repeated_peak, out_dir / "02_repeated_half_metric.png")

    # 3) Align the training region using the known TX offsets relative to sync.
    offset_training_from_sync = meta.training_start - meta.sync_start
    train_start_rx = sync_start_rx + offset_training_from_sync
    y0 = max(0, train_start_rx)
    y1 = min(len(rx), train_start_rx + len(training) + h_len - 1)
    y_train = rx[y0:y1]

    # Pad if recording ends too early.
    needed = len(training) + h_len - 1
    if len(y_train) < needed:
        y_train = np.pad(y_train, (0, needed - len(y_train)))

    # 4) Estimate the channel in two ways.
    h_fd = estimate_channel_freqdomain(training, y_train, h_len=h_len, reg=1e-4)
    h_ls = estimate_channel_sampled_lstsq(training, y_train, h_len=min(2_400, h_len), ridge=1e-3)
    save_channel_plots(h_fd, h_ls, fs, out_dir, h_true=h_true, f_min=meta.f_min, f_max=meta.f_max)
    save_spectrogram(rx, fs, out_dir / "00_recording_spectrogram.png", "Received signal spectrogram")

    # 5) Optional OFDM/DMT equalisation demonstration if true channel is known.
    ofdm_metrics: Dict[str, float] = {}
    if h_true is not None:
        ofdm_metrics = ofdm_equalisation_demo(h_true, h_ls, fs, out_dir, f_min=meta.f_min, f_max=meta.f_max)

    results = {
        "fs": float(fs),
        "matched_filter_sync_start_sample": float(sync_start_rx),
        "matched_filter_sync_start_seconds": float(sync_start_rx / fs),
        "matched_filter_peak_score": float(np.max(np.abs(sync_score))),
        "matched_filter_peak_to_sidelobe_ratio": peak_to_sidelobe_ratio(sync_score),
        "repeated_half_peak_sample": float(repeated_peak),
        "repeated_half_peak_seconds": float(repeated_peak / fs) if repeated_peak >= 0 else np.nan,
        "estimated_training_start_sample": float(train_start_rx),
        "estimated_training_start_seconds": float(train_start_rx / fs),
        **ofdm_metrics,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.save(out_dir / "h_est_freqdomain.npy", h_fd)
    np.save(out_dir / "h_est_sampled_lstsq.npy", h_ls)
    return results


def load_probe_dir(probe_dir: Path) -> Tuple[int, Array, Array, Array, ProbeMetadata]:
    fs_sync, sync = read_wav(probe_dir / "sync_template.wav")
    fs_rep, repeated = read_wav(probe_dir / "repeated_template.wav")
    fs_train, training = read_wav(probe_dir / "training_template.wav")
    if not (fs_sync == fs_rep == fs_train):
        raise ValueError("Probe WAV files have inconsistent sample rates.")
    meta_dict = json.loads((probe_dir / "metadata.json").read_text())
    meta = ProbeMetadata(**meta_dict)
    return fs_sync, sync, repeated, training, meta


def run_demo(out_dir: Path, fs: int = 48_000, sync_kind: str = "chirp") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tx, sync, repeated, training, meta = make_measurement_waveform(fs=fs, sync_kind=sync_kind)
    h_true = make_room_channel(fs=fs, max_len=2_400)
    rx = apply_channel(tx, h_true, snr_db=25.0, leading_delay=3_000, clipping=None)

    write_wav(out_dir / "demo_tx.wav", fs, tx, peak=0.75)
    write_wav(out_dir / "demo_rx_simulated.wav", fs, rx, peak=0.75)
    np.save(out_dir / "h_true.npy", h_true)
    (out_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2))

    results = analyse_recording(rx, fs, sync, repeated, training, meta, out_dir, h_len=2_400, h_true=h_true)
    print(json.dumps(results, indent=2))
    print(f"Demo plots written to {out_dir.resolve()}")


def run_analyse(probe_dir: Path, rx_path: Path, out_dir: Path, h_len: int = 2_400) -> None:
    fs_probe, sync, repeated, training, meta = load_probe_dir(probe_dir)
    fs_rx, rx = read_wav(rx_path)
    if fs_rx != fs_probe:
        raise ValueError(f"Sample rate mismatch: probes are {fs_probe} Hz, recording is {fs_rx} Hz")
    results = analyse_recording(rx, fs_rx, sync, repeated, training, meta, out_dir, h_len=h_len)
    print(json.dumps(results, indent=2))
    print(f"Analysis plots written to {out_dir.resolve()}")


# -----------------------------------------------------------------------------
# Optional live audio I/O helper
# -----------------------------------------------------------------------------

def live_play_record(tx_path: Path, out_rx_path: Path, seconds_extra: float = 0.75) -> None:
    """
    Optional helper: play a WAV and record from the default microphone.

    Requires: pip install sounddevice
    On macOS, give Terminal/VS Code microphone permission.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Install sounddevice first: pip install sounddevice") from exc

    fs, tx = read_wav(tx_path)
    n_record = len(tx) + seconds_to_samples(fs, seconds_extra)
    print(f"Recording {n_record / fs:.2f} s at {fs} Hz while playing {tx_path}...")
    rec = sd.playrec(tx, samplerate=fs, channels=1, dtype="float64")
    sd.wait()

    # If playrec only records the same length as tx, extend via a second recording block.
    rec = np.asarray(rec).reshape(-1)
    if len(rec) < n_record:
        extra = sd.rec(n_record - len(rec), samplerate=fs, channels=1, dtype="float64")
        sd.wait()
        rec = np.concatenate([rec, np.asarray(extra).reshape(-1)])

    write_wav(out_rx_path, fs, rec, peak=0.8)
    print(f"Wrote recording to {out_rx_path.resolve()}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GF3 synchronisation/channel-estimation measurement lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("make-probes", help="generate WAV probes for real measurements")
    p_probe.add_argument("--out", type=Path, default=Path("probes"))
    p_probe.add_argument("--fs", type=int, default=48_000)
    p_probe.add_argument("--sync-kind", choices=["chirp", "step"], default="chirp")

    p_demo = sub.add_parser("demo", help="run a simulated demo and create plots")
    p_demo.add_argument("--out", type=Path, default=Path("results_demo"))
    p_demo.add_argument("--fs", type=int, default=48_000)
    p_demo.add_argument("--sync-kind", choices=["chirp", "step"], default="chirp")

    p_analyse = sub.add_parser("analyse", help="analyse a real recorded WAV")
    p_analyse.add_argument("--probe-dir", type=Path, required=True)
    p_analyse.add_argument("--rx", type=Path, required=True)
    p_analyse.add_argument("--out", type=Path, default=Path("results_real"))
    p_analyse.add_argument("--h-len", type=int, default=2_400)

    p_live = sub.add_parser("live", help="optional live play-record helper")
    p_live.add_argument("--tx", type=Path, required=True)
    p_live.add_argument("--rx-out", type=Path, required=True)
    p_live.add_argument("--seconds-extra", type=float, default=0.75)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "make-probes":
        make_probes(args.out, fs=args.fs, sync_kind=args.sync_kind)
    elif args.cmd == "demo":
        run_demo(args.out, fs=args.fs, sync_kind=args.sync_kind)
    elif args.cmd == "analyse":
        run_analyse(args.probe_dir, args.rx, args.out, h_len=args.h_len)
    elif args.cmd == "live":
        live_play_record(args.tx, args.rx_out, seconds_extra=args.seconds_extra)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
