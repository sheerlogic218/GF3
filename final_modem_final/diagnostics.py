"""JOSS-F receiver diagnostics.

Generates five diagnostic figures that are saved to
``<output_dir>/diagnostic_plots/`` beside the decoded WAV file, and shown on
screen when a GUI backend is available.

Figure layout
─────────────
  <stem>_1_sync.png          – Chirp matched-filter, detected chain, SFO fit
  <stem>_2_spectrogram.png   – Time-frequency map of corrected signal
  <stem>_3_channel.png       – Channel magnitude & phase across data bins
  <stem>_4_constellation.png – Raw vs equalised QPSK scatter (first 6 blocks)
  <stem>_5_ldpc_cp.png       – CP self-correlation + LDPC iteration breakdown

Usage
─────
Automatically called by Receiver.decode_wav().  Can also be called directly::

    from diagnostics import plot_all
    plot_all(diag_dict, output_dir=Path("diagnostic_plots"), wav_stem="my_rec")

The *diag* dict is populated by Receiver.decode_raw_signal() when called with
``_diag={}``.  Missing keys are handled gracefully – figures degrade to an
"unavailable" placeholder rather than raising.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
from numpy.typing import NDArray
from scipy.signal import spectrogram as _scipy_spectrogram

from modem import CONFIG, ModemConfig

__all__ = ["plot_all"]

# ── palette ──────────────────────────────────────────────────────────────────

_B  = "#3a7fc1"   # blue     – primary series
_O  = "#e07828"   # orange   – secondary series / pilots
_G  = "#3aab4c"   # green    – success / OK
_R  = "#d43f3f"   # red      – failure / warning
_P  = "#8060c8"   # purple   – residuals
_K  = "#555555"   # dark-grey – neutral lines
_L  = "#bbbbbb"   # light-grey – faint background

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         _L,
    "grid.linewidth":     0.5,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# ── public entry point ────────────────────────────────────────────────────────

def plot_all(
    diag: dict[str, Any],
    output_dir: Path,
    wav_stem: str = "recording",
    config: ModemConfig = CONFIG,
    *,
    show: bool = True,
) -> list[Path]:
    """Render all diagnostic figures, save to *output_dir*, and show.

    Returns the list of saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, Any]] = [
        ("1_sync",          _fig_sync),
        ("2_spectrogram",   _fig_spectrogram),
        ("3_channel",       _fig_channel),
        ("4_constellation", _fig_constellation),
        ("5_ldpc_cp",       _fig_ldpc_cp),
    ]

    saved: list[Path] = []
    for tag, fn in specs:
        try:
            fig = fn(diag, config)
        except Exception as exc:
            warnings.warn(f"[diagnostics] {tag} skipped: {exc}", stacklevel=2)
            continue
        path = output_dir / f"{wav_stem}_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        print(f"  [diag] {path.name}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close("all")

    return saved


# ── helpers ───────────────────────────────────────────────────────────────────

def _missing(ax: plt.Axes, msg: str = "data not available") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color=_K,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec=_L))
    ax.set_xticks([])
    ax.set_yticks([])


def _ds(arr: NDArray, target: int = 10_000) -> tuple[NDArray, NDArray]:
    """Return (x_indices, values) downsampled to *target* points for fast plot."""
    n = len(arr)
    if n == 0:
        return np.array([]), np.array([])
    step = max(1, n // target)
    idx = np.arange(0, n, step)
    return idx, arr[idx]


# ── Figure 1: Synchronisation & SFO ──────────────────────────────────────────

def _fig_sync(diag: dict, config: ModemConfig) -> plt.Figure:
    received    = diag.get("received")
    chirp_metric = diag.get("chirp_metric")
    all_peaks   = diag.get("all_peaks", np.array([], dtype=int))
    best_chain  = diag.get("best_chain", [])
    sfo_ppm     = diag.get("sfo_ppm", 0.0)
    sync_sample = diag.get("sync_sample", 0)
    sfo_ratio   = diag.get("sfo_ratio", 1.0)
    min_chain   = diag.get("min_chain_required", 6)

    fs = config.sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(
        f"Synchronisation & SFO  |  estimated: {sfo_ppm:+.1f} ppm  "
        f"(ratio = {sfo_ratio:.8f})  |  chain length: {len(best_chain)}/{config.chirp_count}",
        fontsize=12,
    )

    # ── 1a: waveform ──
    ax = axes[0]
    if received is not None and len(received):
        idx, vals = _ds(received)
        ax.plot(idx / fs, vals, color=_B, lw=0.5, alpha=0.8)
        ax.axvline(sync_sample / fs, color=_R, lw=1.5,
                   label=f"sync @ sample {sync_sample}")
        ofdm_start = sync_sample + config.preamble_length
        if ofdm_start < len(received):
            ax.axvline(ofdm_start / fs, color=_G, lw=1.5, ls="--",
                       label=f"OFDM start @ {ofdm_start}")
        ax.legend(fontsize=8, loc="upper right")
    else:
        _missing(ax)
    ax.set_ylabel("amplitude")
    ax.set_title("(a) Received waveform")
    ax.set_xlabel("time (s)")

    # ── 1b: matched-filter metric ──
    ax = axes[1]
    if chirp_metric is not None and len(chirp_metric):
        idx, vals = _ds(chirp_metric)
        ax.plot(idx / fs, vals, color=_K, lw=0.5, alpha=0.7, label="metric")
        if len(all_peaks):
            ax.scatter(
                all_peaks / fs, chirp_metric[all_peaks],
                s=18, c=_O, zorder=3, label=f"all peaks ({len(all_peaks)})",
            )
        if len(best_chain):
            ca = np.asarray(best_chain, dtype=int)
            ax.scatter(
                ca / fs, chirp_metric[ca],
                s=60, c=_G, marker="^", zorder=4,
                label=f"best chain (n={len(best_chain)}, min={min_chain})",
            )
        ax.legend(fontsize=8, loc="upper right")
    else:
        _missing(ax)
    ax.set_ylabel("|correlation|")
    ax.set_title("(b) Chirp matched-filter metric")
    ax.set_xlabel("time (s)")

    # ── 1c: SFO linear fit ──
    ax = axes[2]
    if len(best_chain) >= 2:
        observed = np.asarray(best_chain, dtype=float)
        cn = np.arange(len(observed), dtype=float)

        # Sub-sample refinement mirror from receiver (if stored) or plain peaks
        period, intercept = np.polyfit(cn, observed, 1)
        fitted = period * cn + intercept
        residuals = observed - fitted

        ax.plot(cn, observed, "o", color=_B, ms=7, label="observed peaks")
        ax.plot(cn, fitted, "--", color=_R, lw=1.8,
                label=f"linear fit: period={period:.3f} smp  ({sfo_ppm:+.1f} ppm)")

        ax2 = ax.twinx()
        ax2.plot(cn, residuals, "s-", color=_P, lw=1.2, ms=5, alpha=0.75,
                 label="residuals")
        ax2.axhline(0, color=_P, lw=0.6, ls=":")
        ax2.set_ylabel("residual (samples)", color=_P, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=_P)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(_P)
        ax2.spines["top"].set_visible(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    else:
        _missing(ax, f"Chain too short for SFO fit\n(need ≥ 2, got {len(best_chain)})")
    ax.set_xlabel("chirp index")
    ax.set_ylabel("peak position (samples)")
    ax.set_title("(c) Chirp peak positions & SFO linear fit")

    return fig


# ── Figure 2: Spectrogram ─────────────────────────────────────────────────────

def _fig_spectrogram(diag: dict, config: ModemConfig) -> plt.Figure:
    corrected   = diag.get("corrected")
    sfo_ppm     = diag.get("sfo_ppm", 0.0)

    fs = config.sample_rate

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        f"Spectrogram of SFO-corrected signal  (SFO = {sfo_ppm:+.1f} ppm)",
        fontsize=12,
    )

    if corrected is None or len(corrected) == 0:
        _missing(ax)
        return fig

    # Show slightly past the OFDM start so we can see the first few symbols
    view = corrected[: min(len(corrected), config.preamble_length + 20 * config.ofdm_symbol_length)]

    nperseg  = 512
    noverlap = 384
    f, t, Sxx = _scipy_spectrogram(view, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                   window="hann")

    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    vmin   = np.percentile(Sxx_db, 5)
    vmax   = np.percentile(Sxx_db, 99)

    im = ax.pcolormesh(t, f / 1000, Sxx_db, vmin=vmin, vmax=vmax,
                       cmap="inferno", shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax, label="power (dB)", pad=0.01)

    # Data band limits
    for fhz, ls, lbl in [
        (config.data_low_hz,  "--", f"{config.data_low_hz/1000:.0f} kHz"),
        (config.data_high_hz, ":",  f"{config.data_high_hz/1000:.0f} kHz"),
    ]:
        ax.axhline(fhz / 1000, color="cyan", lw=1.2, ls=ls, alpha=0.9, label=lbl)

    # Preamble section boundaries (time in the corrected signal)
    chirp_end_s  = config.chirp_train_length / fs
    ofdm_start_s = config.preamble_length    / fs
    ax.axvline(chirp_end_s,  color="lime",   lw=1.5, label=f"chirp end ({chirp_end_s:.3f} s)")
    ax.axvline(ofdm_start_s, color="yellow", lw=1.5, label=f"OFDM start ({ofdm_start_s:.3f} s)")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_ylim(0, fs / 2000)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.legend(fontsize=8, loc="upper right")

    return fig


# ── Figure 3: Channel estimation ─────────────────────────────────────────────

def _fig_channel(diag: dict, config: ModemConfig) -> plt.Figure:
    golay_h       = diag.get("golay_channel")
    pilot_channels = diag.get("pilot_channels", {})

    data_bins  = config.data_bins
    freqs_khz  = data_bins * config.sample_rate / config.fft_length / 1000

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    fig.suptitle("Channel frequency-response estimate  (Golay + OFDM pilots)", fontsize=12)

    ax_mag, ax_phi = axes

    def _plot_h(ax_m: plt.Axes, ax_p: plt.Axes,
                h: NDArray, label: str, color: str,
                lw: float = 1.5, alpha: float = 1.0) -> None:
        h_d   = h[data_bins]
        mag   = 20 * np.log10(np.abs(h_d) + 1e-14)
        phase = np.unwrap(np.angle(h_d)) * 180 / np.pi
        kw    = dict(color=color, lw=lw, alpha=alpha, label=label)
        ax_m.plot(freqs_khz, mag,   **kw)
        ax_p.plot(freqs_khz, phase, **kw)

    if golay_h is not None:
        _plot_h(ax_mag, ax_phi, golay_h, "Golay estimate", _B, lw=2.0)
    else:
        _missing(ax_mag, "Golay channel not available")

    pilot_colours = [_O, _G, _P, _R]
    for i, (blk, h) in enumerate(sorted(pilot_channels.items())[:4]):
        _plot_h(ax_mag, ax_phi, h,
                f"Pilot #{blk}", pilot_colours[i % 4],
                lw=1.2, alpha=0.7)

    ax_mag.set_ylabel("magnitude (dB)")
    ax_mag.set_title("(a) Magnitude")
    if ax_mag.get_legend_handles_labels()[1]:
        ax_mag.legend(fontsize=8)

    ax_phi.set_xlabel("frequency (kHz)")
    ax_phi.set_ylabel("phase (deg, unwrapped)")
    ax_phi.set_title("(b) Phase")
    if ax_phi.get_legend_handles_labels()[1]:
        ax_phi.legend(fontsize=8)

    return fig


# ── Figure 4: Constellation ───────────────────────────────────────────────────

def _fig_constellation(diag: dict, config: ModemConfig) -> plt.Figure:
    raw_spectra_bins = diag.get("raw_spectra_bins", [])   # list of (n_bins,) arrays
    equalised_rows   = diag.get("equalised_rows",   [])   # list of (n_bins,) arrays

    n_show  = 6
    ideal   = np.array([1+1j, -1+1j, 1-1j, -1-1j])

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6),
                             constrained_layout=True)
    fig.suptitle(
        "Constellation diagrams – first 6 data blocks\n"
        "(top: raw before equalization, bottom: after equalization  |  "
        "colour = distance from nearest ideal point)",
        fontsize=11,
    )

    for col in range(n_show):
        # --- top row: raw spectrum at data bins (no channel correction) ---
        ax = axes[0, col]
        if col < len(raw_spectra_bins):
            sym = np.asarray(raw_spectra_bins[col])
            # Normalise median magnitude to ~sqrt(2) so it fits the plot
            med = float(np.median(np.abs(sym))) + 1e-12
            sym_n = sym * (np.sqrt(2.0) / med)
            ax.scatter(sym_n.real, sym_n.imag, s=0.4, c=_K, alpha=0.35, rasterized=True)
        else:
            _missing(ax, "—")
        for pt in ideal:
            ax.plot(pt.real, pt.imag, "r+", ms=10, mew=2, zorder=5)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(f"Block {col + 1}", fontsize=9)
        ax.axhline(0, color=_L, lw=0.8); ax.axvline(0, color=_L, lw=0.8)
        if col == 0:
            ax.set_ylabel("raw\nImag", fontsize=9)

        # --- bottom row: equalized ---
        ax = axes[1, col]
        if col < len(equalised_rows):
            sym = np.asarray(equalised_rows[col])
            dists = np.min(np.abs(sym[:, None] - ideal[None, :]), axis=1)
            sc = ax.scatter(sym.real, sym.imag, s=0.4, c=dists,
                            cmap="RdYlGn_r", vmin=0.0, vmax=2.0,
                            alpha=0.5, rasterized=True)
            evm_rms = float(np.sqrt(np.mean(dists ** 2)))
            ax.set_title(f"EVM {evm_rms:.3f}", fontsize=8)
        else:
            _missing(ax, "—")
        for pt in ideal:
            ax.plot(pt.real, pt.imag, "b+", ms=10, mew=2, zorder=5)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.axhline(0, color=_L, lw=0.8); ax.axvline(0, color=_L, lw=0.8)
        if col == 0:
            ax.set_ylabel("equalised\nImag", fontsize=9)

    return fig


# ── Figure 5: LDPC stats + CP correlation ─────────────────────────────────────

def _fig_ldpc_cp(diag: dict, config: ModemConfig) -> plt.Figure:
    ldpc_iterations = diag.get("ldpc_iterations", [])
    ldpc_successes  = diag.get("ldpc_successes",  [])
    cp_offsets      = diag.get("cp_offsets")
    cp_scores       = diag.get("cp_scores")
    cp_best_offset  = diag.get("cp_best_offset", 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("CP self-correlation (data-start refinement)  &  LDPC iteration breakdown",
                 fontsize=12)

    # ── 5a: CP self-correlation ──
    ax = axes[0]
    if cp_offsets is not None and cp_scores is not None and len(cp_offsets):
        ax.plot(cp_offsets, cp_scores, color=_B, lw=1.5, label="CP score")
        ax.axvline(cp_best_offset, color=_R, lw=2, ls="--",
                   label=f"chosen offset: {cp_best_offset:+d} samples")
        ax.set_xlabel("offset from nominal OFDM start (samples)")
        ax.set_ylabel("normalised CP correlation")
        ax.set_title("(a) Cyclic-prefix self-correlation")
        ax.legend(fontsize=9)
    else:
        _missing(ax, "CP scores not available")
        ax.set_title("(a) Cyclic-prefix self-correlation")

    # ── 5b: LDPC iterations ──
    ax = axes[1]
    if ldpc_iterations:
        n      = len(ldpc_iterations)
        iters  = np.asarray(ldpc_iterations)
        ok     = np.asarray(ldpc_successes) if ldpc_successes else np.zeros(n, bool)
        colors = [_G if b else _R for b in ok]

        ax.bar(np.arange(n), iters, color=colors, width=1.0, edgecolor="none")
        ax.axhline(config.ldpc_max_iterations, color=_K, lw=1.2, ls=":",
                   label=f"max iters ({config.ldpc_max_iterations})")

        n_ok   = int(np.sum(ok))
        n_fail = n - n_ok
        legend_patches = [
            Patch(fc=_G, label=f"converged ({n_ok})"),
            Patch(fc=_R, label=f"failed ({n_fail})"),
        ]
        ax.legend(handles=legend_patches, fontsize=9)
        ax.set_xlabel("LDPC block index")
        ax.set_ylabel("decoder iterations")
        ax.set_title(f"(b) LDPC per-block iterations  ({n_ok}/{n} = "
                     f"{100*n_ok/max(n,1):.0f}% success)")
        ax.set_xlim(-0.5, max(n, 1) - 0.5)
        ax.set_ylim(0, config.ldpc_max_iterations + 5)
        # Mean-iteration line
        ax.axhline(float(np.mean(iters)), color=_O, lw=1.2, ls="--",
                   label=f"mean = {np.mean(iters):.1f}")
        ax.legend(handles=legend_patches +
                  [plt.Line2D([0], [0], color=_K, lw=1.2, ls=":",
                               label=f"max ({config.ldpc_max_iterations})"),
                   plt.Line2D([0], [0], color=_O, lw=1.2, ls="--",
                               label=f"mean ({np.mean(iters):.1f})")],
                  fontsize=8)
    else:
        _missing(ax, "LDPC statistics not available")
        ax.set_title("(b) LDPC per-block iterations")

    return fig
