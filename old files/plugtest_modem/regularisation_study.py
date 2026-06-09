"""
regularisation_study.py
=======================

Experiments for GF3 Interim Report 2: choosing the channel-estimation
regularisation parameter lambda in the JOSS-D modem.

It re-uses the modem in modem.py without modifying it. The only new piece is a
*parametric* version of the channel estimator that lets me dial the regularisation
strength (est_reg), the number of averaged chirps (M) and an explicit time-domain
truncation length (h_trunc). Everything else (sync, OFDM demod, MMSE equaliser) is
the unmodified modem.

Commands
--------
    python3 regularisation_study.py fig1        # chirp spectrum + example channel
    python3 regularisation_study.py sim         # MSE(lambda) + BER(lambda) in sim
    python3 regularisation_study.py averaging   # MSE(lambda) for M = 1,2,5,10
    python3 regularisation_study.py maketx      # write tx_known.wav to play in room
    python3 regularisation_study.py all         # everything except 'real'

    # one real recording only:
    python3 regularisation_study.py real --phone recorded_phone.wav
    python3 regularisation_study.py real --laptop recorded_laptop.wav

    # both sources on one graph (sim reference curve added automatically):
    python3 regularisation_study.py real --phone recorded_phone.wav \
                                         --laptop recorded_laptop.wav

Figures land in ./figs ; numbers land in ./figs/results.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from modem import AudioModem, _simulated_channel, EPS

FIGS = Path("figs"); FIGS.mkdir(exist_ok=True)
RESULTS = {}

# Fixed known message so I can compute a true BER on real recordings.
MESSAGE = ("GF3 group 9 regularisation study. " * 200).encode("utf-8")
NAME    = "regtest.txt"


# --------------------------------------------------------------------------
# Parametric estimator
# --------------------------------------------------------------------------
def estimate_channel_param(m, rx, sync_start, est_reg, repeats=None, h_trunc=None):
    M    = m.EST_WIN
    X    = np.fft.rfft(m.chirp, M)
    denom = np.abs(X) ** 2
    lam   = est_reg * denom.max()
    if repeats is None:
        repeats = m.CHIRP_REPEATS
    h_acc = np.zeros(m.N)
    for i in range(repeats):
        s = sync_start + i * m.chirp_step
        y = rx[s: s + M]
        if len(y) < M:
            y = np.pad(y, (0, M - len(y)))
        Hf = np.fft.rfft(y, M) * np.conj(X) / (denom + lam)
        hi = np.fft.irfft(Hf, M)[: m.N]
        h_acc += hi
    h = h_acc / repeats
    if h_trunc is not None and h_trunc < m.N:
        h2 = np.zeros_like(h)
        h2[:h_trunc] = h[:h_trunc]
        h = h2
    return h, np.fft.fft(h, m.N)


def expected_bits(m):
    payload = m.build_payload(MESSAGE, NAME)
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8))


def ber_for_H(m, rx, sync_start, H, exp_bits):
    data_start = sync_start + len(m.preamble) + m.RESERVED
    bits = m.ofdm_demodulate(rx[data_start:], H)
    L    = min(len(bits), len(exp_bits))
    return float(np.mean(bits[:L] != exp_bits[:L]))


def nmse_db(H_est, H_ref, bins):
    num = np.mean(np.abs(H_est[bins] - H_ref[bins]) ** 2)
    den = np.mean(np.abs(H_ref[bins]) ** 2) + EPS
    return float(10 * np.log10(num / den + EPS))


def delay_spread_samples(h, frac=0.99):
    e = np.cumsum(h ** 2)
    e /= e[-1] + EPS
    return int(np.searchsorted(e, frac))


def add_awgn(rx, snr_db, sync_start, seed=0):
    rng = np.random.default_rng(seed)
    p   = np.mean(rx[sync_start:] ** 2) + EPS
    return rx + rng.standard_normal(len(rx)) * np.sqrt(p / (10 ** (snr_db / 10)))


# --------------------------------------------------------------------------
# Figure 1: chirp power spectrum + example estimated channel
# --------------------------------------------------------------------------
def fig1():
    m  = AudioModem()
    M  = m.EST_WIN
    X  = np.fft.rfft(m.chirp, M)
    f  = np.fft.rfftfreq(M, 1 / m.FS) / 1000.0
    PX = np.abs(X) ** 2
    PX_db = 10 * np.log10(PX / PX.max() + EPS)

    tx = m.transmit(MESSAGE, NAME, "tx.wav")
    rx = _simulated_channel(tx, snr_db=25, rolloff=12_000, seed=3)
    s  = m.synchronise(rx)
    h, H = estimate_channel_param(m, rx, s, est_reg=m.EST_REG)
    fH   = np.fft.rfftfreq(m.N, 1 / m.FS) / 1000.0
    Hmag = 20 * np.log10(np.abs(np.fft.rfft(h)) + EPS)
    Hmag -= Hmag.max()

    ds = delay_spread_samples(h[:m.N], 0.99)
    RESULTS["example_delay_spread_samples"] = ds
    RESULTS["example_delay_spread_ms"]      = round(1000 * ds / m.FS, 2)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax[0].plot(f, PX_db, lw=1.0)
    ax[0].axvspan(4, 13, color="0.85", zorder=0)
    ax[0].set_xlim(0, 24); ax[0].set_ylim(-40, 2)
    ax[0].set_xlabel("frequency / kHz")
    ax[0].set_ylabel(r"$|X[k]|^2$ / dB (rel. max)")
    ax[0].set_title("transmitted chirp power spectrum")
    ax[0].text(8.5, -36, "data band\n4-13 kHz", ha="center", fontsize=8)

    ax[1].plot(fH, Hmag, lw=0.8, color="C1")
    ax[1].axvspan(4, 13, color="0.85", zorder=0)
    ax[1].set_xlim(0, 24); ax[1].set_ylim(-60, 2)
    ax[1].set_xlabel("frequency / kHz")
    ax[1].set_ylabel(r"$|\hat H[k]|$ / dB (rel. max)")
    ax[1].set_title("example estimated channel (sim)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_chirp_channel.png", dpi=150)
    plt.close(fig)
    print("wrote fig1_chirp_channel.png  | example delay spread:",
          RESULTS["example_delay_spread_ms"], "ms")


# --------------------------------------------------------------------------
# Figure 2: MSE(lambda) and BER(lambda) in simulation
# --------------------------------------------------------------------------
def sim_sweep(snr_db=12, seeds=range(8), trunc_loose=None, trunc_tight=400):
    m   = AudioModem()
    tx  = m.transmit(MESSAGE, NAME, "tx.wav")
    exp = expected_bits(m)
    regs = np.logspace(-6, 0.5, 22)

    rx_clean = _simulated_channel(tx, snr_db=200, rolloff=12_000, seed=11)
    s0 = m.synchronise(rx_clean)
    _, H_ref = estimate_channel_param(m, rx_clean, s0, est_reg=1e-6, repeats=10)

    mse_loose = np.zeros((len(seeds), len(regs)))
    mse_tight = np.zeros((len(seeds), len(regs)))
    ber_loose = np.zeros((len(seeds), len(regs)))
    for si, seed in enumerate(seeds):
        rx = _simulated_channel(tx, snr_db=snr_db, rolloff=12_000, seed=seed)
        s  = m.synchronise(rx)
        for ri, r in enumerate(regs):
            _, Hl = estimate_channel_param(m, rx, s, est_reg=r, h_trunc=trunc_loose)
            _, Ht = estimate_channel_param(m, rx, s, est_reg=r, h_trunc=trunc_tight)
            mse_loose[si, ri] = nmse_db(Hl, H_ref, m.data_bins)
            mse_tight[si, ri] = nmse_db(Ht, H_ref, m.data_bins)
            ber_loose[si, ri] = ber_for_H(m, rx, s, Hl, exp)
    return regs, mse_loose.mean(0), mse_tight.mean(0), ber_loose.mean(0), m


def fig2(snr_db=12):
    regs, mse_l, mse_t, ber_l, m = sim_sweep(snr_db=snr_db)
    default = m.EST_REG

    i_l = int(np.argmin(mse_l)); i_t = int(np.argmin(mse_t))
    RESULTS["sim_snr_db"]               = snr_db
    RESULTS["sim_best_reg_loose"]        = float(regs[i_l])
    RESULTS["sim_best_mse_loose_db"]     = round(float(mse_l[i_l]), 2)
    RESULTS["sim_best_reg_tight"]        = float(regs[i_t])
    RESULTS["sim_best_mse_tight_db"]     = round(float(mse_t[i_t]), 2)
    RESULTS["sim_mse_at_default_loose_db"] = round(
        float(mse_l[np.argmin(np.abs(regs - default))]), 2)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    ax[0].semilogx(regs, mse_l, "o-", ms=3, label="no time truncation (N=1024)")
    ax[0].semilogx(regs, mse_t, "s-", ms=3, label="truncated to 400 samples")
    ax[0].axvline(default, color="0.4", ls="--", lw=1)
    ax[0].text(default * 1.5, ax[0].get_ylim()[0] + 1,
               "default\n$10^{-3}$", fontsize=8, va="bottom")
    ax[0].set_xlabel(r"relative regularisation $\lambda / \max|X|^2$")
    ax[0].set_ylabel("normalised MSE of $\\hat H$ / dB")
    ax[0].set_title(f"channel-estimate error, SNR = {snr_db} dB")
    ax[0].legend(fontsize=8); ax[0].grid(True, which="both", alpha=0.3)

    ax[1].loglog(regs, ber_l + 1e-5, "o-", ms=3, color="C3")
    ax[1].axvline(default, color="0.4", ls="--", lw=1)
    ax[1].set_xlabel(r"relative regularisation $\lambda / \max|X|^2$")
    ax[1].set_ylabel("bit error rate")
    ax[1].set_title("end-to-end BER (no truncation)")
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_mse_ber_lambda.png", dpi=150)
    plt.close(fig)
    print("wrote fig2_mse_ber_lambda.png")
    print("   best lambda (loose):", RESULTS["sim_best_reg_loose"],
          " MSE", RESULTS["sim_best_mse_loose_db"], "dB")
    print("   best lambda (trunc):", RESULTS["sim_best_reg_tight"],
          " MSE", RESULTS["sim_best_mse_tight_db"], "dB")


# --------------------------------------------------------------------------
# Figure 3: MSE(lambda) for different numbers of averaged chirps
# --------------------------------------------------------------------------
def fig3(snr_db=8, seeds=range(8)):
    m   = AudioModem()
    tx  = m.transmit(MESSAGE, NAME, "tx.wav")
    rx_clean = _simulated_channel(tx, snr_db=200, rolloff=12_000, seed=11)
    s0 = m.synchronise(rx_clean)
    _, H_ref = estimate_channel_param(m, rx_clean, s0, est_reg=1e-6, repeats=10)
    regs = np.logspace(-6, 0.5, 22)
    Ms   = [1, 2, 5, 10]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    best = {}
    for M in Ms:
        mse = np.zeros((len(seeds), len(regs)))
        for si, seed in enumerate(seeds):
            rx = _simulated_channel(tx, snr_db=snr_db, rolloff=12_000, seed=seed)
            s  = m.synchronise(rx)
            for ri, r in enumerate(regs):
                _, H = estimate_channel_param(m, rx, s, est_reg=r, repeats=M)
                mse[si, ri] = nmse_db(H, H_ref, m.data_bins)
        mean = mse.mean(0)
        i = int(np.argmin(mean))
        best[M] = (float(regs[i]), round(float(mean[i]), 2))
        ax.semilogx(regs, mean, "o-", ms=3, label=f"M = {M} chirps")
    RESULTS["averaging_snr_db"] = snr_db
    RESULTS["averaging_best"]   = best
    ax.set_xlabel(r"relative regularisation $\lambda / \max|X|^2$")
    ax.set_ylabel("normalised MSE of $\\hat H$ / dB")
    ax.set_title(f"effect of averaging M chirps (SNR = {snr_db} dB)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_averaging.png", dpi=150)
    plt.close(fig)
    print("wrote fig3_averaging.png  | best (lambda, MSE) per M:", best)


# --------------------------------------------------------------------------
# Figure 4: real recordings vs simulation reference — BER(lambda)
# --------------------------------------------------------------------------
def _sim_ber_curve(m, regs, snr_db=12, seeds=range(6)):
    """BER(lambda) over the simulated channel, averaged over noise seeds.
    Used as the 'working system' reference curve in fig4."""
    tx  = m.transmit(MESSAGE, NAME, "tx.wav")
    exp = expected_bits(m)
    bers = np.zeros((len(seeds), len(regs)))
    for si, seed in enumerate(seeds):
        rx = _simulated_channel(tx, snr_db=snr_db, rolloff=12_000, seed=seed)
        s  = m.synchronise(rx)
        for ri, r in enumerate(regs):
            _, H = estimate_channel_param(m, rx, s, est_reg=r)
            bers[si, ri] = ber_for_H(m, rx, s, H, exp)
    return bers.mean(0)


def _sweep_one_recording(m, wav, label, regs, ax, color, out):
    """Sweep lambda for a single real recording (as-recorded only).
    Adds one curve to ax and fills out[label]."""
    rx0 = m._load_wav(wav, m.FS)
    s   = m.synchronise(rx0)
    exp = expected_bits(m)

    h_est, _ = estimate_channel_param(m, rx0, s, est_reg=m.EST_REG)
    ds = delay_spread_samples(h_est[:m.N], 0.99)
    out[label] = {
        "delay_spread_samples": ds,
        "delay_spread_ms":      round(1000 * ds / m.FS, 2),
        "snr_conditions":       {},
    }
    print(f"   [{label}]  delay spread: {out[label]['delay_spread_ms']} ms "
          f"({ds} samples)")

    bers = np.array([
        ber_for_H(m, rx0, s,
                  estimate_channel_param(m, rx0, s, est_reg=r)[1], exp)
        for r in regs
    ])
    ax.loglog(regs, bers + 1e-5, "o-", ms=3, color=color,
              label=f"{label} speaker (as recorded)")
    out[label]["snr_conditions"]["None"] = {
        "min_ber":        float(bers.min()),
        "ber_at_default": float(bers[np.argmin(np.abs(regs - m.EST_REG))]),
    }
    print(f"      as recorded: min BER {bers.min():.4f}  "
          f"(default lambda: {bers[np.argmin(np.abs(regs - m.EST_REG))]:.4f})")


def maketx():
    m = AudioModem()
    m.transmit(MESSAGE, NAME, "tx_known.wav")
    print("wrote tx_known.wav")
    print("payload bits:", len(expected_bits(m)))


def fig4_real(wavs, labels):
    m    = AudioModem()
    regs = np.logspace(-6, 0.5, 22)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    out = {}

    # simulation references: 12 dB down to -6 dB to show the 50% asymptote
    sim_snrs = [12, 6, 3, 0, -3, -6]
    greys    = np.linspace(0.15, 0.70, len(sim_snrs))   # dark -> light
    for snr, grey in zip(sim_snrs, greys):
        print("   [sim %+d dB]  computing ..." % snr)
        sb = _sim_ber_curve(m, regs, snr_db=snr)
        ax.semilogx(regs, sb * 100, "-", lw=1.3, color=str(grey),
                    label="sim, SNR = %+d dB" % snr)
        RESULTS["sim_ber_snr%d" % snr] = {
            "min_ber":        float(sb.min()),
            "ber_at_default": float(sb[np.argmin(np.abs(regs - m.EST_REG))]),
        }
        print("      min BER %.4f  default-lambda BER %.4f" % (
              sb.min(), sb[np.argmin(np.abs(regs - m.EST_REG))]))

    # 50% random-guessing reference
    ax.axhline(50, color="red", ls=":", lw=1.1, label="random guessing (50%)")

    # real recordings, one coloured curve each
    colors = ["C0", "C1", "C2"]
    for wav, lbl, col in zip(wavs, labels, colors):
        # inline so we can plot as percentage on linear y
        rx0 = m._load_wav(wav, m.FS)
        s   = m.synchronise(rx0)
        exp = expected_bits(m)
        h_est, _ = estimate_channel_param(m, rx0, s, est_reg=m.EST_REG)
        ds = delay_spread_samples(h_est[:m.N], 0.99)
        out[lbl] = {"delay_spread_samples": ds,
                    "delay_spread_ms": round(1000*ds/m.FS, 2),
                    "snr_conditions": {}}
        print("   [%s]  delay spread: %.2f ms (%d samples)" % (lbl, 1000*ds/m.FS, ds))
        bers = np.array([
            ber_for_H(m, rx0, s,
                      estimate_channel_param(m, rx0, s, est_reg=r)[1], exp)
            for r in regs
        ])
        nice = {"phone": "phone speaker", "laptop": "laptop speaker",
                "speaker": "ext. speaker"}
        display_lbl = nice.get(lbl, lbl) + " (as recorded)"
        ax.semilogx(regs, bers * 100, "o-", ms=3, color=col,
                    label=display_lbl)
        out[lbl]["snr_conditions"]["None"] = {
            "min_ber":        float(bers.min()),
            "ber_at_default": float(bers[np.argmin(np.abs(regs - m.EST_REG))]),
        }
        print("      as recorded: min BER %.4f  default-lambda BER %.4f" % (
              bers.min(), bers[np.argmin(np.abs(regs - m.EST_REG))]))

    RESULTS["real"] = out
    ax.axvline(m.EST_REG, color="0.55", ls="--", lw=0.8)

    # y-axis: linear percentage scale, explicit ticks
    ax.set_yticks([0, 5, 10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels(["0%","5%","10%","20%","30%","40%","50%","60%","70%"])
    ax.set_ylim(-1, 73)

    ax.set_xlabel(r"relative regularisation $\lambda\,/\,\max|X|^2$")
    ax.set_ylabel("bit error rate")
    ax.set_title("BER vs regularisation: simulation vs real recordings (LR5)")
    ax.legend(fontsize=7, loc="upper right", ncol=1)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_ber_real.png", dpi=150)
    plt.close(fig)
    print("wrote fig4_ber_real.png")

def dump():
    (FIGS / "results.json").write_text(json.dumps(RESULTS, indent=2))
    print("\nnumbers for the report ->", FIGS / "results.json")
    print(json.dumps(RESULTS, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fig1")
    sub.add_parser("sim")
    sub.add_parser("averaging")
    sub.add_parser("maketx")
    sub.add_parser("all")
    r = sub.add_parser("real")
    r.add_argument("--phone",  default=None,
                   help="WAV recorded via phone speaker -> laptop mic")
    r.add_argument("--laptop", default=None,
                   help="WAV recorded via laptop speaker -> laptop mic")
    r.add_argument("--speaker", default=None,
                   help="WAV recorded via external/dedicated speaker -> laptop mic")
    a = p.parse_args()

    if a.cmd == "fig1":
        fig1()
    elif a.cmd == "sim":
        fig1(); fig2()
    elif a.cmd == "averaging":
        fig3()
    elif a.cmd == "maketx":
        maketx()
    elif a.cmd == "real":
        wavs, labels = [], []
        if a.phone:
            wavs.append(a.phone);  labels.append("phone")
        if a.laptop:
            wavs.append(a.laptop); labels.append("laptop")
        if a.speaker:
            wavs.append(a.speaker); labels.append("speaker")
        if not wavs:
            p.error("real: pass at least one of --phone, --laptop, or --speaker")
        fig4_real(wavs, labels)
    elif a.cmd == "all":
        fig1(); fig2(); fig3()
    dump()


if __name__ == "__main__":
    main()