# GF3 synchronisation + channel-estimation lab

Measurement harness for testing synchronisation and channel estimation in the
GF3 audio modem project.

This is **not** a complete file-transmitting modem. It is a diagnostic and
measurement tool intended to:

- compare synchronisation methods;
- estimate the effective speaker-room-microphone channel;
- generate report-ready figures quickly;
- test robustness under echoes, attenuation, and noise;
- motivate later OFDM/DMT equalisation work.

Current features:

- matched-filter synchronisation using either:
  - logarithmic chirps;
  - stepped-frequency templates;
- repeated-half self-similarity metric (Schmidl-Cox-style timing idea);
- frequency-domain and sampled least-squares channel estimation;
- impulse-response estimation;
- channel magnitude/phase response plots;
- simulated OFDM/DMT one-tap equalisation demonstration;
- simulated acoustic-channel demo mode;
- optional live play-record helper.

---

# Install

Required:

```bash
python -m pip install numpy scipy matplotlib