# Receiver revision notes

## Robust receiver revision

- Fixed partial-chirp synchronisation assigning a late chirp to index zero.
- Added Golay-based absolute-origin disambiguation, including the 12,288-sample
  ambiguity caused by one complete Golay-repeat spacing.
- Replaced cubic interpolation with windowed-sinc SFO resampling.
- Added robust four-repeat Golay combination.
- Activated quality-gated pilot blending and temporal interpolation.
- Added null-bin noise estimation and revised soft confidence weights.
- Restricted decision-directed correction to a safe local basin.
- Added LDPC-guided QPSK rotation and timing rescue.
- Expanded diagnostics and regression tests.
