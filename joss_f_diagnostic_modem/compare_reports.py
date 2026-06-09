"""Compare standard-component fingerprints from two diagnostic JSON reports."""
from __future__ import annotations

from pathlib import Path
import argparse
import json


FINGERPRINT_KEYS = [
    "chirp_hash",
    "chirp_train_hash",
    "golay_a_hash",
    "golay_b_hash",
    "golay_waveform_hash",
    "known_pilot_file_sha256",
    "known_pilot_symbol_hash",
    "known_pilot_frequency_hash",
    "known_pilot_waveform_hash",
    "interleaver_hash",
    "zero_information_ldpc_codeword_hash",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configuration(report: dict) -> dict:
    return report.get("stages", {}).get("configuration", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()

    first = configuration(load(args.first))
    second = configuration(load(args.second))
    width = max(len(key) for key in FINGERPRINT_KEYS)
    mismatch = False
    for key in FINGERPRINT_KEYS:
        a = first.get(key)
        b = second.get(key)
        if a is None and b is None:
            continue
        equal = a == b
        mismatch |= not equal
        print(f"{key:<{width}} : {'MATCH' if equal else 'MISMATCH'}")
        if not equal:
            print(f"  first : {a}")
            print(f"  second: {b}")
    raise SystemExit(1 if mismatch else 0)


if __name__ == "__main__":
    main()
