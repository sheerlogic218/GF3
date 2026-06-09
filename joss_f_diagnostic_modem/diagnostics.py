"""Non-invasive diagnostics helpers for the JOSS-F modem.

This module never changes the transmit or receive decisions.  It only records
values that have already been computed, saves arrays, and creates plots/reports.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import traceback

import numpy as np
from numpy.typing import NDArray


EPS = 1e-12


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    metadata = f"{contiguous.dtype.str}|{contiguous.shape}".encode("ascii")
    return sha256_bytes(metadata + contiguous.tobytes())


def rms(values: NDArray[Any]) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def normalised_correlation(a: NDArray[Any], b: NDArray[Any]) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    length = min(a.size, b.size)
    if length == 0:
        return 0.0
    a = a[:length]
    b = b[:length]
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float(abs(np.vdot(a, b)) / denominator)


def complex_gain_and_nmse(
    observed: NDArray[np.complex128],
    reference: NDArray[np.complex128],
) -> tuple[complex, float, float]:
    """Return least-squares gain, gain-corrected NMSE and coherence."""
    observed = np.asarray(observed, dtype=np.complex128).reshape(-1)
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    length = min(observed.size, reference.size)
    if length == 0:
        return 0j, float("inf"), 0.0
    observed = observed[:length]
    reference = reference[:length]
    denominator = np.vdot(reference, reference)
    gain = 0j if abs(denominator) < EPS else np.vdot(reference, observed) / denominator
    residual = observed - gain * reference
    nmse = float(np.vdot(residual, residual).real / (np.vdot(observed, observed).real + EPS))
    coherence = normalised_correlation(observed, reference)
    return complex(gain), nmse, coherence


def qpsk_evm(symbols: NDArray[np.complex128]) -> tuple[float, float]:
    """Return RMS and median EVM after nearest unnormalised QPSK slicing."""
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if symbols.size == 0:
        return float("inf"), float("inf")
    hard = np.where(symbols.real >= 0.0, 1.0, -1.0) + 1j * np.where(
        symbols.imag >= 0.0, 1.0, -1.0
    )
    error = np.abs(symbols - hard) / np.sqrt(2.0)
    return float(np.sqrt(np.mean(error**2))), float(np.median(error))


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return _json_value(value.tolist())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "minimum": float(np.nanmin(np.abs(value))) if value.size else 0.0,
            "maximum": float(np.nanmax(np.abs(value))) if value.size else 0.0,
        }
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "+inf"
        if np.isneginf(value):
            return "-inf"
    return value


class DiagnosticSession:
    """Collect and persist diagnostics even when decoding raises an exception."""

    def __init__(self, directory: str | Path, source: str | Path | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source = None if source is None else str(source)
        self.report: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "status": "running",
            "stages": {},
            "warnings": [],
            "events": [],
        }
        self.arrays: dict[str, NDArray[Any]] = {}
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        line = str(message)
        self.lines.append(line)
        print(f"[diag] {line}")

    def stage(self, name: str, **values: Any) -> None:
        current = self.report["stages"].setdefault(name, {})
        current.update(values)

    def event(self, name: str, **values: Any) -> None:
        self.report["events"].append({"name": name, **values})

    def warning(self, message: str) -> None:
        self.report["warnings"].append(str(message))
        self.log(f"WARNING: {message}")

    def array(self, name: str, values: NDArray[Any]) -> None:
        self.arrays[name] = np.asarray(values).copy()

    def rows(self, name: str, rows: list[dict[str, Any]]) -> None:
        self.tables[name] = rows

    def write_text(self, filename: str, text: str) -> Path:
        path = self.directory / filename
        path.write_text(text, encoding="utf-8")
        return path

    def save_figure(self, filename: str, figure: Any) -> Path:
        path = self.directory / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        return path

    def finalise(self, status: str, error: BaseException | None = None) -> Path:
        self.report["status"] = status
        if error is not None:
            self.report["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }

        if self.arrays:
            np.savez_compressed(self.directory / "diagnostic_arrays.npz", **self.arrays)

        for name, rows in self.tables.items():
            if not rows:
                continue
            fields: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            with (self.directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: _json_value(value) for key, value in row.items()})

        report_path = self.directory / "diagnostic_report.json"
        report_path.write_text(
            json.dumps(_json_value(self.report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.directory / "diagnostic_console.txt").write_text(
            "\n".join(self.lines) + ("\n" if self.lines else ""),
            encoding="utf-8",
        )
        return report_path
