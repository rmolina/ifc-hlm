"""Configuration module for HLM vectorized model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import toml


@dataclass(init=False)
class Config:
    start_time: np.datetime64  # seconds since epoch
    end_time: np.datetime64  # seconds since epoch
    time_step: np.timedelta64  # seconds
    integrator: Literal["euler", "rk4", "rk8", "rk45", "dop853", "lsoda"]

    edges_file: Path
    initials_file: Path
    parameters_file: Path

    globals: dict[str, float] | None = None # Global parameters (optional)

    time_units: Literal["s"] = "s"

    def __init__(self, path: str) -> None:
        data: dict[str, str] = toml.load(path)

        self.start_time = np.datetime64(data["start_time"], self.time_units)
        self.end_time = np.datetime64(data["end_time"], self.time_units)
        self.time_step = np.timedelta64(data["time_step"], self.time_units)
        self.integrator = data.get("integrator", "rk4")

        self.edges_file = Path(data["edges_file"])
        self.initials_file = Path(data["initials_file"])
        self.parameters_file = Path(data["parameters_file"])

        self.globals = data.get("globals", None) 

        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time must be strictly greater than start_time, got "
                f"start_time={self.start_time}, end_time={self.end_time}"
            )

        if self.time_step <= np.timedelta64(0, self.time_units):
            raise ValueError(
                "time_step must be positive, got " f"time_step={self.time_step}"
            )

        if self.integrator not in {"euler", "rk4", "rk8", "rk45", "dop853", "lsoda"}:
            raise ValueError(
                "integrator must be one of 'euler', 'rk4', 'rk8', 'rk45', "
                f"'dop853' or 'lsoda', got {self.integrator!r}"
            )

        if not self.edges_file.exists():
            raise ValueError(
                f"edges_file does not exist, got edges_file={self.edges_file}"
            )
