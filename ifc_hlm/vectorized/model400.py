"""Concrete implementation of a vectorized HLM (Model 400)."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .bmi_model import BmiModel

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when numba is absent
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def _numba_compute_fluxes(
    rainfall: NDArray[np.float64],
    e_pot: NDArray[np.float64],
    temperature: NDArray[np.float64],
    frozen_ground: NDArray[np.bool_],
    h0: NDArray[np.float64],
    h1: NDArray[np.float64],
    h2: NDArray[np.float64],
    h3: NDArray[np.float64],
    h4: NDArray[np.float64],
    temp_thres: float,
    Hu: float,
    infiltration: float,
    percolation: float,
    alpha2: float,
    alpha3: float,
    alpha4: float,
    melt_factor: float,
    l_i: NDArray[np.float64],
    a_h: NDArray[np.float64],
    d0: NDArray[np.float64],
    out0: NDArray[np.float64],
    d1: NDArray[np.float64],
    out1: NDArray[np.float64],
    d2: NDArray[np.float64],
    out2: NDArray[np.float64],
    d3: NDArray[np.float64],
    out3: NDArray[np.float64],
    d4: NDArray[np.float64],
    out4: NDArray[np.float64],
) -> None:
    for i in range(rainfall.shape[0]):
        x1 = 0.0
        snowmelt = 0.0

        if temperature[i] >= temp_thres:
            snowmelt = h0[i]
            candidate = temperature[i] * melt_factor
            if candidate < snowmelt:
                snowmelt = candidate
            x1 = rainfall[i] + snowmelt
            d0[i] = 0.0
        else:
            d0[i] = rainfall[i]

        out0[i] = snowmelt

        x2 = x1 + h1[i] - Hu
        if x2 < 0.0:
            x2 = 0.0
        if frozen_ground[i]:
            x2 = x1

        d1[i] = x1 - x2
        out1[i] = e_pot[i]
        if h1[i] < out1[i]:
            out1[i] = h1[i]

        infiltration_rate = infiltration
        if frozen_ground[i]:
            infiltration_rate = 0.0

        x3 = x2
        if infiltration_rate < x3:
            x3 = infiltration_rate

        d2[i] = x2 - x3

        w = alpha2 * l_i[i] / a_h[i]
        if w > 1.0:
            w = 1.0
        out2[i] = h2[i] * w

        x4 = x3
        if percolation < x4:
            x4 = percolation

        d3[i] = x3 - x4
        out3[i] = h3[i] / alpha3

        d4[i] = x4
        out4[i] = h4[i] / alpha4


@njit(cache=True)
def _numba_compute_derivatives(
    q: NDArray[np.float64],
    h0: NDArray[np.float64],
    h1: NDArray[np.float64],
    h2: NDArray[np.float64],
    h3: NDArray[np.float64],
    h4: NDArray[np.float64],
    q_external: NDArray[np.float64],
    invtau: NDArray[np.float64],
    lambda_1: float,
    a_h: NDArray[np.float64],
    d0: NDArray[np.float64],
    out0: NDArray[np.float64],
    d1: NDArray[np.float64],
    out1: NDArray[np.float64],
    d2: NDArray[np.float64],
    out2: NDArray[np.float64],
    d3: NDArray[np.float64],
    out3: NDArray[np.float64],
    d4: NDArray[np.float64],
    out4: NDArray[np.float64],
    dq: NDArray[np.float64],
    dh0: NDArray[np.float64],
    dh1: NDArray[np.float64],
    dh2: NDArray[np.float64],
    dh3: NDArray[np.float64],
    dh4: NDArray[np.float64],
) -> None:
    for i in range(q.shape[0]):
        dh0[i] = d0[i] - out0[i]
        dh1[i] = d1[i] - out1[i]
        dh2[i] = d2[i] - out2[i]
        dh3[i] = d3[i] - out3[i]
        dh4[i] = d4[i] - out4[i]

        discharge = -q[i] + (out2[i] + out3[i] + out4[i]) * a_h[i] + q_external[i]
        if lambda_1 < 1.0 and q[i] < 0.0:
            discharge = 0.0

        q_pos = q[i]
        if q_pos < 0.0:
            q_pos = 0.0
        dq[i] = invtau[i] * (q_pos**lambda_1) * discharge


@dataclass
class Globals:

    v_0: float | None = field(
        metadata={
            "description": "reference velocity",
            "units": "m s-1",
        },
        default=None,
    )

    lambda_1: float | None = field(
        metadata={
            "description": "discharge exponent",
            "units": "1",
        },
        default=None,
    )
    lambda_2: float | None = field(
        metadata={
            "description": "drainage area exponent",
            "units": "1",
        },
        default=None,
    )
    Hu: float | None = field(
        metadata={
            "description": "max available storage in static tank",
            "units": "m",
        },
        default=None,
    )
    infiltration: float | None = field(
        metadata={
            "description": "infiltration rate",
            "units": "m s-1",
        },
        default=None,
    )
    percolation: float | None = field(
        metadata={
            "description": "percolation rate to aquifer",
            "units": "m s-1",
        },
        default=None,
    )
    alpha2: float | None = field(
        metadata={
            "description": "velocity",
            "units": "m s-1",
        },
        default=None,
    )
    alpha3: float | None = field(
        metadata={
            "description": "residence time",
            "units": "s",
        },
        default=None,
    )
    alpha4: float | None = field(
        metadata={
            "description": "residence time",
            "units": "s",
        },
        default=None,
    )
    melt_factor: float | None = field(
        metadata={
            "description": "melting factor for snowmelt",
            "units": "m s-1 degC-1",
        },
        default=None,
    )
    temp_thres: float | None = field(
        metadata={
            "description": "temperature threshold for snow/rain partitioning",
            "units": "degC",
        },
        default=None,
    )


@dataclass
class Forcings:
    rainfall: NDArray[np.floating] = field(
        metadata={
            "description": "rainfall intensity",
            "units": "m s-1",
        }
    )
    e_pot: NDArray[np.floating] = field(
        metadata={
            "description": "potential evaporation",
            "units": "m s-1",
            "location": "face",
        }
    )
    temperature: NDArray[np.floating] = field(
        metadata={
            "description": "air temperature",
            "units": "degC",
            "location": "face",
        }
    )
    frozen_ground: NDArray[np.bool] = field(
        metadata={
            "description": "boolean array indicating if the ground is frozen",
            "units": "1",
            "location": "face",
        }
    )


@dataclass
class States:
    q: NDArray[np.floating] = field(
        metadata={
            "description": "discharge",
            "units": "m3 s-1",
            "location": "node",
        }
    )
    h0: NDArray[np.floating] = field(
        metadata={
            "description": "snow storage",
            "units": "m",
            "location": "face",
        }
    )
    h1: NDArray[np.floating] = field(
        metadata={
            "description": "static storage",
            "units": "m",
            "location": "face",
        }
    )
    h2: NDArray[np.floating] = field(
        metadata={
            "description": "water in the hillslope surface",
            "units": "m",
            "location": "face",
        }
    )
    h3: NDArray[np.floating] = field(
        metadata={
            "description": "water in the gravitational storage in the upper part of soil",
            "units": "m",
            "location": "face",
        }
    )
    h4: NDArray[np.floating] = field(
        metadata={
            "description": "water in the aquifer storage",
            "units": "m",
            "location": "face",
        }
    )


@dataclass
class Derivatives:
    q: NDArray[np.floating] = field(
        metadata={
            "description": "change in discharge",
            "units": "m3 s-2",
        }
    )
    h0: NDArray[np.floating] = field(
        metadata={
            "description": "change in snow storage",
            "units": "m s-1",
        }
    )
    h1: NDArray[np.floating] = field(
        metadata={
            "description": "change in static storage",
            "units": "m s-1",
        }
    )
    h2: NDArray[np.floating] = field(
        metadata={
            "description": "change in water in the hillslope surface",
            "units": "m s-1",
        }
    )
    h3: NDArray[np.floating] = field(
        metadata={
            "description": "change in water in the gravitational storage in the upper part of soil",
            "units": "m s-1",
        }
    )
    h4: NDArray[np.floating] = field(
        metadata={
            "description": "change in water in the aquifer storage",
            "units": "m s-1",
        }
    )
    h5: NDArray[np.floating] = field(
        metadata={
            "description": "change in snow storage",
            "units": "m s-1",
        }
    )


@dataclass
class Parameters:
    l_i: NDArray[np.floating] = field(
        metadata={
            "description": "Length of the channel",
            "units": "m",
        }
    )
    a_i: NDArray[np.floating] = field(
        metadata={
            "description": "drainage area",
            "units": "m2",
        }
    )
    a_h: NDArray[np.floating] = field(
        metadata={
            "description": "area of the hillslope surface",
            "units": "m2",
        }
    )

    # These are computed in compute_extra_parameters() and filled in later
    # Note: use field(init=False) to indicate they are not passed to the constructor
    invtau: NDArray[np.floating] = field(init=False, metadata={"units": "s-1"})  # s-1


@dataclass
class Externals:
    q: NDArray[np.floating] = field(
        metadata={
            "description": "discharge",
            "units": "m3 s-1",
        }
    )


@dataclass
class Fluxes:
    d0: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    out0: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    d1: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    out1: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    d2: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    out2: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    d3: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    out3: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    d4: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    out4: NDArray[np.floating] = field(metadata={"units": "m s-1"})


class Model400(
    BmiModel[Forcings, States, Parameters, Globals, Derivatives, Externals, Fluxes]
):

    InputsType = Forcings
    OutputsType = States
    ParametersType = Parameters
    GlobalsType = Globals
    DerivativesType = Derivatives
    ExternalsType = Externals
    FluxesType = Fluxes

    def compute_fluxes(self) -> None:
        """Compute all fluxes and store in self.fluxes."""
        if NUMBA_AVAILABLE:
            self.inputs.frozen_ground = self.inputs.frozen_ground.astype(bool)
            _numba_compute_fluxes(
                self.inputs.rainfall,
                self.inputs.e_pot,
                self.inputs.temperature,
                self.inputs.frozen_ground,
                self.outputs.h0,
                self.outputs.h1,
                self.outputs.h2,
                self.outputs.h3,
                self.outputs.h4,
                float(self.globals.temp_thres),
                float(self.globals.Hu),
                float(self.globals.infiltration),
                float(self.globals.percolation),
                float(self.globals.alpha2),
                float(self.globals.alpha3),
                float(self.globals.alpha4),
                float(self.globals.melt_factor),
                self.parameters.l_i,
                self.parameters.a_h,
                self.fluxes.d0,
                self.fluxes.out0,
                self.fluxes.d1,
                self.fluxes.out1,
                self.fluxes.d2,
                self.fluxes.out2,
                self.fluxes.d3,
                self.fluxes.out3,
                self.fluxes.d4,
                self.fluxes.out4,
            )
            return

        self.legacy_compute_fluxes()

    def legacy_compute_fluxes(self) -> None:
        """Reference NumPy implementation of the flux calculations."""
        # snow storage

        x1 = np.zeros_like(self.inputs.temperature)

        mask = self.inputs.temperature >= self.globals.temp_thres

        snowfall = np.zeros_like(x1)
        # if temperature is below the threshold, all rainfall goes to snow storage
        snowfall[~mask] = self.inputs.rainfall[~mask]

        # Snowmelt is only active for nodes above the rain/snow threshold.
        snowmelt = np.zeros_like(x1)
        snowmelt[mask] = np.minimum(
            self.outputs.h0[mask],
            self.inputs.temperature[mask] * self.globals.melt_factor,
        )

        x1[mask] = self.inputs.rainfall[mask] + snowmelt[mask]

        self.fluxes.out0 = snowmelt
        self.fluxes.d0 = snowfall

        # == static storage ==

        # excedance flow to the second storage
        x2: NDArray[np.floating] = np.maximum(
            0.0, x1 + self.outputs.h1 - self.globals.Hu
        )

        # if ground is frozen, x1 goes directly to the surface
        # therefore nothing is diverted to static tank
        # print(f"{self.inputs.frozen_ground=}")
        self.inputs.frozen_ground = self.inputs.frozen_ground.astype(
            bool
        )  # Ensure boolean type

        x2[self.inputs.frozen_ground] = x1[self.inputs.frozen_ground]

        #  input to static tank
        self.fluxes.d1 = x1 - x2

        # evaporation from the static tank. it cannot evaporate more than h1 [m]
        self.fluxes.out1 = np.minimum(self.inputs.e_pot, self.outputs.h1)

        # float out1 = (e_pot > h1) ? e_pot : 0.0;

        # ===

        # surface storage tank

        infiltration = np.full_like(x2, self.globals.infiltration)
        infiltration[self.inputs.frozen_ground] = (
            0.0  # if ground is frozen, no infiltration occurs
        )
        x3: NDArray[np.floating] = np.minimum(x2, infiltration)

        # water that infiltrates to gravitational storage [m/min]
        self.fluxes.d2 = x2 - x3

        #  the input to surface storage
        # ASYNCH keeps the raw `km` / `km^2` magnitudes in this term.
        # We convert the SI ratio back to the reference scale so the surface
        # residence time matches the C implementation numerically.
        w = (
            self.globals.alpha2  # m s-1
            * self.parameters.l_i  # m
            / self.parameters.a_h  # m2
            # * 1.0e3
        )  # 1/s

        # print(f"{w=}")
        w = np.minimum(1.0, w)
        # water can take less than 1 min (dt) to leave surface

        self.fluxes.out2 = self.outputs.h2 * w  # direct runoff [m/s]

        #  SUBSURFACE storage
        percolation = np.full_like(x2, self.globals.percolation)
        x4: NDArray[np.floating] = np.minimum(x3, percolation)

        # water that percolates to aquifer storage [m/s]
        self.fluxes.d3 = x3 - x4

        #  input to gravitational storage [m/s]
        self.fluxes.out3 = self.outputs.h3 / self.globals.alpha3  # interflow [m/s]

        # == aquifer storage ==
        x5 = 0.0  # water loss to deeper aquifer [m]

        self.fluxes.d4 = x4 - x5

        self.fluxes.out4 = self.outputs.h4 / self.globals.alpha4

    def compute_derivatives(self) -> None:
        """Compute derivatives using current fluxes and forcings."""
        if NUMBA_AVAILABLE:
            _numba_compute_derivatives(
                self.outputs.q,
                self.outputs.h0,
                self.outputs.h1,
                self.outputs.h2,
                self.outputs.h3,
                self.outputs.h4,
                self.externals.q,
                self.parameters.invtau,
                float(self.globals.lambda_1),
                self.parameters.a_h,
                self.fluxes.d0,
                self.fluxes.out0,
                self.fluxes.d1,
                self.fluxes.out1,
                self.fluxes.d2,
                self.fluxes.out2,
                self.fluxes.d3,
                self.fluxes.out3,
                self.fluxes.d4,
                self.fluxes.out4,
                self.derivatives.q,
                self.derivatives.h0,
                self.derivatives.h1,
                self.derivatives.h2,
                self.derivatives.h3,
                self.derivatives.h4,
            )
            self.derivatives.h5[:] = 0.0
            return

        self.legacy_compute_derivatives()

    def legacy_compute_derivatives(self) -> None:
        """Reference NumPy implementation of the derivative calculations."""

        self.derivatives.h0 = self.fluxes.d0 - self.fluxes.out0
        self.derivatives.h1 = self.fluxes.d1 - self.fluxes.out1
        self.derivatives.h2 = self.fluxes.d2 - self.fluxes.out2
        self.derivatives.h3 = self.fluxes.d3 - self.fluxes.out3
        self.derivatives.h4 = self.fluxes.d4 - self.fluxes.out4

        discharge = (
            -self.outputs.q
            + (self.fluxes.out2 + self.fluxes.out3 + self.fluxes.out4)
            # * (self.parameters.a_h / 1.0e6)
            # rmolina
            * (self.parameters.a_h)
            + self.externals.q
        )  # m3 s-1

        Q_R = 1.0  # reference discharge [m3 s-1]
        if self.globals.lambda_1 < 1.0:
            discharge = np.where(self.outputs.q < 0.0, 0.0, discharge)

        self.derivatives.q[:] = (
            self.parameters.invtau
            * np.power(np.maximum(self.outputs.q / Q_R, 0.0), self.globals.lambda_1)
            * discharge
        )  # m3 s-2

    def compute_external_fluxes(self) -> None:
        """Compute external fluxes and store in self.externals."""
        q_src = np.maximum(self.outputs.q[self.edges_src], 0.0)
        q_in = np.bincount(
            self.edges_dst,
            weights=q_src,
            minlength=self.num_nodes,
        ).astype(np.float64)
        self.externals.q[:] = q_in

    def compute_extra_parameters(self) -> None:
        """Compute derived parameters and store in self.parameters."""

        # The Cedar parameter files store channel length in km and areas in km^2.
        # Convert once here so the rest of the vectorized implementation works in SI.
        self.parameters.a_i[:] *= 1.0e6
        self.parameters.l_i[:] *= 1.0e3
        self.parameters.a_h[:] *= 1.0e6

        A_R = 1.0e6  # m2 (Reference area)

        self.parameters.invtau[:] = (
            self.globals.v_0
            * np.power(self.parameters.a_i / A_R, self.globals.lambda_2)
            / ((1.0 - self.globals.lambda_1) * self.parameters.l_i)
        )  # s-1

    def initialize(self, config_file: str) -> None:
        # Call the parent class's initialize method to set up the model
        super().initialize(config_file)

        # Assert model-specific constraints
        assert self.globals.alpha4 >= 1, "alpha4 must be >= 1"
        assert self.globals.alpha3 >= 1, "alpha3 must be >= 1"
