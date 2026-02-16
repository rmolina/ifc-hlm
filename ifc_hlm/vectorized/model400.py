"""Concrete implementation of a vectorized HLM (Model 400)."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .bmi_model import BmiModel


@dataclass
class Globals:

    v_0: float = field(
        metadata={
            "description": "reference velocity",
            "units": "m s-1",
        }
    )

    lambda_1: float = field(
        metadata={
            "description": "discharge exponent",
            "units": "1",
        }
    )
    lambda_2: float = field(
        metadata={
            "description": "drainage area exponent",
            "units": "1",
        }
    )
    Hu: float = field(
        metadata={
            "description": "max available storage in static tank",
            "units": "m",
        }
    )
    infiltration: float = field(
        metadata={
            "description": "infiltration rate",
            "units": "m s-1",
        }
    )
    percolation: float = field(
        metadata={
            "description": "percolation rate to aquifer",
            "units": "m s-1",
        }
    )
    alpha2: float = field(
        metadata={
            "description": "velocity",
            "units": "m s-1",
        }
    )
    alpha3: float = field(
        metadata={
            "description": "residence time",
            "units": "s",
        }
    )
    alpha4: float = field(
        metadata={
            "description": "residence time",
            "units": "s",  # TODO: this is in seconds, but the original model uses days. Make sure to convert appropriately in the code.
        }
    )
    melt_factor: float = field(
        metadata={
            "description": "mm day-1 degCelsius-1",
            "units": "mm day-1 degCelsius-1",
        }
    )
    temp_thres: float = field(
        metadata={
            "description": "degCelsius",
            "units": "degCelsius",
        }
    )


@dataclass
class Forcings:
    rainfall: NDArray[np.floating] = field(
        metadata={
            "units": "m s-1",
            "location": "face",
        }
    )
    e_pot: NDArray[np.floating] = field(
        metadata={
            "units": "m s-1",
            "location": "face",
        }
    )
    temperature: NDArray[np.floating] = field(
        metadata={
            "units": "degC",
            "location": "face",
        }
    )
    frozen_ground: NDArray[np.bool] = field(
        metadata={
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
            "location": "node",
        }
    )
    h0: NDArray[np.floating] = field(
        metadata={
            "description": "change in snow storage",
            "units": "m s-1",
            "location": "face",
        }
    )
    h1: NDArray[np.floating] = field(
        metadata={
            "description": "change in static storage",
            "units": "m s-1",
            "location": "face",
        }
    )
    h2: NDArray[np.floating] = field(
        metadata={
            "description": "change in water in the hillslope surface",
            "units": "m s-1",
            "location": "face",
        }
    )
    h3: NDArray[np.floating] = field(
        metadata={
            "description": "change in water in the gravitational storage in the upper part of soil",
            "units": "m s-1",
            "location": "face",
        }
    )
    h4: NDArray[np.floating] = field(
        metadata={
            "description": "change in water in the aquifer storage",
            "units": "m s-1",
            "location": "face",
        }
    )
    h5: NDArray[np.floating] = field(
        metadata={
            "description": "change in snow storage",
            "units": "m s-1",
            "location": "face",
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
            "location": "node",
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
        # snow storage

        x1 = np.zeros_like(self.inputs.temperature)

        mask = self.inputs.temperature >= self.globals.temp_thres

        snowfall = np.zeros_like(x1)
        snowfall[~mask] = self.inputs.rainfall[~mask]

        snowmelt = np.minimum(
            self.outputs.h0, self.inputs.temperature * self.globals.melt_factor
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
        x2[self.inputs.frozen_ground] = x1[self.inputs.frozen_ground]

        self.fluxes.d1 = x1 - x2

        #  input to static tank
        self.fluxes.out1 = np.minimum(self.inputs.e_pot, self.outputs.h1)

        # evaporation from the static tank. it cannot evaporate more than h1 [m]
        # float out1 = (e_pot > h1) ? e_pot : 0.0;

        # surface storage tank

        infiltration = np.full_like(x2, self.globals.infiltration)
        infiltration[self.inputs.frozen_ground] = (
            0.0  # if ground is frozen, no infiltration occurs
        )
        x3: NDArray[np.floating] = np.minimum(x2, infiltration)

        # water that infiltrates to gravitational storage [m/min]
        self.fluxes.d2 = x2 - x3

        #  the input to surface storage
        w = self.globals.alpha2 * self.parameters.l_i / self.parameters.a_h  # [1/s]

        w = min(1, w)
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

        self.derivatives.h0 = self.fluxes.d0 - self.fluxes.out0
        self.derivatives.h1 = self.fluxes.d1 - self.fluxes.out1
        self.derivatives.h2 = self.fluxes.d2 - self.fluxes.out2
        self.derivatives.h3 = self.fluxes.d3 - self.fluxes.out3
        self.derivatives.h4 = self.fluxes.d4 - self.fluxes.out4

        q = (
            -self.outputs.q
            + (self.fluxes.out2 + self.fluxes.out3 + self.fluxes.out4)
            * self.parameters.a_h
            + self.externals.q
        )  # m3 s-1

        Q_R = 1.0  # reference discharge [m3 s-1]
        self.derivatives.q = (
            self.parameters.invtau * pow(q / Q_R, self.globals.lambda_1) * q
        )  # m3 s-2

    def compute_external_fluxes(self) -> None:
        """Compute external fluxes and store in self.externals."""
        q_src = np.maximum(self.outputs.q[self.edges_src], 1e-6)  # Clamp sources
        q_in = np.bincount(
            self.edges_dst,
            weights=q_src,
            minlength=self.num_nodes,
        ).astype(np.float64)
        self.externals.q[:] = q_in

    def compute_extra_parameters(self) -> None:
        """Compute derived parameters and store in self.parameters."""

        # TODO: DO NOT REMOVE! This assertion is critical to ensure the model behaves correctly and does not produce unphysical results. The original model assumes that alpha4 is greater than or equal to 1, which corresponds to a residence time of at least 1 minute. If alpha4 is less than 1, it would imply a residence time of less than 1 minute, which could lead to division by zero or negative outflow in the computation of the aquifer outflow. By keeping this assertion, we ensure that the model remains stable and produces physically meaningful results.
        assert (
            self.globals.alpha4 >= 1
        ), "alpha4 must be >= 1 to avoid division by zero or negative outflow"

        assert (
            self.globals.alpha3 >= 1
        ), "alpha3 must be >= 1 to avoid division by zero or negative outflow"

        A_R = 1.0e6  # m2 (Reference area)

        self.parameters.invtau[:] = (
            self.globals.v_0
            * np.power(self.parameters.a_i / A_R, self.globals.lambda_2)
            / ((1.0 - self.globals.lambda_1) * self.parameters.l_i)
        )  # s-1
