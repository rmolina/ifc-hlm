"""Concrete implementation of a vectorized HLM (Model 252)."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .bmi_model import BmiModel


@dataclass
class Globals:
    v_0: float = field(default=0.33, metadata={"units": "m s-1"})  # [m s-1]
    lambda_1: float = field(default=0.2, metadata={"units": "1"})  # [1]
    lambda_2: float = field(default=-0.1, metadata={"units": "1"})  # [1]
    v_h: float = field(default=0.02, metadata={"units": "m s-1"})  # [m s-1]
    k_3: float = field(default=2.0425e-6 / 60.0, metadata={"units": "s-1"})  # [s-1]
    k_i_factor: float = field(default=0.02, metadata={"units": "1"})  # [1]
    h_b: float = field(default=0.5, metadata={"units": "m"})  # [m]
    s_l: float = field(default=0.1, metadata={"units": "m"})  # [m]
    a: float = field(default=0.0, metadata={"units": "1"})  # [1]
    b: float = field(default=99.0, metadata={"units": "1"})  # [1]
    exponent: float = field(default=3.0, metadata={"units": "1"})  # [1]
    v_b: float = field(default=0.75, metadata={"units": "m s-1"})  # [m s-1]


@dataclass
class Forcings:
    pcp: NDArray[np.floating] = field(metadata={"units": "m s-1", "location": "face"})
    pet: NDArray[np.floating] = field(metadata={"units": "m s-1", "location": "face"})


@dataclass
class States:
    q: NDArray[np.floating] = field(metadata={"units": "m3 s-1", "location": "node"})
    s_p: NDArray[np.floating] = field(metadata={"units": "m", "location": "face"})
    s_t: NDArray[np.floating] = field(metadata={"units": "m", "location": "face"})
    s_s: NDArray[np.floating] = field(metadata={"units": "m", "location": "face"})


@dataclass
class Derivatives:
    q: NDArray[np.floating] = field(metadata={"units": "m3 s-2"})
    s_p: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    s_t: NDArray[np.floating] = field(metadata={"units": "m s-1"})
    s_s: NDArray[np.floating] = field(metadata={"units": "m s-1"})


@dataclass
class Parameters:
    a_i: NDArray[np.floating] = field(metadata={"units": "m2"})
    l_i: NDArray[np.floating] = field(metadata={"units": "m"})
    a_h: NDArray[np.floating] = field(metadata={"units": "m2"})

    # These are computed in compute_extra_parameters() and filled in later
    # Note: use field(init=False) to indicate they are not passed to the constructor
    invtau: NDArray[np.floating] = field(init=False, metadata={"units": "s-1"})  # s-1
    k_2: NDArray[np.floating] = field(init=False, metadata={"units": "s-1"})  # s-1
    k_i: NDArray[np.floating] = field(init=False, metadata={"units": "s-1"})  # s-1


@dataclass
class Externals:
    q_in: NDArray[np.floating] = field(metadata={"units": "m3 s-1", "location": "node"})


@dataclass
class Fluxes:
    # Evaporation [m s-1]
    e_p: NDArray[np.floating] = field(metadata={"units": "m s-1", "location": "node"})
    e_t: NDArray[np.floating] = field(metadata={"units": "m s-1", "location": "node"})
    e_s: NDArray[np.floating] = field(metadata={"units": "m s-1", "location": "node"})

    # Vertical transfers [m s-1]
    q_pl: NDArray[np.floating] = field(
        metadata={
            "description": "from ponded to link",
            "units": "m s-1",
            "location": "node",
        }
    )
    q_pt: NDArray[np.floating] = field(
        metadata={
            "description": "from ponded to link",
            "units": "m s-1",
            "location": "node",
        }
    )
    q_ts: NDArray[np.floating] = field(
        metadata={
            "description": "from topsoil to subsurface",
            "units": "m s-1",
            "location": "node",
        }
    )
    q_sl: NDArray[np.floating] = field(
        metadata={
            "description": "from subsurface to link",
            "units": "m s-1",
            "location": "node",
        }
    )

    # Lateral discharge driver [m3 s-1]
    discharge: NDArray[np.floating] = field(
        metadata={"units": "m3 s-1", "location": "node"}
    )


class Model252(
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
        # === Evaporation fluxes ===
        S_R = 1.0  # m (Reference length)
        self.fluxes.e_p[:] = 0.0
        self.fluxes.e_t[:] = 0.0
        self.fluxes.e_s[:] = 0.0

        corr = (
            self.outputs.s_p / S_R
            + self.outputs.s_t / self.globals.s_l
            + self.outputs.s_s / (self.globals.h_b - self.globals.s_l)
        )

        mask = (self.inputs.pet > 0.0) & (corr > 1e-12)
        self.fluxes.e_p[mask] = (self.outputs.s_p[mask] / S_R) * (
            self.inputs.pet[mask] / corr[mask]
        )
        self.fluxes.e_t[mask] = (self.outputs.s_t[mask] / self.globals.s_l) * (
            self.inputs.pet[mask] / corr[mask]
        )
        self.fluxes.e_s[mask] = (
            self.outputs.s_s[mask] / (self.globals.h_b - self.globals.s_l)
        ) * (self.inputs.pet[mask] / corr[mask])

        # === Storage fluxes ===
        sat_def = 1.0 - self.outputs.s_t / self.globals.s_l
        pow_term = np.zeros_like(sat_def)
        mask = sat_def > 0.0
        pow_term[mask] = np.power(sat_def[mask], self.globals.exponent)
        k_t = self.parameters.k_2 * (self.globals.a + self.globals.b * pow_term)

        self.fluxes.q_pl[:] = self.parameters.k_2 * self.outputs.s_p
        self.fluxes.q_pt[:] = k_t * self.outputs.s_p
        self.fluxes.q_ts[:] = self.parameters.k_i * self.outputs.s_t
        self.fluxes.q_sl[:] = self.globals.k_3 * self.outputs.s_s

        # === Discharge ===
        discharge = (
            -self.outputs.q
            + self.parameters.a_h * (self.fluxes.q_pl + self.fluxes.q_sl)
            + self.externals.q_in
        )
        if self.globals.lambda_1 < 1.0:
            discharge = np.where(self.outputs.q < 0.0, 0.0, discharge)

        # Store in Fluxes dataclass
        self.fluxes.discharge[:] = discharge

    def compute_derivatives(self) -> None:
        """Compute derivatives using current fluxes and forcings."""
        # Unpack for convenience
        Q_R = 1.0  # reference discharge

        # Store all in derivatives dataclass
        self.derivatives.q[:] = (
            self.parameters.invtau
            * np.power(np.maximum(self.outputs.q / Q_R, 1e-10), self.globals.lambda_1)
            * self.fluxes.discharge
        )
        self.derivatives.s_p[:] = (
            self.inputs.pcp - self.fluxes.q_pl - self.fluxes.q_pt - self.fluxes.e_p
        )
        self.derivatives.s_t[:] = self.fluxes.q_pt - self.fluxes.q_ts - self.fluxes.e_t
        self.derivatives.s_s[:] = self.fluxes.q_ts - self.fluxes.q_sl - self.fluxes.e_s

    # def equations(self) -> None:
    #     """Compute fluxes and derivatives."""
    #     self.compute_fluxes()
    #     self.compute_derivatives()

    def compute_external_fluxes(self) -> None:
        """Compute external fluxes and store in self.externals."""
        q_src = np.maximum(self.outputs.q[self.edges_src], 1e-6)  # Clamp sources
        q_in = np.bincount(
            self.edges_dst,
            weights=q_src,
            minlength=self.num_nodes,
        ).astype(np.float64)
        self.externals.q_in[:] = q_in

    def compute_extra_parameters(self) -> None:
        """Compute derived parameters and store in self.parameters."""

        A_R = 1.0e6  # m2 (Reference area)

        self.parameters.invtau[:] = (
            self.globals.v_0
            * np.power(self.parameters.a_i / A_R, self.globals.lambda_2)
            / ((1.0 - self.globals.lambda_1) * self.parameters.l_i)
        )  # s-1

        self.parameters.k_2[:] = (
            self.globals.v_h * self.parameters.l_i / self.parameters.a_h
        )  # s-1

        self.parameters.k_i[:] = self.parameters.k_2 * self.globals.k_i_factor  # [s-1]
