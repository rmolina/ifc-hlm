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
    e_p: NDArray[np.floating]
    e_t: NDArray[np.floating]
    e_s: NDArray[np.floating]

    # Vertical transfers [m s-1]
    q_pl: NDArray[np.floating]
    q_pt: NDArray[np.floating]
    q_ts: NDArray[np.floating]
    q_sl: NDArray[np.floating]

    # Lateral discharge driver [m3 s-1]
    discharge: NDArray[np.floating]


class Model252(BmiModel[Forcings, States, Parameters, Globals, Derivatives, Externals, Fluxes]):

    InputsType = Forcings
    OutputsType = States
    ParametersType = Parameters
    GlobalsType = Globals
    DerivativesType = Derivatives
    ExternalsType = Externals
    FluxesType = Fluxes

    def calculate_fluxes(self) -> None:
        """Compute all fluxes and store in self.fluxes."""
        # === Evaporation fluxes ===
        S_R = 1.0  # m (Reference length)
        e_p = np.zeros(shape=self.num_nodes, dtype=np.float64)
        e_t = np.zeros(shape=self.num_nodes, dtype=np.float64)
        e_s = np.zeros(shape=self.num_nodes, dtype=np.float64)

        corr = (
            self.outputs.s_p / S_R
            + self.outputs.s_t / self.globals.s_l
            + self.outputs.s_s / (self.globals.h_b - self.globals.s_l)
        )

        mask = (self.inputs.pet > 0.0) & (corr > 1e-12)
        e_p[mask] = (self.outputs.s_p[mask] / S_R) * (self.inputs.pet[mask] / corr[mask])
        e_t[mask] = (self.outputs.s_t[mask] / self.globals.s_l) * (self.inputs.pet[mask] / corr[mask])
        e_s[mask] = (self.outputs.s_s[mask] / (self.globals.h_b - self.globals.s_l)) * (
            self.inputs.pet[mask] / corr[mask]
        )

        # === Storage fluxes ===
        sat_def = 1.0 - self.outputs.s_t / self.globals.s_l
        pow_term = np.zeros_like(sat_def)
        mask = sat_def > 0.0
        pow_term[mask] = np.power(sat_def[mask], self.globals.exponent)
        k_t = self.parameters.k_2 * (self.globals.a + self.globals.b * pow_term)

        q_pl = self.parameters.k_2 * self.outputs.s_p
        q_pt = k_t * self.outputs.s_p
        q_ts = self.parameters.k_i * self.outputs.s_t
        q_sl = self.globals.k_3 * self.outputs.s_s

        discharge = -self.outputs.q + self.parameters.a_h * (q_pl + q_sl) + self.externals.q_in
        if self.globals.lambda_1 < 1.0:
            discharge = np.where(self.outputs.q < 0.0, 0.0, discharge)

        # Store all in Fluxes dataclass
        self.fluxes.e_p[:] = e_p
        self.fluxes.e_t[:] = e_t
        self.fluxes.e_s[:] = e_s
        self.fluxes.q_pl[:] = q_pl
        self.fluxes.q_pt[:] = q_pt
        self.fluxes.q_ts[:] = q_ts
        self.fluxes.q_sl[:] = q_sl
        self.fluxes.discharge[:] = discharge

    def calculate_derivatives(self) -> None:
        """Compute derivatives using current fluxes and forcings."""
        # Unpack for convenience
        Q_R = 1.0  # reference discharge

        deriv_discharge = self.parameters.invtau * np.power(np.maximum(self.outputs.q / Q_R, 1e-10), self.globals.lambda_1) * self.fluxes.discharge
        deriv_ponded = self.inputs.pcp - self.fluxes.q_pl - self.fluxes.q_pt - self.fluxes.e_p
        deriv_topsoil = self.fluxes.q_pt - self.fluxes.q_ts - self.fluxes.e_t
        deriv_subsurface = self.fluxes.q_ts - self.fluxes.q_sl - self.fluxes.e_s

        self.derivatives.q[:] = deriv_discharge
        self.derivatives.s_p[:] = deriv_ponded
        self.derivatives.s_t[:] = deriv_topsoil
        self.derivatives.s_s[:] = deriv_subsurface

    def equations(self) -> None:
        """Compute fluxes and derivatives."""
        self.calculate_fluxes()
        self.calculate_derivatives()
        
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
