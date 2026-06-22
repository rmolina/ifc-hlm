"""Base model class for vectorized HLM implementation."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import asdict, fields, is_dataclass
from typing import Generic, Type, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.rk import dop853_coefficients

from .config import Config

INTEGRATION_DTYPE = np.float64


I = TypeVar("I")  # Inputs
O = TypeVar("O")  # Outputs
P = TypeVar("P")  # Parameters
G = TypeVar("G")  # Globals
D = TypeVar("D")  # Derivaitives
E = TypeVar("E")  # Externals
F = TypeVar("F")  # Fluxes


class BaseModel(ABC, Generic[I, O, P, G, D, E, F]):

    InputsType: Type[I]
    OutputsType: Type[O]
    ParametersType: Type[P]
    GlobalsType: Type[G]
    DerivativesType: Type[D]
    ExternalsType: Type[E]
    FluxesType: Type[F]

    inputs: I
    outputs: O
    parameters: P
    globals: G
    derivatives: D
    externals: E
    fluxes: F

    config: Config
    current_time: np.datetime64

    node_ids: NDArray[np.int_]
    node_to_idx: dict[int, int]
    edges_src: NDArray[np.int_]
    edges_dst: NDArray[np.int_]

    initialized: bool = False

    @property
    def input_names(self) -> tuple[str, ...]:
        assert is_dataclass(self.InputsType)
        return tuple(f.name for f in fields(self.InputsType))

    @property
    def input_units(self) -> tuple[str, ...]:
        assert is_dataclass(self.InputsType)
        return tuple(f.metadata.get("units", "") for f in fields(self.InputsType))

    @property
    def output_names(self) -> tuple[str, ...]:
        assert is_dataclass(self.OutputsType)
        return tuple(f.name for f in fields(self.OutputsType))

    @property
    def output_units(self) -> tuple[str, ...]:
        assert is_dataclass(self.OutputsType)
        return tuple(f.metadata.get("units", "") for f in fields(self.OutputsType))

    def __repr__(self) -> str:
        if not self.initialized:
            return f"{self.__class__.__name__} (uninitialized)"

        return f"{self.__class__.__name__} ({self.num_nodes} nodes, {len(self.edges_dst)} edges)"

    def _load_config_file(self, config_file: str) -> None:
        self.config = Config(config_file)

        if self.config.start_time >= self.config.end_time:
            raise ValueError("`start_time` must be strictly less than `end_time`")

        duration = self.config.end_time - self.config.start_time
        if duration % self.config.time_step != 0:
            raise ValueError(
                "total simulation duration must be an integer multiple of `time_step`"
            )

    def _initialize_arrays(self) -> None:

        assert (
            is_dataclass(self.InputsType)
            and is_dataclass(self.OutputsType)
            and is_dataclass(self.DerivativesType)
            and is_dataclass(self.ExternalsType)
            and is_dataclass(self.FluxesType)
        )

        # Build inputs
        self.inputs = self.InputsType(
            **{f.name: np.full(self.num_nodes, np.nan) for f in fields(self.InputsType)}
        )

        # Build fluxes
        self.fluxes = self.FluxesType(
            **{f.name: np.full(self.num_nodes, np.nan) for f in fields(self.FluxesType)}
        )

        # Build outputs
        self.outputs = self.OutputsType(
            **{
                f.name: np.full(self.num_nodes, np.nan)
                for f in fields(self.OutputsType)
            }
        )

        # Build derivatives
        self.derivatives = self.DerivativesType(
            **{
                f.name: np.full(self.num_nodes, np.nan)
                for f in fields(self.DerivativesType)
            }
        )

        # Build externals
        self.externals = self.ExternalsType(
            **{
                f.name: np.full(self.num_nodes, np.nan)
                for f in fields(self.ExternalsType)
            }
        )

    def _assert_finite_outputs(self, stage: str) -> None:
        assert is_dataclass(self.outputs)

        invalid_summaries: list[str] = []
        for field_obj in fields(self.outputs):
            name = field_obj.name
            arr = getattr(self.outputs, name)
            invalid_mask = ~np.isfinite(arr)
            if not np.any(invalid_mask):
                continue

            invalid_idx = np.flatnonzero(invalid_mask)
            node_preview = self.node_ids[invalid_idx[:5]].tolist()
            invalid_summaries.append(
                f"{name}: {len(invalid_idx)} invalid values; sample node_ids={node_preview}"
            )

        if invalid_summaries:
            joined = "; ".join(invalid_summaries)
            raise FloatingPointError(
                f"Non-finite model outputs detected {stage}: {joined}"
            )

    def _initialize_globals(self) -> None:
        assert is_dataclass(self.GlobalsType)
        # print(f"Initializing globals for {self.__class__.__name__}...")

        # Start from defaults
        data = asdict(self.GlobalsType())

        # print(f"Config: {self.config}")

        # Overlay config overrides (if any)
        overrides = getattr(self.config, "globals", None)
        if overrides:
            data.update(overrides)

        # print(f"Overrides: {overrides}")
        # print(f"Globals after applying config overrides: {data}")

        for key, value in data.items():
            if data[key] is None:
                raise ValueError(
                    f"Global '{key}' is required but not set in config or defaults"
                )

        # Rebuild dataclass
        self.globals = self.GlobalsType(**data)

    def _initialize_parameters(self) -> None:
        assert is_dataclass(self.ParametersType)

        if not self.config.parameters_file.exists():
            raise ValueError(
                f"parameters_file does not exist: {self.config.parameters_file}"
            )

        with open(self.config.parameters_file, "r") as parameters_file:
            reader = csv.DictReader(parameters_file)
            rows = list(reader)

        if not rows:
            raise ValueError("parameters_file is empty")

        # Get column names (excluding node ID column)
        header = rows[0].keys()
        id_col = "node_id"
        var_cols = [col for col in header if col != id_col]

        # Get only init=True parameter names
        init_param_names = tuple(f.name for f in fields(self.ParametersType) if f.init)

        # Verify all init parameters are present
        missing_vars = set(init_param_names) - set(var_cols)
        if missing_vars:
            raise ValueError(f"parameters_file missing parameters: {missing_vars}")

        extra_vars = set(var_cols) - set(init_param_names)
        if extra_vars:
            raise ValueError(
                f"parameters_file contains unexpected parameters: {extra_vars}"
            )

        # Extract node IDs
        param_node_ids = np.array([int(row[id_col]) for row in rows])

        # Verify node IDs match exactly
        expected_nodes = set(self.node_ids)
        provided_nodes = set(param_node_ids)

        missing_nodes = expected_nodes - provided_nodes
        if missing_nodes:
            raise ValueError(f"parameters_file missing nodes: {sorted(missing_nodes)}")

        unexpected_nodes = provided_nodes - expected_nodes
        if unexpected_nodes:
            raise ValueError(
                f"parameters_file contains unexpected nodes: {sorted(unexpected_nodes)}"
            )

        # Initialize with init=True fields
        init_fields = {
            name: np.empty(self.num_nodes, dtype=np.float64)
            for name in init_param_names
        }

        self.parameters = self.ParametersType(**init_fields)

        # Load values by variable name
        for field_obj in fields(self.ParametersType):
            if not field_obj.init:
                continue

            var_name = field_obj.name
            arr = getattr(self.parameters, var_name)

            for row in rows:
                node_id = int(row[id_col])
                idx = self.node_to_idx[node_id]
                arr[idx] = float(row[var_name])

        # Allocate arrays for init=False fields
        for f in fields(self.ParametersType):
            if not f.init:
                setattr(
                    self.parameters, f.name, np.empty(self.num_nodes, dtype=np.float64)
                )

        # Now compute derived parameters
        self.compute_extra_parameters()

    def model_initialize(self, config_file: str) -> None:
        self._load_config_file(config_file)
        self._build_graph()
        self._initialize_arrays()
        self._initialize_globals()
        self._initialize_parameters()
        self._load_initial_conditions()
        self._assert_finite_outputs("after initialization")

        self.current_time = self.config.start_time
        self.initialized = True

    def euler_step(self) -> None:
        """Advance outputs by one time step using forward Euler."""
        dt = float(self.config.time_step / np.timedelta64(1, "s"))

        # Compute externals and derivatives
        self.compute_external_fluxes()
        self.compute_fluxes()
        self.compute_derivatives()

        # Update each output field
        for name in self.output_names:
            setattr(
                self.outputs,
                name,
                getattr(self.outputs, name) + dt * getattr(self.derivatives, name),
            )
        # Enforce non-negativity for all state variables
        for name in self.output_names:
            arr = getattr(self.outputs, name)
            arr[:] = np.maximum(arr, 0.0)

    def _outputs_to_vector(self) -> NDArray[np.float64]:
        return np.concatenate(
            [
                np.asarray(getattr(self.outputs, name), dtype=np.float64)
                for name in self.output_names
            ]
        )

    def _vector_to_outputs(self, values: NDArray[np.float64]) -> None:
        offset = 0
        for name in self.output_names:
            arr = getattr(self.outputs, name)
            width = arr.size
            arr[:] = values[offset : offset + width]
            offset += width

    def scipy_step(self, method: str) -> None:
        """Advance outputs by one time step using SciPy's adaptive solvers."""
        dt = float(self.config.time_step / np.timedelta64(1, "s"))
        y0 = self._outputs_to_vector()
        original_outputs = self.outputs

        def rhs(_: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            candidate = self._state_from_vector(y)
            self.outputs = candidate
            self.compute_external_fluxes()
            self.compute_fluxes()
            self.compute_derivatives()
            return np.concatenate(
                [
                    np.asarray(getattr(self.derivatives, name), dtype=np.float64)
                    for name in self.output_names
                ]
            )

        try:
            sol = solve_ivp(
                rhs,
                (0.0, dt),
                y0,
                method=method,
                t_eval=(dt,),
                vectorized=False,
                # atol=1e-2,
                # rtol=1e-2,
            )
            if not sol.success or sol.y.size == 0:
                raise RuntimeError(f"SciPy solver {method!r} failed: {sol.message}")

            self.outputs = original_outputs
            self._vector_to_outputs(sol.y[:, -1])
            for name in self.output_names:
                arr = getattr(self.outputs, name)
                arr[:] = np.maximum(arr, 0.0)
        finally:
            self.outputs = original_outputs

    def _state_from_vector(self, values: NDArray[np.float64]) -> O:
        cls: type[O] = type(self.outputs)
        offset = 0
        arrays: dict[str, NDArray[np.float64]] = {}
        for name in self.output_names:
            width = self.num_nodes
            arrays[name] = np.array(
                values[offset : offset + width], dtype=np.float64, copy=True
            )
            offset += width
        return cls(**arrays)

    def _explicit_rk_step(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> None:
        assert is_dataclass(self.outputs)
        assert is_dataclass(self.derivatives)

        dt = float(self.config.time_step / np.timedelta64(1, "s"))
        states = self.outputs
        state_fields = fields(states)
        stage_count = len(b)

        stage_derivatives = {
            field_obj.name: np.empty((stage_count, self.num_nodes), dtype=np.float64)
            for field_obj in state_fields
        }

        def evaluate_stage(stage_index: int, candidate_state: O) -> None:
            self.outputs = candidate_state
            self.compute_external_fluxes()
            self.compute_fluxes()
            self.compute_derivatives()
            for field_obj in state_fields:
                name = field_obj.name
                stage_derivatives[name][stage_index, :] = getattr(
                    self.derivatives, name
                )

        evaluate_stage(0, states)

        cls: type[O] = type(states)
        for stage_index in range(1, stage_count):
            candidate_state = cls(
                **{
                    field_obj.name: getattr(states, field_obj.name)
                    + dt
                    * np.sum(
                        a[stage_index, :stage_index, np.newaxis]
                        * stage_derivatives[field_obj.name][:stage_index],
                        axis=0,
                    )
                    for field_obj in state_fields
                }
            )
            evaluate_stage(stage_index, candidate_state)

        self.outputs = states
        for field_obj in state_fields:
            name = field_obj.name
            arr = getattr(states, name)
            arr[:] = arr + dt * np.sum(
                b[:, np.newaxis] * stage_derivatives[name],
                axis=0,
            )

        for name in self.output_names:
            arr = getattr(self.outputs, name)
            arr[:] = np.maximum(arr, 0.0)

    def rk4_step(self) -> None:
        a = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        b = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=np.float64)
        self._explicit_rk_step(a, b)

    def rk8_step(self) -> None:
        stage_count = dop853_coefficients.N_STAGES
        a = np.array(
            dop853_coefficients.A[:stage_count, :stage_count],
            dtype=np.float64,
            copy=True,
        )
        b = np.array(dop853_coefficients.B[:stage_count], dtype=np.float64, copy=True)
        self._explicit_rk_step(a, b)

    def rk45_step(self) -> None:
        self.scipy_step("RK45")

    def dop853_step(self) -> None:
        self.scipy_step("DOP853")

    def lsoda_step(self) -> None:
        self.scipy_step("LSODA")

    def model_update(self) -> None:
        if self.config.integrator == "euler":
            self.euler_step()
        elif self.config.integrator == "rk45":
            self.rk45_step()
        elif self.config.integrator == "dop853":
            self.dop853_step()
        elif self.config.integrator == "lsoda":
            self.lsoda_step()
        elif self.config.integrator == "rk8":
            self.rk8_step()
        else:
            self.rk4_step()
        self._assert_finite_outputs(
            f"after update at time {self.current_time.astype('datetime64[s]')}"
        )
        self.current_time += self.config.time_step

    def _build_graph(self) -> None:
        """Load edges from CSV and create index mappings."""

        with open(self.config.edges_file, "r") as f:
            reader = csv.DictReader(f)
            edges_list = [(int(row["src"]), int(row["dst"])) for row in reader]

        edges = np.array(edges_list, dtype=int)

        self.node_ids = np.unique(edges)
        self.num_nodes = len(self.node_ids)

        self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}

        self.edges_src = np.array([self.node_to_idx[u] for u in edges[:, 0]])
        self.edges_dst = np.array([self.node_to_idx[v] for v in edges[:, 1]])

        if self._has_cycle():
            raise ValueError("Graph contains cycles - not a valid DAG")

    def _load_initial_conditions(self) -> None:

        if not self.config.initials_file.exists():
            raise ValueError(
                f"initials_file does not exist: {self.config.initials_file}"
            )

        with open(self.config.initials_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("initials_file is empty")

        # Get column names (excluding node ID column)
        header = rows[0].keys()
        id_col = "node_id"  # or detect first column
        var_cols = [col for col in header if col != id_col]

        # Verify all output variables are present
        missing_vars = set(self.output_names) - set(var_cols)
        if missing_vars:
            raise ValueError(f"initials_file missing output variables: {missing_vars}")

        extra_vars = set(var_cols) - set(self.output_names)
        if extra_vars:
            raise ValueError(
                f"initials_file contains unexpected variables: {extra_vars}"
            )

        # Extract node IDs
        ic_node_ids = np.array([int(row[id_col]) for row in rows])

        # Verify node IDs match exactly
        expected_nodes = set(self.node_ids)
        provided_nodes = set(ic_node_ids)

        missing_nodes = expected_nodes - provided_nodes
        if missing_nodes:
            raise ValueError(f"initials_file missing nodes: {sorted(missing_nodes)}")

        unexpected_nodes = provided_nodes - expected_nodes
        if unexpected_nodes:
            raise ValueError(
                f"initials_file contains unexpected nodes: {sorted(unexpected_nodes)}"
            )

        # Load values by variable name
        assert is_dataclass(self.outputs)

        for field_obj in fields(self.outputs):
            var_name = field_obj.name
            arr = getattr(self.outputs, var_name)
            arr.fill(np.nan)

            for row in rows:
                node_id = int(row[id_col])
                idx = self.node_to_idx[node_id]
                arr[idx] = float(row[var_name])

    def _has_cycle(self) -> bool:
        """Check for cycles using DFS."""
        n = len(self.node_ids)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = np.zeros(n, dtype=int)

        # Build adjacency list
        adj: list[list[int]] = [[] for _ in range(n)]
        for src, dst in zip(self.edges_src, self.edges_dst):
            adj[int(src)].append(dst)

        def visit(node: int) -> bool:
            if color[node] == GRAY:
                return True  # Back edge found
            if color[node] == BLACK:
                return False

            color[node] = GRAY
            for neighbor in adj[node]:
                if visit(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in range(n):
            if color[node] == WHITE and visit(node):
                return True
        return False

    @abstractmethod
    def compute_extra_parameters(self) -> None:
        """Compute derived parameters and store in self.parameters."""
        ...

    @abstractmethod
    def compute_external_fluxes(self) -> None:
        """Compute external fluxes and store in self.externals."""
        ...

    @abstractmethod
    def compute_fluxes(self) -> None:
        """Compute fluxes and store in self.fluxes."""
        ...

    @abstractmethod
    def compute_derivatives(self) -> None:
        """Compute derivatives and store in self.derivatives."""
        ...
