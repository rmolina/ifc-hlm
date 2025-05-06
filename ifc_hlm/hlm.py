"""Base HLM class."""

import logging as log
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import scipy.integrate as spi
import yaml
from numpy.typing import NDArray
from tqdm.auto import tqdm


@dataclass
class ModelParameters:
    """Base class for model parameters."""


@dataclass
class NodeParameters:
    """Base class for node parameters."""


@dataclass
class ModelForcings:
    """Base class for forcings."""


@dataclass
class ModelStates:
    """Base class for model states."""

    @classmethod
    @abstractmethod
    def init(cls, arr: NDArray[np.float64]):
        """Init an instance from an array."""

    @abstractmethod
    def from_array(self, arr: NDArray[np.float64]):
        """Update an instance from an array."""

    @abstractmethod
    def to_array(self) -> NDArray[np.float64]:
        """Convert the instance to an array."""


@dataclass
class Hlm(ABC):
    """BaseModel class."""

    config: dict = field(default_factory=dict)
    parameters: ModelParameters = field(default_factory=ModelParameters)
    forcings_data: ModelForcings = field(
        default_factory=ModelForcings
    )  # TODO: load into each node?
    initial_values: ModelStates = field(
        default_factory=ModelStates
    )  # TODO: load into each node?
    current_forcings: np.ndarray = field(default_factory=lambda: np.array([]))
    current_states: np.ndarray = field(default_factory=lambda: np.array([]))
    network: nx.DiGraph = field(default_factory=nx.DiGraph)
    current_time = 0.0
    flow_partitions: dict = field(default_factory=dict)  # TODO: load into each node?

    def _load_model_parameters(self) -> None:
        """Update model parameters from YAML file."""
        if (
            isinstance(self.config, dict)
            and "parameters" in self.config
            and isinstance(self.config["parameters"], dict)
        ):
            for key, value in self.config["parameters"].items():
                if hasattr(self.parameters, key):
                    log.info(
                        "Updating `parameters.%s` (old: %s, new: %s)",
                        key,
                        getattr(self.parameters, key),
                        value,
                    )
                    setattr(self.parameters, key, value)
                else:
                    log.warning("Unexpected `parameters.%s` in YAML file", key)

    def _load_initial_values_from_config(self) -> None:
        """Update initial_values from YAML file."""
        if (
            isinstance(self.config, dict)
            and "initial_values" in self.config
            and isinstance(self.config["initial_values"], dict)
        ):
            for key, value in self.config["initial_values"].items():
                if hasattr(self.initial_values, key):
                    log.info(
                        "Updating `initial_values.%s` (old: %s, new: %s)",
                        key,
                        getattr(self.initial_values, key),
                        value,
                    )
                    setattr(self.initial_values, key, value)
                else:
                    log.warning("Unexpected `initial_values.%s` in YAML file", key)

    def _load_config(self, config_file: str) -> None:
        """Load self.config from a YAML file."""

        log.info("Config file: %s", config_file)

        try:
            with open(config_file, "r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file)
                log.debug(self.config)
        except (FileNotFoundError, PermissionError, yaml.YAMLError) as exc:
            log.critical("Error reading the config file: `%s`", exc)

    def _load_network_topology(self) -> None:
        """Create a DAG from a *.rvr file"""

        log.info("Network topology file: %s", self.config["rvr_file"])

        with open(self.config["rvr_file"], "r", encoding="utf-8") as file:
            lines = file.readlines()

        num_nodes = int(lines[0].strip())

        index = 1
        while index < len(lines):
            while index < len(lines) and lines[index].strip() == "":
                index += 1
            if index >= len(lines):
                break
            node_id = int(lines[index].strip())
            index += 1
            if index < len(lines) and lines[index].strip() != "":
                node_info = lines[index].strip().split()
                num_children = int(node_info[0])
                children = list(map(int, node_info[1 : num_children + 1]))
                assert num_children == len(children)
                index += 1
            self.network.add_node(node_id)
            for child_id in children:
                self.network.add_edge(node_id, child_id)

        if num_nodes != self.network.number_of_nodes():
            log.warning(
                "Number of nodes mismatch (expected: %s, actual: %s)",
                num_nodes,
                self.network.number_of_nodes(),
            )

        log.info("Loaded a %s", self.network)

    def _load_network_parameters(self) -> None:
        """Add node parameters to a DAG from a *.prm file"""

        log.info("Network parameters file: %s", self.config["prm_file"])

        try:
            with open(self.config["prm_file"], "r", encoding="utf-8") as file:
                lines = file.readlines()
        except (FileNotFoundError, PermissionError) as exc:
            log.critical(exc)

        try:
            num_nodes = int(lines[0].strip())

            if num_nodes != self.network.number_of_nodes():
                log.critical("Number of nodes mismatch")

            index = 1
            while index < len(lines):
                while index < len(lines) and lines[index].strip() == "":
                    index += 1
                if index >= len(lines):
                    break
                node_id = int(lines[index].strip())
                index += 1
                if index < len(lines) and lines[index].strip() != "":
                    node_params = list(map(float, lines[index].strip().split()))
                    self.network.nodes[node_id]["parameters"] = node_params
                    index += 1
        except (IndexError, ValueError) as exc:
            log.critical("Invalid file: %s", exc)

        log.info(
            "Loaded %d parameters per node into %d nodes", len(node_params), num_nodes
        )

    def _initialize_node_states(self) -> None:
        """Add initial_values to a DAG from the config file"""

        log.info("Initial values: %s", self.initial_values)

        for node_id in self.network.nodes:
            self.network.nodes[node_id][
                "current_states"
            ] = self.initial_values.to_array()

    def _initialize_time(self) -> None:
        """Load mandatory time fields and start the clock."""

        try:
            start_time: datetime = self.config["start_time"]
            end_time: datetime = self.config["end_time"]
            _ = self.config["time_step"]
        except KeyError as exc:
            log.critical("Config file is missing a required key: %s", exc)

        try:
            self.config["start_time"] = start_time.timestamp()
            log.info("start_time: %s (%s)", self.config["start_time"], start_time)
        except AttributeError:
            log.critical(
                "Wrong value for 'start_time' (expected format: YYYY-MM-DD hh:mm:ss)"
            )

        try:
            self.config["end_time"] = end_time.timestamp()
            log.info("end_time: %s (%s)", self.config["end_time"], end_time)
        except AttributeError:
            log.critical(
                "Wrong value for 'end_time' (expected format: YYYY-MM-DD hh:mm:ss)"
            )

        try:
            self.config["time_step"] = float(self.config["time_step"])
        except (TypeError, ValueError):
            log.critical("Wrong value for 'time_step' (expected type: float)")

        self.current_time = self.config["start_time"]  # start the clock

    def _get_forcing_filename(self, key: str) -> str:
        """Get the forcing file name from the config."""
        try:
            forcings = self.config["forcings"]
        except KeyError:
            log.critical("Expected a `forcings` key in the `config` dict")
            raise

        try:
            return forcings[key]
        except KeyError:
            log.critical("Expected a `%s` key in the `forcings` dict", key)
            raise
        except TypeError:
            log.critical(
                "Mismatched types for `forcings` (expected: dict, found: %s)",
                type(forcings).__name__,
            )
            raise

    def _read_csv(self, forcing_csv) -> pd.DataFrame:
        """Get the forcing file name from the config."""

        # Read the CSV file into a DataFrame
        try:
            forcing_df = pd.read_csv(
                forcing_csv,
                parse_dates=True,
            )
        except (FileNotFoundError, PermissionError) as e:
            log.critical(e)
            raise
        except pd.errors.EmptyDataError:
            log.critical("Found no columns to parse in `%s`", forcing_csv)

        # Set the "time" column as index
        try:
            forcing_df.set_index("time", inplace=True)
        except KeyError:
            log.critical("Forcing files must have a `time` column")
            raise

        # Convert the "index" column into timestamps
        try:
            forcing_df.index = pd.Index(
                [
                    dt.timestamp()
                    for dt in pd.to_datetime(forcing_df.index, format="ISO8601")
                ]
            )
        except ValueError:
            log.critical("Data in the `time` column must be in the ISO 8601 format")
            raise

        return forcing_df

    def _load_forcing_series(self) -> None:
        """Get forcing series for each node."""
        for key in self.forcings_data.__dict__:
            forcing_csv = self._get_forcing_filename(key)
            series = self._read_csv(forcing_csv)
            setattr(self.forcings_data, key, series)

    def get_input(self, variable: pd.DataFrame, node_id: int, t: float) -> float:
        """Get a variable value at a given time and node_id."""
        return np.array(variable[str(node_id)].asof(t)).item()

    def _get_current_forcings(self) -> None:
        """Get forcings at current time for each node."""
        forcings = {}
        for key in self.forcings_data.__dict__:
            forcing_data = getattr(self.forcings_data, key)
            timestamps = forcing_data.index

            # Check if self.current_time is in the index
            if self.current_time in timestamps:
                forcings[key] = forcing_data.loc[self.current_time]
            else:
                # Use searchsorted to find the nearest indices
                idx = timestamps.searchsorted(self.current_time)

                # Handle edge cases: current_time is outside the range of the data
                if idx == 0:  # current_time is before the first timestamp
                    forcings[key] = forcing_data.iloc[0]
                elif idx == len(timestamps):  # current_time is after the last timestamp
                    forcings[key] = forcing_data.iloc[-1]
                else:
                    # Get the two closest times: one before and one after
                    time_before = timestamps[idx - 1]
                    time_after = timestamps[idx]

                    # Get values at the two nearest times
                    val_before = forcing_data.loc[time_before]
                    val_after = forcing_data.loc[time_after]

                    # Perform linear interpolation between the two points (timestamps)
                    delta_time = self.current_time - timestamps[idx - 1]
                    delta_total = timestamps[idx] - timestamps[idx - 1]

                    # Linear interpolation formula
                    interpolated_value = val_before + (val_after - val_before) * (
                        delta_time / delta_total
                    )

                    forcings[key] = interpolated_value

        current_forcings = pd.DataFrame(forcings)
        current_forcings.index = current_forcings.index.astype(int)
        target_order = self.get_nodes_in_reverse_topological_order()
        df_reordered = current_forcings.loc[target_order]  # .reset_index(drop=True)
        current_forcings_array = df_reordered.to_numpy()

        if self.current_forcings.size == 0:
            self.current_forcings = current_forcings_array
        else:
            self.current_forcings[:] = current_forcings_array

    def _get_current_states(self) -> None:
        """Get states at current time for each node."""

        node_ids = self.get_nodes_in_reverse_topological_order()

        current_states = []

        for node_id in node_ids:
            node = self.network.nodes[node_id]
            current_states.append(node["current_states"])

        if self.current_states.size == 0:
            self.current_states = np.array(current_states)
        else:
            self.current_states[:] = np.array(current_states)

    def startup(self, config_file: str) -> None:
        """Initialize HLM."""
        self._load_config(config_file)
        self._load_model_parameters()
        self._load_network_topology()
        self._load_network_parameters()
        self._load_initial_values_from_config()
        self._initialize_node_states()
        self._get_current_states()
        self._calculate_additional_network_parameters()
        self._load_forcing_series()
        self._initialize_time()
        self._get_current_forcings()  # to be able to get forcings at time 0 before the first update

    @abstractmethod
    def equations(
        self, t: float, y: NDArray[np.float64], node_id: int
    ) -> NDArray[np.float64]:
        """Compute the updated derivatives."""

    @abstractmethod
    def _calculate_additional_network_parameters(self) -> None:
        """Compute model-specific node parameters."""

    def get_nodes_in_reverse_topological_order(self) -> list[int]:
        """Get a list of node_ids in reverse topological order."""
        return list(nx.topological_sort(self.network))[::-1]

    def _solve_ivp_node(self, node_id: int, target_time: float) -> None:
        """Runs solve_ivp() for a node, from self.current_time to target_time"""

        def equations_wrapper(t, y):
            """Wraps self.equations() to match the signature expected by the integrator."""
            return self.equations(t, y, node_id)

        t_eval = np.arange(
            self.current_time,
            target_time + self.config["time_step"],
            self.config["time_step"],
        )

        ode_result = spi.solve_ivp(
            equations_wrapper,
            t_span=[self.current_time, target_time],
            t_eval=t_eval,
            y0=self.network.nodes[node_id]["current_states"],
            method=self.config["ode_solver"]["method"],
            atol=float(self.config["ode_solver"]["atol"]),
            rtol=float(self.config["ode_solver"]["rtol"]),
            max_step=float(self.config["ode_solver"]["max_step"]),
            dense_output=True,
        )

        if not ode_result.success:
            log.warning("Node ID #%s: %s", node_id, ode_result.message)

        ode_result.y[ode_result.y < 1e-12] = 0.0  # OutputConstraints_Model256_Hdf5

        self.network.nodes[node_id]["ode_result"] = ode_result

        self._update_current_states(node_id)
        self._stack_results(node_id)

    def _update_current_states(self, node_id: int):
        """Set node.current_states."""
        node = self.network.nodes[node_id]
        node["current_states"] = node["ode_result"].y[:, -1]

    def _stack_results(self, node_id: int):
        """Stack ode_results for a node."""
        node = self.network.nodes[node_id]
        ode_result = node["ode_result"]

        if "ode_result_t" not in node:
            node["ode_result_t"] = ode_result.t
        else:
            node["ode_result_t"] = np.hstack(
                (node["ode_result_t"], ode_result.t),
            )

        if "ode_result_y" not in node:
            node["ode_result_y"] = ode_result.y
        else:
            node["ode_result_y"] = np.hstack((node["ode_result_y"], ode_result.y))

    def _run_until(self, end_time: float) -> None:
        """Advance model state until the given time."""
        node_ids = self.get_nodes_in_reverse_topological_order()

        dt = np.array(self.current_time).astype("datetime64[s]")
        for node_id in tqdm(node_ids, unit="node", desc=f"{dt}"):
            self._solve_ivp_node(node_id, end_time)

        self._get_current_states()
        self._get_current_forcings()  # to be able to get forcings at time 0 before the first update

        self.current_time = end_time

    def solve(self) -> None:
        """Advance model state until the end time."""
        self._run_until(self.config["end_time"])

    def get_children_solutions(
        self,
        node_id: int,
        t: float,
    ) -> list[tuple[NDArray[np.float64], float]]:
        """Get the children solutions and flow partitions for a node."""
        children = list(self.network.successors(node_id))

        children_solutions = []
        for child_id in children:
            child_ode_result = self.network.nodes[child_id]["ode_result"]

            interpolated = child_ode_result.sol(t)

            if child_id in self.flow_partitions:
                child_flow_partition = self.flow_partitions[child_id][node_id]
                log.info(
                    "Node #%d will get %g·Q from Node #%d",
                    node_id,
                    child_flow_partition,
                    child_id,
                )
            else:
                child_flow_partition = 1.0
            children_solutions.append((interpolated, child_flow_partition))
        return children_solutions
