"""Adds BMI functions to the base Hlm class."""

from dataclasses import dataclass, field

import numpy as np
from bmipy import Bmi
from numpy.typing import NDArray

from .hlm import Hlm


@dataclass
class BmiFields:
    """Required fields for every BMI-compliant model."""

    component_name: str = field(default_factory=str)
    input_var_names: tuple[str, ...] = field(default_factory=tuple)
    input_var_units: tuple[str, ...] = field(default_factory=tuple)
    output_var_names: tuple[str, ...] = field(default_factory=tuple)
    output_var_units: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class HlmBmi(Hlm, Bmi):  # pylint: disable=too-many-public-methods
    """Extend the Hlm base class to make it BMI-compliant."""

    bmi_fields: BmiFields = field(default_factory=BmiFields)

    # Model control functions

    def initialize(self, config_file: str) -> None:  # DONE
        """Perform startup tasks for the model."""
        self.startup(config_file)

    def update(self) -> None:  # DONE
        """Advance model state by one time step."""
        self._run_until(self.current_time + self.config["time_step"])

    def update_until(self, time: float) -> None:  # DONE
        """Advance model state until the given time."""
        self._run_until(time)

    def finalize(self) -> None:
        """Perform tear-down tasks for the model."""
        # TODO

    # Model information functions

    def get_component_name(self) -> str:  # DONE
        """Name of the component."""
        return self.bmi_fields.component_name

    def get_input_item_count(self) -> int:  # DONE
        """Count of a model's input variables."""
        return len(self.bmi_fields.input_var_names)

    def get_output_item_count(self) -> int:  # DONE
        """Count of a model's output variables."""
        return len(self.bmi_fields.output_var_names)

    def get_input_var_names(self) -> tuple[str, ...]:  # type: ignore
        # https://github.com/csdms/bmi-python/blob/b0ef8b640aee4b499c5d7a8b3b30d625c28a2194/src/bmipy/bmi.py#L110
        """Get the model's input variables."""
        return self.bmi_fields.input_var_names

    def get_output_var_names(self) -> tuple[str, ...]:  # type: ignore
        # https://github.com/csdms/bmi-python/blob/b0ef8b640aee4b499c5d7a8b3b30d625c28a2194/src/bmipy/bmi.py#L133
        """Get the model's output variables."""
        return self.bmi_fields.output_var_names

    # Variable information functions

    def get_var_grid(self, name: str) -> int:  # DONE
        """Get grid identifier for the given variable."""
        var_names = self.get_input_var_names() + self.get_output_var_names()
        for var_idx, var_name in enumerate(var_names):
            if var_name == name:
                return var_idx
        raise KeyError

    def get_var_type(self, name: str) -> str:  # DONE
        """Get data type of the given variable."""
        return str(self.get_value_ptr(name).dtype)

    def get_var_units(self, name: str) -> str:  # DONE
        """Get units of the given variable."""
        return (self.bmi_fields.input_var_units + self.bmi_fields.output_var_units)[
            self.get_var_grid(name)
        ]

    def get_var_itemsize(self, name: str) -> int:  # DONE
        """Get memory use for each array element in bytes."""
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_nbytes(self, name: str) -> int:  # DONE
        """Get size, in bytes, of the given variable."""
        return self.get_value_ptr(name).nbytes

    def get_var_location(self, name: str) -> str:
        """Get the grid element type that the a given variable is defined on."""
        return "node"  # FIXME: *node*: A point that has a coordinate pair or triplet

    # Time functions

    def get_current_time(self) -> float:
        """Return the current time of the model."""
        return self.current_time

    def get_start_time(self) -> float:
        """Start time of the model."""
        return self.config["start_time"]

    def get_end_time(self) -> float:
        """End time of the model."""
        return self.config["end_time"]

    def get_time_units(self) -> str:
        """Time units of the model."""
        return "s"

    def get_time_step(self) -> float:
        """Return the current time step of the model."""
        return self.config["time_step"]

    # Variable getter and setter functions

    def get_value(
        self, name: str, dest: NDArray[np.float64]
    ) -> NDArray[np.float64]:  # DONE
        """Get a copy of values of the given variable."""
        dest[:] = self.get_value_ptr(name).flatten()
        return dest

    def get_value_ptr(self, name: str) -> NDArray[np.float64]:  # DONE
        """Get a reference to values of the given variable."""

        for idx, var_name in enumerate(self.get_input_var_names()):
            if var_name == name:
                return self.current_forcings[:, idx]

        for idx, var_name in enumerate(self.get_output_var_names()):
            if var_name == name:
                return self.current_states[:, idx]

        raise KeyError

    def get_value_at_indices(
        self, name: str, dest: NDArray[np.float64], inds: NDArray[np.int_]
    ) -> NDArray[np.float64]:  # DONE
        """Get values at particular indices."""
        dest[:] = self.get_value_ptr(name)[inds]
        return dest

    def set_value(self, name: str, src: NDArray[np.float64]) -> None:  # DONE
        """Specify a new value for a model variable."""

        for idx, var_name in enumerate(self.get_input_var_names()):
            if var_name == name:
                self.current_forcings[:, idx] = src[:]
                return

        for idx, var_name in enumerate(self.get_output_var_names()):
            if var_name == name:
                self.current_states[:, idx] = src[:]
                return

        raise KeyError

    def set_value_at_indices(
        self, name: str, inds: NDArray[np.int_], src: NDArray[np.float64]
    ) -> None:
        """Specify a new value for a model variable at particular indices."""
        self.get_value_ptr(name)[inds] = src

    # # Grid information

    def get_grid_rank(self, grid: int) -> int:
        """Get number of dimensions of the computational grid."""
        raise NotImplementedError("`get_grid_rank` is not implemented")

    def get_grid_size(self, grid: int) -> int:
        """Get the total number of elements in the computational grid."""
        raise NotImplementedError("`get_grid_size` is not implemented")

    def get_grid_type(self, grid: int) -> str:
        """Get the grid type as a string."""
        return "unstructured"

    # Uniform rectilinear

    def get_grid_shape(self, grid: int, shape: NDArray[np.int_]) -> NDArray[np.int_]:
        """Get dimensions of the computational grid."""
        raise NotImplementedError("`get_grid_shape` is not implemented")

    def get_grid_spacing(
        self, grid: int, spacing: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Get distance between nodes of the computational grid."""
        raise NotImplementedError("`get_grid_spacing` is not implemented")

    def get_grid_origin(
        self, grid: int, origin: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Get coordinates for the lower-left corner of the computational grid."""
        raise NotImplementedError("`get_grid_origin` is not implemented")

    # Non-uniform rectilinear, curvilinear

    def get_grid_x(self, grid: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the x direction."""
        raise NotImplementedError("`get_grid_x` is not implemented")

    def get_grid_y(self, grid: int, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the y direction."""
        raise NotImplementedError("`get_grid_y` is not implemented")

    def get_grid_z(self, grid: int, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the z direction."""
        raise NotImplementedError("`get_grid_z` is not implemented")

    def get_grid_node_count(self, grid: int) -> int:
        """Get the number of nodes in the grid."""
        return self.network.number_of_nodes()

    def get_grid_edge_count(self, grid: int) -> int:
        """Get the number of edges in the grid."""
        return self.network.number_of_edges()

    def get_grid_face_count(self, grid: int) -> int:
        """Get the number of faces in the grid."""
        raise NotImplementedError("`get_grid_face_count` is not implemented")

    def get_grid_edge_nodes(
        self, grid: int, edge_nodes: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Get the edge-node connectivity."""
        edge_nodes[:] = np.array(
            [node for edge in self.network.edges for node in reversed(edge)],
        )
        return edge_nodes

    def get_grid_face_edges(
        self, grid: int, face_edges: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the face-edge connectivity."""
        raise NotImplementedError("`get_grid_face_edges` is not implemented")

    def get_grid_face_nodes(
        self, grid: int, face_nodes: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the face-node connectivity."""
        raise NotImplementedError("`get_grid_face_nodes` is not implemented")

    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the number of nodes for each face."""
        raise NotImplementedError("`get_grid_nodes_per_face` is not implemented")
