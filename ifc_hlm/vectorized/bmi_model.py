"""BmiModel implementation for the vectorized HLM."""

from typing import Any, TypeVar

import numpy as np
from bmipy import Bmi
from numpy.typing import NDArray

from .base_model import BaseModel

I = TypeVar("I")
O = TypeVar("O")
P = TypeVar("P")
G = TypeVar("G")
D = TypeVar("D")
E = TypeVar("E")


class BmiModel(Bmi, BaseModel[I, O, P, G, D, E]):

    def initialize(self, config_file: str) -> None:  # NextGen
        """Perform startup tasks for the model."""
        self.model_initialize(config_file)

    def update(self) -> None:  # NextGen
        """Advance model state by one time step."""
        self.model_update()

    def update_until(self, time: float) -> None:
        """Advance model state until the given time."""
        while self.get_current_time() < time:
            self.update()

    def finalize(self) -> None:  # NextGen
        """Perform tear-down tasks for the model."""
        ...  # Nothing to do

    def get_component_name(self) -> str:  # NextGen
        """Name of the component."""
        return self.__class__.__name__

    def get_input_item_count(self) -> int:
        """Count of a model's input variables."""
        return len(self.get_input_var_names())

    def get_output_item_count(self) -> int:
        """Count of a model's output variables."""
        return len(self.get_output_var_names())

    def get_input_var_names(self) -> tuple[str, ...]:  # type: ignore[override]
        """Get the model's input variables."""
        return self.input_names

    def get_output_var_names(self) -> tuple[str, ...]:  # type: ignore[override]
        """Get the model's output variables."""
        return self.output_names

    def get_var_grid(self, name: str) -> int:  # NextGen **
        """Get grid identifier for the given variable."""
        return 0  # TODO: Confirm

    def get_var_type(self, name: str) -> str:  # NextGen
        """Get data type of the given variable."""
        return self.get_value_ptr(name).dtype.name

    def get_var_units(self, name: str) -> str:  # NextGen
        """Get units of the given variable."""

        for var_name, unit in zip(self.input_names, self.input_units):
            if var_name == name:
                return unit

        for var_name, unit in zip(self.output_names, self.output_units):
            if var_name == name:
                return unit

        raise KeyError(f"Variable '{name}' not found in inputs or outputs.")

    def get_var_itemsize(self, name: str) -> int:  # NextGen
        """Get memory use for each array element in bytes."""
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:  # NextGen
        """Get size, in bytes, of the given variable."""
        return self.get_value_ptr(name).nbytes

    def get_var_location(self, name: str) -> str:
        """Get the grid element type that the a given variable is defined on."""
        raise NotImplementedError("get_var_location")  # TODO

    def get_current_time(self) -> float:  # NextGen
        """Return the current time of the model."""
        return self.current_time.astype(np.float64).item()

    def get_start_time(self) -> float:  # NextGen
        """Start time of the model."""
        return self.config.start_time.astype(np.float64).item()

    def get_end_time(self) -> float:  # NextGen
        """End time of the model."""
        return self.config.end_time.astype(np.float64).item()

    def get_time_units(self) -> str:  # NextGen
        """Time units of the model."""
        return self.config.time_units

    def get_time_step(self) -> float:  # NextGen
        """Return the current time step of the model."""
        return self.config.time_step.astype(np.float64).item()

    def get_value(self, name: str, dest: NDArray[Any]) -> NDArray[Any]:  # NextGen
        """Get a copy of values of the given variable."""
        dest[:] = self.get_value_ptr(name)[:]
        return dest

    def get_value_ptr(self, name: str) -> NDArray[Any]:  # NextGen *
        """Get a reference to values of the given variable."""
        if name in self.get_input_var_names():
            return getattr(self.inputs, name)

        if name in self.get_output_var_names():
            return getattr(self.outputs, name)

        raise KeyError(f"Variable '{name}' not found in inputs or outputs.")

    def get_value_at_indices(
        self, name: str, dest: NDArray[Any], inds: NDArray[np.int_]
    ) -> NDArray[Any]:  # NextGen
        """Get values at particular indices."""
        dest[:] = self.get_value_ptr(name)[inds]
        return dest

    def set_value(self, name: str, src: NDArray[Any]) -> None:  # NextGen
        """Specify a new value for a model variable."""
        self.get_value_ptr(name)[:] = src

    def set_value_at_indices(
        self, name: str, inds: NDArray[np.int_], src: NDArray[Any]
    ) -> None:
        """Specify a new value for a model variable at particular indices."""
        self.get_value_ptr(name)[inds] = src

    # Grid information

    def get_grid_rank(self, grid: int) -> int:
        """Get number of dimensions of the computational grid."""
        return 1  # TODO: Confirm

    def get_grid_size(self, grid: int) -> int:
        """Get the total number of elements in the computational grid."""
        return self.num_nodes  # TODO: Confirm

    def get_grid_type(self, grid: int) -> str:
        """Get the grid type as a string."""
        return "points"  # TODO: Confirm

    # Uniform rectilinear

    def get_grid_shape(self, grid: int, shape: NDArray[np.int_]) -> NDArray[np.int_]:
        """Get dimensions of the computational grid."""
        raise NotImplementedError("get_grid_shape")

    def get_grid_spacing(
        self, grid: int, spacing: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Get distance between nodes of the computational grid."""
        raise NotImplementedError("get_grid_spacing")

    def get_grid_origin(
        self, grid: int, origin: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Get coordinates for the lower-left corner of the computational grid."""
        raise NotImplementedError("get_grid_origin")

    # Non-uniform rectilinear, curvilinear

    def get_grid_x(self, grid: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the x direction."""
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid: int, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the y direction."""
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid: int, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the z direction."""
        raise NotImplementedError("get_grid_z")

    def get_grid_node_count(self, grid: int) -> int:
        """Get the number of nodes in the grid."""
        raise NotImplementedError("get_grid_node_count")

    def get_grid_edge_count(self, grid: int) -> int:
        """Get the number of edges in the grid."""
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_face_count(self, grid: int) -> int:
        """Get the number of faces in the grid."""
        raise NotImplementedError("get_grid_face_count")

    def get_grid_edge_nodes(
        self, grid: int, edge_nodes: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the edge-node connectivity."""
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_edges(
        self, grid: int, face_edges: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the face-edge connectivity."""
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_face_nodes(
        self, grid: int, face_nodes: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the face-node connectivity."""
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the number of nodes for each face."""
        raise NotImplementedError("get_grid_nodes_per_face")
