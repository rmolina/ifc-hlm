# ifc-hlm

 A Python version of the Iowa Flood Center’s (IFC’s) Hillslope-Link Model (HLM) implementing the Basic Model Interface (BMI).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15025535.svg)](https://doi.org/10.5281/zenodo.15025535)
 
## Installation

```bash
pip install ifc-hlm
```

## Usage

```python
import numpy as np

from ifc_hlm.vectorized import Model252

CONFIG_TOML_FILE = "config.toml"

PCP_MPS = 100.0 / (1000 * 3600)  # mm/hour -> m/s
PET_MPS = 19.0 / (1000 * 30 * 24 * 3600)  # mm/month -> m/s


def main() -> None:

    # Instantiate Model252 and initialize it from a config file
    model = Model252()
    model.initialize(CONFIG_TOML_FILE)

    # Retrieve component name
    component_name = model.get_component_name()
    print(f"{component_name=}")
    # component_name='Model252'

    # Retrieve input and output variable_names and units
    input_vars = {
        name: model.get_var_units(name) for name in model.get_input_var_names()
    }
    print(f"{input_vars=}")
    # input_vars={'pcp': 'm s-1', 'pet': 'm s-1'}

    output_vars = {
        name: model.get_var_units(name) for name in model.get_output_var_names()
    }
    print(f"{output_vars=}")
    # output_vars={'q': 'm3 s-1', 's_p': 'm', 's_t': 'm', 's_s': 'm'}

    # Get grid id for precipitation (or any other var since all vars use the same grid)
    grid = model.get_var_grid("pcp")
    print(f"{grid=}")
    # grid=0

    # Retrieve number of elements ( = number of links = number of hillslopes)
    grid_size = model.get_grid_size(grid)
    print(f"{grid_size=}")
    # grid_size=6359

    # Set precicipitation value (same value on all hillslopes)
    pcp = np.full(shape=grid_size, fill_value=PCP_MPS)
    model.set_value("pcp", pcp)

    # Set potential evapotranspiration value (same value on all hillslopes)
    pet = np.full(shape=grid_size, fill_value=PET_MPS)
    model.set_value("pet", pet)

    # Run 15 steps using this values
    target_time = model.get_start_time() + 15 * model.get_time_step()
    model.update_until(target_time)

    # Stop rain (set value to zero)
    pcp = np.zeros(shape=grid_size)
    model.set_value("pcp", pcp)

    # Run without rain until the end of the simulation
    target_time = model.get_end_time()
    model.update_until(target_time)

    # Get final values for discharge
    q = model.get_value_ptr("q")
    print(f"{q=}")
    # q=array([2.99072493e-06, 1.44644734e-05, 2.32179070e-05, ...,
    #     6.39214042e-06, 3.08428987e-06, 3.50394797e-06], shape=(6359,))

    model.finalize()


if __name__ == "__main__":
    main()

```
