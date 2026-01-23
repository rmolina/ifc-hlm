# ifc-hlm

 A Python version of the Iowa Flood Center’s (IFC’s) Hillslope-Link Model (HLM) implementing the Basic Model Interface (BMI).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15025535.svg)](https://doi.org/10.5281/zenodo.15025535)
 
## Installation

```bash
pip install ifc-hlm
```

## Usage

```python
from ifc_hlm.vectorized import Model252

PCP_MPS = 5 / (1000 * 3600)  # mm/hour -> m/s
PET_MPS = 0.5 / (1000 * 86400)  # mm/day -> m/s

NUM_NODES = 100

model = Model252()
model.initialize("config.toml")

model.set_value("pet", np.full(shape=NUM_NODES, fill_value=PET_MPS))
model.set_value("pcp", np.full(shape=NUM_NODES, fill_value=PCP_MPS))

for _ in range(10):
    model.update()

model.set_value("pcp", np.zeros(shape=NUM_NODES))

for _ in range(10):
    model.update()

q = model.get_value_ptr("q").copy()

model.finalize()
```
