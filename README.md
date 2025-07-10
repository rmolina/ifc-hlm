# ifc-hlm

 A Python version of the Iowa Flood Center’s (IFC’s) Hillslope-Link Model (HLM) implementing the Basic Model Interface (BMI).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15025535.svg)](https://doi.org/10.5281/zenodo.15025535)
 
## Installation

```bash
pip install ifc-hlm
```

## Usage

```python
from ifc_hlm.models import Model252

model = Model252()
model.initialize("config.yaml")
model.solve()
```
