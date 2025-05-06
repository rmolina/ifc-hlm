# ifc-hlm

 A Python version of the Hillslope-Link Model (HLM) implementing the Basic Model Interface (BMI).
 
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
