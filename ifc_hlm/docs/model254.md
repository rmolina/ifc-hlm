# `model254.py` – Nonlinear Routing and Variable Infiltration Hillslope-Link Model

## 1. Name
**`model254.py`**  
Implements the **Nonlinear Routing and Variable Infiltration version of the Hillslope-Link Model (HLM)** as developed at the Iowa Flood Center (IFC).

---

## 2. Purpose / Use Case
This model extends the baseline HLM by incorporating:

- **Nonlinear routing:** Channel and hillslope fluxes are modeled with nonlinear storage–discharge relationships.  
- **Variable infiltration:** Infiltration capacity varies dynamically with soil saturation, storm intensity, and hillslope properties.  

This formulation provides a more physically realistic representation of runoff generation and streamflow hydrographs, especially for events with high rainfall intensity, steep slopes, or tile drainage influence.

---

## 3. Inputs

- **Climate forcings (mm/hour):**
  - **Precipitation** – rainfall rate applied to each hillslope.  
  - **Evapotranspiration** – evapotranspiration rate reducing soil storage.  

- **Hillslope / channel properties:**
  - **Hillslope area (km²)** – contributing surface area draining into the channel.  
  - **Channel accumulated drainage area (km²)** – total upstream area contributing to a channel link.  
  - **Channel length (km)** – physical length of the channel link.  

- **Initial conditions:**
  - Soil moisture storage per hillslope (mm).  
  - Baseflow / groundwater storage (mm or m³/s equivalent).  
  - Channel storage states (m³ or equivalent depth).  

- **Configuration parameters:**
  - Nonlinear exponents and coefficients for hillslope and channel routing.  
  - Infiltration scaling parameters.  
  - Time step Δt (hours by default).  

---

## 4. Outputs

- **Hydrographs:** Streamflow at the outlet and at internal channel links.  
- **Hydrologic states:**
  - Soil moisture  
  - Channel storage  
  - Baseflow contributions  
- **Fluxes:**
  - Infiltration  
  - Evapotranspiration losses  
  - Overland flow  
  - Subsurface/tile drainage  
  - Routed channel discharge  

---

## 5. Governing Equations

1. **Surface Soil:**
$$
\frac{ds_p}{dt}=p-q_{pc}-q_{pt}-e_p
$$
   

2. **Top soil later:**
$$
\frac{ds_t}{dt}=q_{pt}-q_{ts}-e_t
$$

3. **Subsurface layer:**
   
$$
\frac{ds_s}{dt}=q_{ts}-q_{sc}-e_s
$$
   

4. **Nonlinear channel routing:**
The mass transport equation for each channel link in the network is given by
$$
   \frac{dq}{dt} =L\,\frac{v_{r}}{1-\lambda_1}\,\frac{q}{q_r}^{\lambda_1}\frac{A}{A_r}^{\lambda_2}\,[-q+q_{pc}+q_{sc}\frac{A_h}{60}+q_{in}] 
$$

 
   where \(P\) is precipitation rate (mm/h), \(I\) infiltration, and \(ET\) evapotranspiration. 
5. **Appendix**
The parameters controlling the flux among storages are given by
$$

$$
---

## 6. Dependencies

- Python ≥3.8  
- `numpy`  
- `scipy` (if numerical integration used)  
- Internal `ifc_hlm` modules  

---

## 7. Example Usage

```python
from ifc_hlm.models import model254

# Define parameters (see lines 62–73 for parameter set)
params = {
    "hillslope_area": [...],   # km²
    "channel_area": [...],     # km²
    "channel_length": [...],   # km
    "infiltration": {...},
    "routing": {...},
    "initial_state": {...}
}

# Initialize model
hlm = model254.HLMModel(params)

# Forcings in mm/hour
precip = [2.5, 5.0, 0.0, ...]   # precipitation time series
et     = [0.1, 0.1, 0.2, ...]   # evapotranspiration series

# Run model (dt = 1.0 hour)
results = hlm.run(precip, et, dt=1.0)

Qout = results["discharge"]
