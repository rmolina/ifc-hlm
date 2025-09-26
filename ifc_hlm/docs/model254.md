# `model254.py` 

## Description
**`model254.py`**  
Implements the **Nonlinear Routing and Variable Infiltration version of the Hillslope-Link Model (HLM)** as developed at the Iowa Flood Center (IFC).
- This model is a baseline model with simplified processes
- This model uses precipitation and evapotranspiration forcings
- This model does not include snow processes
- This model does not include tile drainage processes
- This model includes routing across the river network created by the hillslopes

---

## Purpose / Use Case
This model extends the baseline HLM by incorporating:

- **Nonlinear routing:** Channel and hillslope fluxes are modeled with nonlinear storage–discharge relationships.  
- **Variable infiltration:** Infiltration capacity varies dynamically with soil saturation, storm intensity, and hillslope properties.  

This formulation provides a representation of runoff generation and streamflow hydrographs routing, especially for events with high rainfall intensity.

---

## Inputs

- **Forcings :**
  - **Precipitation** – rainfall rate applied to each hillslope, in mm/hour  
  - **Evapotranspiration** – evapotranspiration rate, in mm/month.  

- **Hillslope / channel properties:**
  - **Hillslope area** – contributing surface area draining into the channel, in square kilometers.
  - **Channel accumulated drainage area** – total upstream area contributing to a channel link, in square kilometers.  
  - **Channel length** – physical length of the channel link, in kilometers.  

- **Initial conditions:**
  - Discharge in the channel [m3/s]  
  - Water ponded in the surface [m]
  - Water stored in the upper layer of soil [m]
  - Water stored in the bottom layer of soil [m] 
  - Accumulated precipitation [m]
  - Accumulated runoff [m]
  - Baseflow [m3/s]
  

- **Parameter and default values**

    - channel reference velocity $v_r=0.3\,m/s$
    -  exponent of channel velocity discharge $\lambda_1=0.3$  (dimensionless)
    -  exponent of channel velocity area $\lambda_2=-0.1$  (dimensionless)
    - velocity of water on the hillslope surface $v_h=0.1m/s$
    - infiltration from subsurface to channel $k_3=2.3\times10^{-5}(1/min)$
    - factor of infiltration from top soil to subsurface $k_i=0.02$ (dimensionless),
    - total hillslope depth $h_b=0.5m$
    - total topsoil depth $S_L = 0.1m$
    - surface to topsoil infiltration additive factor A=0 (dimensionless)
    - surface to topsoil infiltration multiplicative factor B=99 (dimensionless),
    - surface to topsoil infiltration exponent factor $\alpha=3$ (dimensionless)

    Other parameters used internally in the model are
    - reference area $Ar=1km^2$
    - reference discharge $q_r=1m^3/s$ 
    - $\beta=0.05$ dimensionless
  

---

## Outputs

- Discharge in the channel [m3/s]  
- Water ponded in the surface [m]
- Water stored in the upper layer of soil [m]
- Water stored in the bottom layer of soil [m] 
- Accumulated precipitation [m]
- Accumulated runoff [m]
- Baseflow [m3/s]

---

## Governing Equations

**Surface Soil:**
$$
\frac{ds_p}{dt}=p-q_{pc}-q_{pt}-e_p
$$
$p$ is the precipitation in the hillslope


$q_{pc}$ is the flux of water ponded on the surface to the channel and is defined by $q_pc = k_2s_p$

 $q_{pt}$ is the flux of water ponded on the surface to the top layer storage and is defined by $qpt = k_t\,s_p$


$e_p$ is the evapotranspiration in the surface of soil

**Top soil layer:**
$$
\frac{ds_t}{dt}=q_{pt}-q_{ts}-e_t
$$
$q_{ts}$ is the flux of water from the top layer storage to the subsurface is defined by $q_{ts} = k_is_t$

$e_t$ is the evapotranspiration in the top layer of soil


**Subsurface layer:**
   
$$
\frac{ds_s}{dt}=q_{ts}-q_{sc}-e_s
$$

 $q_{sc}$ is the flux of water from the subsurface to the channel is defined by $q_{sc} = k_3s_s$
 
 The definitions of the terms $k_2$, $k_t$, $k_i$, and $k_3$ controlling the flux among storages, as well as the terms $e_p$, $e_t$, and $e_s$ controlling evapotranspiration, are shown below in appendix.   

**Nonlinear channel routing:**

The mass transport equation for each channel link in the network is given by
$$
   \frac{dq}{dt} =L\,\frac{v_{r}}{1-\lambda_1}\,\frac{q}{q_r}^{\lambda_1}\frac{A}{A_r}^{\lambda_2}\,[-q+(q_{pc}+q_{sc})\frac{A_h}{60}+q_{in}] 
$$

 
  
**Appendix**

The parameters controlling the flux among storages are given by

$k_2=v_h(L/A_h) ×10^{-3}[1/min]$

$k_3=v_g(L/A_h)×10^{-3}[1/min]$

$k_i=k_2\beta$

$k_t=k_2 [A+B(1-s_t/s_L )^\alpha ][1/min]$


Fluxes representing evaporation are given by:

$c =(s_p/s_r)+(s_t/s_L)+(s_s/(h_b-s_L))$

$e_p=e_{pot}(s_p/s_r)/c$

$e_t=e_{pot}(s_t/s_L)/c$

$e_s=e_{pot}(s_s/(h_b-s_L))/c$



---

## Dependencies

- Python ≥3.8  
- `numpy`  
- `scipy` (if numerical integration used)  
- Internal `ifc_hlm` modules  

---

## Example Usage

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
