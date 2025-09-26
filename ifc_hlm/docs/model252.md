# `model252.py` 

## Description
**`model252.py`**  
Simplified version of `model254.py`
- Compared to `model254.py` , this formulation do not calculate accumulated precipitation, accumulated runoff or baseflow
- This formulation is a baseline model with fewer processes included
- This formulation uses two forcings: precipitation and evapotranspiration
- This model does not include snow processes
- This formulation does not include tile drainage processes
- This formulation includes routing across the river network created by the hillslopes

This formulation includes:

- **Nonlinear routing:** Channel and hillslope fluxes are modeled with nonlinear storage–discharge relationships.  
- **Variable infiltration:** Infiltration capacity varies dynamically with soil saturation, storm intensity, and hillslope properties.  

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

    Other parameters used internally in the formulation are
    - reference area $Ar=1km^2$
    - reference discharge $q_r=1m^3/s$ 
    - $\beta=0.05$ dimensionless
  

---

## Outputs

- Discharge in the channel [m3/s]  
- Water ponded in the surface [m]
- Water stored in the upper layer of soil [m]
- Water stored in the bottom layer of soil [m] 

---

## Equations
In the formulation equations $L$ is the length of the channel, $A_h$ is the area of the hillslope. $s_p$ is the water stored in the ponds, $s_t$ is the water stored in the top layer of soil. $s_s$ is the water stored in the soil subsurface. $q$ is the discharge in the channel. 

**Surface Soil**
$$
\frac{ds_p}{dt}=p-q_{pc}-q_{pt}-e_p
$$
$p$ is the precipitation in the hillslope


$q_{pc}$ is the flux of water ponded on the surface to the channel and is defined by $q_pc = k_2s_p$, where $k_2=v_h(L/A_h) ×10^{-3}[1/min]$

 $q_{pt}$ is the flux of water ponded on the surface to the top layer storage and is defined by $q_{pt} = k_t\,s_p$, where $k_t=k_2 [A+B(1-s_t/s_L )^\alpha ][1/min]$

$e_p$ is the evapotranspiration in the surface of soil

**Top soil layer**
$$
\frac{ds_t}{dt}=q_{pt}-q_{ts}-e_t
$$
$q_{ts}$ is the flux of water from the top layer storage to the subsurface is defined by $q_{ts} = k_i\,s_t$ , where $k_i=k_2\beta$

$e_t$ is the evapotranspiration in the top layer of soil


**Subsurface layer**
   
$$
\frac{ds_s}{dt}=q_{ts}-q_{sc}-e_s
$$

 $q_{sc}$ is the flux of water from the subsurface to the channel is defined by $q_{sc} = k_3\,s_s$   

**Nonlinear channel routing**

The mass transport equation for each channel link in the network is given by
$$
   \frac{dq}{dt} =L\,\frac{v_{r}}{1-\lambda_1}\,\frac{q}{q_r}^{\lambda_1}\frac{A}{A_r}^{\lambda_2}\,[-q+(q_{pc}+q_{sc})\frac{A_h}{60}+q_{in}] 
$$

where $q_{in}$ is the flux from upstream channels.
 
  
**Appendix**

Fluxes representing evaporation are given by:

$c =(s_p)+(s_t/s_L)+(s_s/(h_b-s_L))$

$e_p=e_{pot}(s_p)/c$

$e_t=e_{pot}(s_t/s_L)/c$

$e_s=e_{pot}(s_s/(h_b-s_L))/c$



---

## Dependencies

- Python ≥3.8  
- `numpy`  
- Internal `ifc_hlm` modules  

---

## Example Usage

```python
from ifc_hlm.models import model252

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
hlm = model252.HLMModel(params)

# Forcings in mm/hour
precip = [2.5, 5.0, 0.0, ...]   # precipitation time series
et     = [0.1, 0.1, 0.2, ...]   # evapotranspiration series

# Run model (dt = 1.0 hour)
results = hlm.run(precip, et, dt=1.0)

Qout = results["discharge"]
```


