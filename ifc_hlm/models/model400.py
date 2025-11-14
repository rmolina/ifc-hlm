from dataclasses import dataclass
from typing import TypeAlias

MILLIMETER_TO_METER: float = 1000.0
HOUR_PER_DAY: float = 24.0
MINUTE_PER_HOUR: float = 60.0
DAY_PER_MONTH: float = 30.0
MINUTE_PER_DAY: float = HOUR_PER_DAY * MINUTE_PER_HOUR
MINUTE_PER_MONTH: float = MINUTE_PER_DAY * DAY_PER_MONTH


@dataclass
class States:
    q: float  # [m3 s-1]
    h1: float  # static storage [m]
    h2: float  # water in the hillslope surface [m]
    h3: float  # water in the gravitational storage in the upper part of soil [m]
    h4: float  # water in the aquifer storage [m]
    h5: float  # snow storage [m]


Derivatives: TypeAlias = States  # Same fileds but all units divided by time


@dataclass
class Params:
    # 1
    L: float  #  Length of the channel [m]
    # 2
    A_h: float  # Area of the hillslopes [m2]
    # 3
    invtau: float
    # 4
    # c_1:float # factor .converts [mm/hr]
    # 5
    c_2: float


@dataclass
class ForcingValues:
    # 0
    rainfall: float  # rainfall [mm/hr]
    # 1
    e_pot: float  # potential et[mm/month]
    # 2
    temperature: float  # daily temperature in Celsius
    # 3
    frozen_ground: float  #  1 if ground is frozen, 0 if not frozen


@dataclass
class GlobalParams:

    # 1
    lambda_1: float

    # 3
    Hu: float  # max available storage in static tank [mm]
    # 4
    infiltration: float  # infiltration rate [mm hr-1] # TODO: confirm units
    # 5
    percolation: float  #  percolation rate to aquifer [mm hr-1] # TODO: confirm units
    # 6
    alfa2: float  # velocity in m/s
    # 7
    alfa3: float  # residence time [day]
    # 8
    alfa4: float  # residence time [day]
    # 9
    melt_factor: float  # mm day-1 degCelsius-1
    # 10
    temp_thres: float  # degCelsius


def model400(
    # float t, \
    y_i: States,
    # unsigned int dim, \
    y_p: list[States],  # unsigned short num_parents, \
    # unsigned int max_dim, \
    global_params: GlobalParams,
    params: Params,
    forcing_values: ForcingValues,  # const QVSData * const qvs, \
    # int state, \
    # void* user, \
    derivatives: Derivatives,
):
    # unsigned short i; # auxiliary variable for loops

    L = params.L  #  Length of the channel [m]
    A_h = params.A_h  # Area of the hillslopes [m2]
    # c_1:float  = params[4]; # factor .converts [mm/hr] to [m/min]

    rainfall = forcing_values.rainfall / MILLIMETER_TO_METER / MINUTE_PER_HOUR
    # rainfall. from [mm/hr] to [m/min]
    e_pot = forcing_values.e_pot / MILLIMETER_TO_METER / MINUTE_PER_MONTH
    # potential et[mm/month] -> [m/min]
    temperature = forcing_values.temperature  # daily temperature in Celsius

    temp_thres: float = global_params.temp_thres
    #  celsius degrees
    melt_factor: float = (
        global_params.melt_factor / MILLIMETER_TO_METER / MINUTE_PER_DAY
    )
    #  mm/day/degree to m/min/degree

    frozen_ground: float = (
        forcing_values.frozen_ground
    )  #  1 if ground is frozen, 0 if not frozen
    x1: float = 0

    # states
    # unsigned int STATE_DISCHARGE=0;
    # unsigned int STATE_STATIC= 1;
    # unsigned int STATE_SURFACE=2;
    # unsigned int STATE_SUBSURF=3;
    # unsigned int STATE_GW = 4;
    # unsigned int STATE_SNOW = 5;

    # INITIAL VALUES
    h5: float = y_i.h5  # [STATE_SNOW];
    h1: float = y_i.h1  # [STATE_STATIC];
    h2: float = y_i.h2  # [STATE_SURFACE];
    h3: float = y_i.h3  # [STATE_SUBSURF];
    h4: float = y_i.h4  # [STATE_GW];
    q: float = y_i.q  # [STATE_DISCHARGE];

    # snow storage
    # temperature =0 is the flag for no forcing the variable. no snow process
    if temperature == 0:
        x1 = rainfall
        derivatives.h5 = 0

    else:
        if temperature >= temp_thres:
            snowmelt = min(h5, temperature * melt_factor)  #  in [m]
            derivatives.h5 = -snowmelt  # melting outs of snow storage
            x1 = rainfall + snowmelt  #  in [m]
            #  printf("temp > th: %f\n", temperature);
            #  printf("snowmelt : %f\n", snowmelt);
        if temperature != 0 and temperature < temp_thres:
            derivatives.h5 = rainfall  # all precipitation is stored in the snow storage
            x1 = 0
            # printf("temp < th: %f\n", temperature);

    # static storage
    Hu = (
        global_params.Hu / MILLIMETER_TO_METER
    )  # max available storage in static tank [mm] to [m]
    x2: float = max(0, x1 + h1 - Hu)
    # excedance flow to the second storage [m] [m/min] check units
    # if ground is frozen, x1 goes directly to the surface
    # therefore nothing is diverted to static tank
    if frozen_ground == 1:
        x2 = x1

    d1: float = x1 - x2
    #  the input to static tank [m/min]
    out1: float = min(e_pot, h1)
    # evaporation from the static tank. it cannot evaporate more than h1 [m]
    # float out1 = (e_pot > h1) ? e_pot : 0.0;
    derivatives.h1 = d1 - out1  # differential equation of static storage

    # surface storage tank
    infiltration = (
        global_params.infiltration / MILLIMETER_TO_METER / MINUTE_PER_HOUR
    )  # infiltration rate [m/min]
    if frozen_ground == 1:
        infiltration = 0

    x3: float = min(x2, infiltration)
    # water that infiltrates to gravitational storage [m/min]
    d2: float = x2 - x3
    #  the input to surface storage [m] check units
    alfa2 = global_params.alfa2  # velocity in m/s
    w: float = alfa2 * L / A_h * 60
    #  [1/min]
    w = min(1, w)
    # water can take less than 1 min (dt) to leave surface
    out2 = h2 * w  # direct runoff [m/min]
    derivatives.h2 = d2 - out2
    # differential equation of surface storage

    #  SUBSURFACE storage
    percolation = (
        global_params.percolation / MILLIMETER_TO_METER / MINUTE_PER_HOUR
    )  #  percolation rate to aquifer [m/min]
    x4: float = min(x3, percolation)
    # water that percolates to aquifer storage [m/min]
    d3: float = x3 - x4
    #  input to gravitational storage [m/min]
    alfa3 = global_params.alfa3 * MINUTE_PER_DAY  # residence time [days] to [min].
    out3: float = 0
    if alfa3 >= 1:
        out3 = h3 / alfa3  # interflow [m/min]

    derivatives.h3 = d3 - out3
    # differential equation for gravitational storage

    # aquifer storage
    x5: float = 0  # water loss to deeper aquifer [m]
    d4: float = x4 - x5
    alfa4 = global_params.alfa4 * MINUTE_PER_DAY  # residence time [days] to [min].
    out4: float = 0
    if alfa4 >= 1:
        out4 = h4 / alfa4  # base flow [m/min]
    derivatives.h4 = d4 - out4
    # differential equation for aquifer storage

    # channel storage

    lambda_1 = global_params.lambda_1

    invtau: float = (
        params.invtau
    )  #  60.0*v_0*pow(A_i, lambda_2) / ((1.0 - lambda_1)*L_i);	# [1/min]  invtau
    c_2: float = params.c_2  #  = A_h / 60.0;	#   c_2

    derivatives.q = -q + (out2 + out3 + out4) * c_2
    # [m/min] to [m3/s]

    derivatives.q += sum(y_p_i.q for y_p_i in y_p)
    derivatives.q = invtau * pow(q, lambda_1) * derivatives.q
    #  discharge[0]

    #  if (forcing_values[0]>1 && ratio<1) {
    #      printf("time: %f\n", t);
    #      printf(" rain in mm/hour: %f\n", forcing_values[0]);
    #      printf(" area hill, area basin, area ratio: %f %f %f\n", A_h,A_i,ratio);
    #      MPI_Abort(MPI_COMM_WORLD, 1);
    #  }
