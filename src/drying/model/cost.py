"""Energy cost, final-moisture penalty, and path cost (Eq. 14-18).

Planned public interface: ControlParams, g0, g1, penalty, path_cost
"""

from dataclasses import dataclass

from .constants import (
    DEFAULT_ALPHA,
    M_DOT_AIR_CP,
    P_US_W,
    T0_DEG_C,
    W_PENALTY,
    X0_WET,
    XD_WET_TARGET,
)
from .kinetics import moisture_update


@dataclass
class ControlParams: 
    t: float
    T: float

def g0(eta, alpha, t_ambient_c):
    return alpha * M_DOT_AIR_CP * (eta.T - t_ambient_c) * eta.t

def g1(eta, alpha, t_ambient_c, p_us):
    return g0(eta, alpha, t_ambient_c) + p_us * eta.t

def penalty(x_wet, xd_wet) -> float:
    if x_wet <= xd_wet: return 0
    else: return W_PENALTY * (x_wet - xd_wet)

def path_cost(path, etas, x0_wet=X0_WET, xd_wet=XD_WET_TARGET, alpha=DEFAULT_ALPHA, t_ambient_c=T0_DEG_C, p_us=P_US_W):
    x = x0_wet
    total = 0.0

    for gamma, eta in zip(path, etas):
        total += g0(eta, alpha, t_ambient_c) + (gamma * p_us * eta.t)
        x = moisture_update(gamma, x, eta.t, eta.T+273.15) # C -> K conversion
    
    total += penalty(x, xd_wet)

    return total