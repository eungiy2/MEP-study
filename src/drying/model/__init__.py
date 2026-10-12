"""model — Physics & cost model of the paper, Eq. (9)-(18).

Pure functions that only perform process physics/cost calculations.
No external dependencies such as scipy.
(Note: do not import sub-modules here before they exist -- update the
 imports only after the corresponding file is actually created.)
"""

from .constants import (
    DEFAULT_ALPHA,
    P_US_W,
    T0_DEG_C,
    X0_DRY_INIT,
    X0_WET,
    XD_WET_TARGET,
)
from .cost import ControlParams, g0, g1, path_cost, penalty
from .kinetics import K, M_eq, moisture_update, t_star
from .paths import enumerate_paths

__all__ = [
    "DEFAULT_ALPHA",
    "P_US_W",
    "T0_DEG_C",
    "X0_DRY_INIT",
    "X0_WET",
    "XD_WET_TARGET",
    "ControlParams",
    "K",
    "M_eq",
    "enumerate_paths",
    "g0",
    "g1",
    "moisture_update",
    "path_cost",
    "penalty",
    "t_star",
]