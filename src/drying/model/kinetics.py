"""Drying kinetics -- Lewis model (Eq. 11-13).

Planned public interface: K, M_eq, t_star, moisture_update
"""

import numpy as np

from .constants import K_COEFF, MEQ_COEFFS, X0_DRY_INIT

# T is in Kelvin

def K(gamma, T):
    if gamma == 0: 
        return (K_COEFF[0][0]*T**2 + K_COEFF[0][1]*T + K_COEFF[0][2]) / 1000
    else: 
        return (K_COEFF[1][0]*T**2 + K_COEFF[1][1]*T + K_COEFF[1][2]) / 1000

def M_eq(gamma, T):
    if gamma == 0: 
        return (MEQ_COEFFS[0][0]*T**2 + MEQ_COEFFS[0][1]*T + MEQ_COEFFS[0][2]) / 1000
    else: 
        return (MEQ_COEFFS[1][0]*T**2 + MEQ_COEFFS[1][1]*T + MEQ_COEFFS[1][2]) / 1000

def t_star(gamma, T, x, t):
    M_val = M_eq(gamma, T)
    K_val = K(gamma, T)

    return 1/K_val * np.log((X0_DRY_INIT - M_val) / ((x / (1 - x)) - M_val)) + t

def moisture_update(gamma, x_wet, t, T):
    M_val = M_eq(gamma, T)
    K_val = K(gamma, T)
    ts = t_star(gamma, T, x_wet, t)

    num = np.exp(-K_val * ts) * (X0_DRY_INIT - M_val) + M_val
    denom = 1 + np.exp(-K_val * ts) * (X0_DRY_INIT - M_val) + M_val

    return num / denom