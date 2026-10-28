import numpy as np

from .constants import K_COEFF, MEQ_COEFFS, X0_DRY_INIT


def K(gamma, T):
    """Lewis rate constant K in 1/min for sub-process gamma at temperature T.

    T is given in Kelvin.  Internally, gamma == 0 (HA) coefficients are
    evaluated in Celsius (see module docstring for why); gamma == 1 (HA/US)
    in Kelvin.
    """
    return (K_COEFF[gamma][0] * T ** 2
            + K_COEFF[gamma][1] * T
            + K_COEFF[gamma][2]) / 1000


def M_eq(gamma, T):
    """Equilibrium dry-basis moisture content (Eq. 12), T in Kelvin."""
    return (MEQ_COEFFS[gamma][0] * T ** 2
            + MEQ_COEFFS[gamma][1] * T
            + MEQ_COEFFS[gamma][2]) / 10000


def _dry_basis(x_wet):
    """Wet-basis moisture x in [0,1) -> dry-basis X = x/(1-x)."""
    return x_wet / (1.0 - x_wet)


def t_star(gamma, T, x, t):
    """Equivalent (elapsed) time t* given current wet-basis x and added t (Eq. 13)."""
    M_val = M_eq(gamma, T)
    K_val = K(gamma, T)
    arg = (X0_DRY_INIT - M_val) / max(_dry_basis(x) - M_val, np.finfo(float).eps)
    return 1.0 / K_val * np.log(arg) + t


def moisture_update(gamma, x_wet, t, T):
    """Wet-basis moisture content after a stage of duration t at T (Eq. 11).

    Closed-form Lewis solution (numerically stable; see module notes):
        X(t) = M + (X_0 - M) * exp(-K t),   x = X/(1+X)
    This is algebraically identical to the t_star formulation of Eq. (11)-(13)
    while remaining finite for all X_0 >= 0 (moisture simply stops changing
    once the product has reached equilibrium M).
    """
    M_val = M_eq(gamma, T)
    K_val = K(gamma, T)
    X = M_val + (_dry_basis(x_wet) - M_val) * np.exp(-K_val * t)
    return X / (1.0 + X)