"""Drying kinetics -- Lewis model (Eq. 11-13).

Public interface: K, M_eq, t_star, moisture_update

Temperature-unit reconstruction note
------------------------------------
The paper's Eq. (11) annotation says "Tk in Kelvin" for BOTH K0 and K1, but
numerically that cannot be right:

* K0(T) = (0.074493*T^2 - 45.5058*T + 6839.9)/1000 is NEGATIVE over the whole
  operating range [30,70] C when T is in Kelvin (it turns positive only above
  ~70.6 C).  That would make pure hot-air (HA) drying unphysical (moisture
  would *increase* with time).  The same polynomial is positive and
  well-behaved over [30,70] C when T is in Celsius, so K0 was evidently
  fitted with T in Celsius -- a transcription inconsistency in the paper,
  likely inherited from the HA kinetics reference.

* K1(T) = (-0.05811*T^2 + 39.962*T - 6680.1)/1000 is positive and plausible
  (K1 ~ 0.09..0.19 1/min) only when T is in Kelvin.

* M0/M1 (Eq. 12) are also sensible only with T in Kelvin (dry-basis
  equilibrium moisture ~0.02..0.08, decreasing with T), and Eq. (12) divides
  by 10000 (an earlier draft of this repo mistakenly divided by 1000).

So we adopt the reconstructed convention:  K0 in Celsius, K1 and M_eq in
Kelvin, M_eq divided by 10000.  All public functions take T in Kelvin (as
the paper's API suggests); the Celsius conversion happens inside K().
"""

import numpy as np

from .constants import K_COEFF, MEQ_COEFFS, X0_DRY_INIT


def K(gamma, T):
    """Lewis rate constant K in 1/min for sub-process gamma at temperature T.

    T is given in Kelvin.  Internally, gamma == 0 (HA) coefficients are
    evaluated in Celsius (see module docstring for why); gamma == 1 (HA/US)
    in Kelvin, per Eq. (11).
    """
    if gamma == 0:
        T_eval = T - 273.15            # K0 coefficients are fitted in Celsius
    else:
        T_eval = T                     # K1 coefficients are fitted in Kelvin
    return (K_COEFF[gamma][0] * T_eval ** 2
            + K_COEFF[gamma][1] * T_eval
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
    # C-Eq. (13) is derived from the Lewis solution by inverting the elapsed
    # time, and is only valid while the current content X is above equilibrium
    # M.  Clamp the argument so X <= M (already at/under equilibrium) maps to
    # t* -> +inf instead of a log of a non-positive number.
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