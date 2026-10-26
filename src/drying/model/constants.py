"""Physical & control-domain constants (Eq. 9, 15-17) and Lewis model coefficients (Eq. 11-12).
"""

X0_WET = 0.755 
XD_WET_TARGET = 0.075 
T0_DEG_C = 25.0 # Initial ambient room temperature (C)
DEFAULT_ALPHA = 0.5
P_US_W = 15.0 # Assumed
X0_DRY_INIT = 3.16
T_LOW_DEG_C = 30.0 
T_HIGH_DEG_C = 70.0 
T_STEP_MIN = 2.0
M_DOT_AIR_CP = 1.0
W_PENALTY = 5.0
    
K_COEFF = {0: (0.07449, -45.0799, 6846.657),   # HA, refit from Malvandi et al. 2022 (ref [5]); see docs/pseudocode.md erratum
           1: (-0.05811, 39.962, -6680.1)}
MEQ_COEFFS = {0: (0.2479, -172.09, 30133),
              1: (0.1468, -107.27, 19720)}