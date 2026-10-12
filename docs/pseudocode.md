# Pseudo-code

**Source paper**: *Towards Efficient Modularity in Industrial Drying: A Combinatorial
Optimization Viewpoint*
(Bayati, Srivastava, Malvandi, Feng, Salapaka — ACC 2023)

> This document is a reconstruction of the paper's **Algorithm 1 (Combinatorial
> Optimization using MEP)** from an implementation viewpoint. It follows Python-like
> syntax; equation numbers refer to the original paper.

---

## 1. Notation and data structures

```
M          : maximum number of allowed stages (sub-processes)
Gamma_k    : set of sub-processes selectable at stage k
             (batch example: Gamma_k = {0(HA), 1(HA/US)})
gamma_k    : sub-process type chosen at stage k, gamma_k in Gamma_k
omega       : path  omega = (gamma_1, gamma_2, ..., gamma_M) in Omega
Omega       : set of all valid paths  Omega = Gamma_1 x Gamma_2 x ... x Gamma_M
             (batch example: |Omega| = 2^M)
eta_k       : control parameters of stage k, e.g. eta_k = (t_k, T_k)
U           : set of admissible control parameters
             U = {(t, T) : t >= t_0(=2min), T in [30, 70] C}          // Eq. (9)
nu(omega) {0,1} : whether path omega is chosen (original discrete problem)  // Eq. (5)
p(omega) in [0,1] : relaxed probability weight,  sum_omega p(omega) = 1      // Eq. (21)
x_0         : initial wet-basis moisture content (e.g. 75.5%)
x_d         : upper bound (target) of final moisture content (e.g. 7.5%)
beta         : MEP Lagrange multiplier (starts small, increases gradually)
```

---

## 2. Model functions (batch drying example)

### 2.1 Lewis constant and equilibrium moisture content — Eq. (11), (12)

```
# T is in Kelvin
function K(gamma, T):                       # Lewis constant
    if gamma == 0:  return ( 0.074493*T^2 - 45.5058*T + 6839.9) / 1000
    else:           return (-0.05811*T^2  + 39.962*T  - 6680.1) / 1000

function M_eq(gamma, T):                    # equilibrium dry-basis moisture content
    if gamma == 0:  return ( 0.2479*T^2 - 172.09*T + 30133) / 10000
    else:           return ( 0.1468*T^2 - 107.27*T + 19720) / 10000
```

### 2.2 Drying dynamics — Eq. (11), (13)

```
# Equivalent (elapsed) time: invert the current state x (before drying) into the
# time already "spent" in the model, then add t
function t_star(gamma, T, x, t):                                # Eq. (13)
    K_g <- K(gamma, T);  M <- M_eq(gamma, T)
    X <- x / (1 - x)                        # wet-basis -> dry-basis
    return (1/K_g) * ln( (3.16 - M) / (X - M) ) + t

# Moisture content after one stage (Lewis model solution)
#   dry-basis: X(t) = M + (3.16 - M)*exp(-K*t),  wet-basis: x = X/(1+X)
function moisture_update(gamma, x, eta):                        # Eq. (11)
    (t, T) <- eta
    K_g <- K(gamma, T);  M <- M_eq(gamma, T);  ts <- t_star(gamma, T, x, t)
    X <- M + (3.16 - M) * exp(-K_g * ts)     # dry-basis moisture content
    return X / (1 + X)                       # convert to wet-basis
```

> Note: the fraction layout of the two-line formulas in the PDF is ambiguous, so this
> is reconstructed from the Lewis model solution (Ref. [5], [13]).
> Cross-check against Eqs. (11)-(13) of the original paper.

### 2.3 Energy cost — Eq. (15)~(17)

```
function g_0(t, T):                      # energy of a pure HA sub-process
    return alpha * (rho * A * v_air * c_p) * (T - T_0) * t      # alpha: HA cost weight
    #  = alpha * m_dot_air * c_p * (T - T_0) * t   (m_dot_air = rho*A*v_air)

function g_1(t, T):                      # energy of a HA/US sub-process
    return g_0(t, T) + P_US * t          # P_US: ultrasound transducer power consumption
```

### 2.4 Path cost — Eq. (14), (18)

```
function penalty(x_M, x_d):              # penalty for violating the final-moisture constraint
    if x_M <= x_d:  return 0
    else:           return w * (x_M - x_d)     # w: large constant (barrier/exponential form allowed)

function path_cost(omega, eta, x_0, x_d):
    x <- x_0
    total <- 0
    for k = 1 .. M:
        gamma_k <- omega[k]
        total <- total + (gamma_k == 0 ? g_0(eta_k.t, eta_k.T) : g_1(eta_k.t, eta_k.T))
        x <- moisture_update(gamma_k, x, eta_k)
    return total + penalty(x, x_d)
```

---

## 3. MEP framework helper functions

```
# Eq. (25): Gibbs distribution -- probability weight for each path based on its cost
function gibbs_weights(beta, Omega, eta, x_0, x_d):
    for omega in Omega:
        D[omega] <- path_cost(omega, eta, x_0, x_d)
        logit[omega] <- -beta * D[omega]
    Z <- sum_omega exp(logit[omega])               # partition function
    for omega in Omega:
        p[omega] <- exp(logit[omega]) / Z
    return p, D, Z

# Eq. (26): free energy F* -- inner objective to minimize over parameters {eta_k}
function free_energy(beta, Omega, eta, x_0, x_d):
    p, D, Z <- gibbs_weights(beta, Omega, eta, x_0, x_d)
    return -(1/beta) * ln(Z)
    #   = sum_omega p(omega)*D(omega) - (1/beta) * H(p)    (H: Shannon entropy)
    # gradient (when available): dF*/d eta_k = sum_omega p(omega) * dD(omega)/d eta_k
```

---

## 4. Main algorithm — Algorithm 1 (Combinatorial Optimization using MEP)

```
function mep_optimize(M, {Gamma_k}, U, x_0, x_d):

    # ---------- initialization ----------
    beta   <- beta_min                # e.g. 1e-4  (smaller -> objective closer to convex)
    zeta   <- growth factor >= 1      # e.g. 1.5    (beta growth rate)
    eps    <- tolerance               # e.g. 1e-3   (criterion: max p -> 1)
    eta[k] <- suitable initial value in U, for each k = 1..M   # e.g. t = t_0, T = 50 C

    Omega <- enumerate(Gamma_1 x Gamma_2 x ... x Gamma_M)          # |Omega| = prod_k |Gamma_k| (e.g. 2^M)
    p, D, Z <- gibbs_weights(beta, Omega, eta, x_0, x_d)

    # ---------- homotopy iteration (convex function -> original non-convex function) ----------
    while max_omega p[omega] <= 1 - eps:                  # until the most likely path is practically fixed

        # [1] parameter optimization: minimize F* over {eta*_k} under the current weights p
        eta* <- constrained_optimize(
                 eta  |->  free_energy(beta, Omega, eta, x_0, x_d),
                 constraint = eta_k in U for all k,
                 initial    = eta               # previous solution as initial guess
             )
        eta <- eta*                                 # parameter update (F* decreases monotonically)

        # [2] recompute path weights with the new parameters (Eq. 25)
        p, D, Z <- gibbs_weights(beta, Omega, eta, x_0, x_d)

        # [3] increase beta -> reduce the influence of the entropy term (gradual transition
        #     to the original cost function)
        beta <- zeta * beta

    # ---------- final selection ----------
    omega* <- argmax_omega p[omega]                 # optimal path once probabilities converge to 1
    return omega*, eta*
```

### 4.1 Inner constrained optimization (original paper: interior-point)

```
function constrained_optimize(F, U, eta_initial):
    # constraints: t_k >= t_0,  T_L <= T_k <= T_U
    #  - directly: use a constrained optimizer such as interior-point / SLSQP
    #  - or reparameterize into an unconstrained problem:
    #      t = t_0 + exp(s),  T = T_L + (T_U - T_L) * sigmoid(u)
    eta* <- argmin_eta F(eta),  subject to eta_k in U,  warm-start at eta_initial
    return eta*
```

---

## 5. How the algorithm works (summary)

| Stage | State | Meaning |
|---|---|---|
| initial `beta ~ 0` | `F ~ -(1/beta) H(p)` | entropy term dominates -> objective is convex, reliably finds the global minimum |
| beta increases | | the mean-cost `sum p*D` grows in weight; the optimum tracks the global minimum of the non-convex surface |
| `max p -> 1` | `p` converges to {0,1} | discrete path choice restored -> final `omega*` decided |

- Caution: **termination condition `max_omega p >= 1-eps`** — as beta -> inf, `p(omega) -> 1`
  (the argmin path) (Eq. 28), so the maximum-probability path at loop exit is the optimal path.
- Both steps decrease `Fbar` every iteration, so **convergence is guaranteed**.
- Complexity scales with `|Omega|` (= 2^M in the example); for large M this is a
  combinatorial explosion — the paper mentions the Principle of Optimality as future work.

### 5.1 Example usage (batch drying, paper simulation settings)

```
M <- 4                       # optimal number of stages (experiments ranged 2-6)
Gamma_k <- {0, 1} for all k      # 0: HA, 1: HA/US
U  <- {(t, T) : t >= 2min, T in [30, 70] C}
x_0 <- 75.5% , x_d <- 7.5%
alpha  <- 0.5                    # relative HA cost weight
beta_min <- 1e-4, zeta <- 1.5, eps <- 1e-3
```

```
Interpreting the results (paper Fig. 4)
- M=2: about 11.96% energy saving vs. a single-stage HA/US
- M>=4: saving converges to 12.09% (diminishing returns from adding stages)
- As alpha decreases, HA modules can enter the optimal sequence (Fig. 5)
```