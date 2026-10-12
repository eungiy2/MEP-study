"""Gibbs weights / free energy -- paper Eq. (25)-(26).

MEP framework (Section III):

1. Relax the discrete path choice nu(omega) in {0,1} to probability weights
   p(omega) in [0,1]  (Eq. 21)
2. Maximizing Shannon entropy H(p) subject to the mean-cost constraint
   Dbar = D0 (Eq. 23) yields the Gibbs distribution as the stationary
   point of the Lagrangian:
       p(omega) = exp(-beta*D(omega)) / sum_omega' exp(-beta*D(omega'))   (Eq. 25)
3. Substituting p back into F gives the minimum free energy:
       F*(beta) = -(1/beta) * log sum_omega exp(-beta*D(omega))          (Eq. 26)
                = sum_omega p(omega)*D(omega) - (1/beta) * H(p)
"""

from collections.abc import Sequence

import numpy as np


def gibbs_probs(beta: float, costs: Sequence[float]) -> np.ndarray:
    """Compute the Gibbs weights p(omega) from path costs D(omega).  Eq. (25)

    beta -> 0:  uniform over all paths (max entropy)
    beta -> inf: p -> 1 for the minimum-cost path (see Eq. 28)
    """
    costs = np.asarray(costs, dtype=float)
    logits = -beta * costs
    logits -= np.max(logits)              # numerical stability (softmax max-shift)
    p = np.exp(logits)
    return p / np.sum(p)


def free_energy(beta: float, costs: Sequence[float]) -> float:
    """Free energy F*(beta) = -(1/beta) * ln sum_omega exp(-beta*D(omega)).  Eq. (26)

    Uses the numerically stable log-sum-exp trick.
    Properties:  F* <= min_omega D(omega), and F* -> min_omega D(omega) as beta -> inf
    """
    costs = np.asarray(costs, dtype=float)
    d_min = float(costs.min())
    # log sum exp(-beta D) = -beta*d_min + log sum exp(-beta (D - d_min))
    logsum = np.log(np.sum(np.exp(-beta * (costs - d_min))))
    return d_min - logsum / beta