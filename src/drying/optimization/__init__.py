"""Optimization layer -- MEP (Maximum Entropy Principle) based Algorithm 1.

- gibbs:  Eq. (25)-(26) Gibbs weights / free energy
- mep:    Algorithm 1 homotopy iteration (with inner constrained optimization)
"""

from .gibbs import free_energy, gibbs_probs
from .mep import MepConfig, MepResult, solve

__all__ = ["free_energy", "gibbs_probs", "solve", "MepConfig", "MepResult"]