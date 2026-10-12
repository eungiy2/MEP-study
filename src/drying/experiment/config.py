"""Simulation settings -- dataclasses to reproduce paper Section IV scenarios.

Fig. 4/6 (varying M):   M in {2,3,4,5,6}, alpha = 0.5
Fig. 5 (varying alpha): M = 5, alpha in {0.2, 0.1, 1/15, 0.05, 0.04}
"""

from dataclasses import dataclass

from ..model.constants import (
    DEFAULT_ALPHA,
    P_US_W,
    T0_DEG_C,
    X0_WET,
    XD_WET_TARGET,
)


@dataclass(frozen=True)
class StageCountScenario:
    """Experiment settings for sweeping the stage count M."""

    name: str
    m: int
    alpha: float
    x0_wet: float = X0_WET
    xd_wet: float = XD_WET_TARGET
    t_ambient_c: float = T0_DEG_C
    p_us: float = P_US_W


@dataclass(frozen=True)
class AlphaScenario:
    """Experiment settings for sweeping the relative HA cost weight alpha."""

    name: str
    alpha: float
    m: int = 5
    x0_wet: float = X0_WET
    xd_wet: float = XD_WET_TARGET
    t_ambient_c: float = T0_DEG_C
    p_us: float = P_US_W


def stage_count_scenarios() -> list[StageCountScenario]:
    """Fig. 4/6: M in {2, ..., 6}, alpha = 0.5."""
    return [
        StageCountScenario(name=f"M={m}", m=m, alpha=DEFAULT_ALPHA)
        for m in range(2, 7)
    ]


def alpha_scenarios() -> list[AlphaScenario]:
    """Fig. 5: M = 5, alpha in {0.2, 0.1, 1/15, 0.05, 0.04}."""
    return [
        AlphaScenario(name="alpha=0.2", alpha=0.2),
        AlphaScenario(name="alpha=0.1", alpha=0.1),
        AlphaScenario(name="alpha=1/15", alpha=1.0 / 15.0),
        AlphaScenario(name="alpha=0.05", alpha=0.05),
        AlphaScenario(name="alpha=0.04", alpha=0.04),
    ]