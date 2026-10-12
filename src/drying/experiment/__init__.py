"""Experiment layer -- paper Section IV scenario settings and visualization."""

from .config import (
    AlphaScenario,
    StageCountScenario,
    alpha_scenarios,
    stage_count_scenarios,
)
from .plots import (
    plot_beta_history,
    plot_cost_vs_alpha,
    plot_cost_vs_stage,
    plot_temperature_profile,
)

__all__ = [
    "StageCountScenario",
    "AlphaScenario",
    "stage_count_scenarios",
    "alpha_scenarios",
    "plot_cost_vs_stage",
    "plot_cost_vs_alpha",
    "plot_temperature_profile",
    "plot_beta_history",
]