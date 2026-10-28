"""
Planned public interface: HA, HA_US, Path, stage_choices, enumerate_paths
"""

from itertools import product

HA = 0
HA_US = 1
Path = tuple[int, ...]

def stage_choices(k: int) -> tuple[int, ...]:
    return (HA, HA_US)

def enumerate_paths(m: int) -> tuple[Path, ...]:
    return tuple(product(*[stage_choices(k) for k in range(m)]))