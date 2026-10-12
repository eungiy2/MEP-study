"""drying — Modular Industrial Drying Optimization (ACC 2023 reproduction).

Batch drying example: arrange HA(0) / HA/US(1) sub-processes into M stages
and jointly optimize the control parameters eta_k=(t_k, T_k) via the MEP
(Maximum Entropy Principle) based Algorithm 1.

Package layout:
- model:         physics & cost model (Eq. 9-18), pure functions only
- optimization:  MEP Algorithm 1 (Eq. 24-28)
- experiment:    paper Section IV scenarios & visualization
"""