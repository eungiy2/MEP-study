# Modular Drying Study

Study repository reproducing the ACC 2023 paper *"Towards Efficient Modularity in
Industrial Drying: A Combinatorial Optimization Viewpoint"* (Bayati et al.).

In a **modular drying process** that combines several drying technologies
(HA / HA/US, ...) in sequence, we **jointly** optimize (a) the choice of
technology at each stage (the sequence) and (b) the control parameters
(residence time and air temperature), so as to **minimize energy consumption**.
The problem is solved with a Maximum Entropy Principle (MEP) based homotopy
algorithm (Algorithm 1).

## Documents

- `docs/` -- original paper PDF + implementation-oriented pseudo-code (`docs/pseudocode.md`)
- `docs/pseudocode.md` -- section-by-section mapping to the paper's equation numbers

## Layout

```
src/drying/            # Python package (src layout)
├── model/             # pure functions of paper Eq. (9)-(18)
│   ├── constants.py   #   physical constants / experiment settings
│   ├── paths.py       #   enumeration of Omega = Gamma1 x ... x GammaM
│   ├── kinetics.py    #   Eq. (11)-(13) drying dynamics (Lewis model)
│   └── cost.py        #   Eq. (14)-(18) energy cost / penalty / path cost
├── optimization/      # MEP algorithm
│   ├── gibbs.py       #   Eq. (25)-(26) Gibbs weights / free energy
│   └── mep.py         #   Algorithm 1 (homotopy loop)
└── experiment/        # experiment settings & plots
    ├── config.py      #   dataclasses reproducing paper Section IV scenarios
    └── plots.py       #   Fig. 4/5/6 style

experiments/           # entry points (uv run python experiments/...)
tests/                 # pytest
outputs/               # generated figures/results (gitignored)
```

## Usage

```bash
uv sync --extra dev          # install dependencies (numpy, scipy + dev extras)
uv run pytest                # run unit tests

uv run python experiments/run_paper_figures.py   # reproduce paper figures
uv run python experiments/stage_count_effect.py
uv run python experiments/alpha_effect.py
```

## Status

- [x] `docs/pseudocode.md` -- algorithm pseudo-code notes
- [x] Skeleton structure / model pure-function scaffolding
- [x] `docs/pseudocode.md` -- recorded an **erratum in paper Eq. (11)**: the
      printed `K0` coefficient set is inconsistent with the source drying data
      (Malvandi et al. 2022, ref [5]) in any temperature unit. Verified against
      the experiments: `K1`/`M0`/`M1` are correct **in Kelvin** with
      `M_eq / 10000`; `K0` must be re-fitted as
      `(0.07449 T^2 - 45.0799 T + 6846.657)/1000`. See `docs/pseudocode.md` §2.1.
- [ ] Apply the corrected `K0` coefficients in `constants.py` (re-fit from ref [5])
- [ ] Tune the inner constrained optimization (solver / bounds / scaling)
- [ ] Verify reproduction of paper Figs. 4/5/6