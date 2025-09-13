<<<<<<< HEAD
Beyblade Metal Fight Limited — Battle Simulator

Overview

This repository contains a lightweight, extensible simulator to model Beyblade Metal Fight Limited (MFL) battles. It focuses on realistic-but-approximate physics for the BB-10 Attack Type stadium, and a composable data model for parts. The goal is to enable quick Monte Carlo experiments over different part combinations and launch parameters to estimate win rates and outcome distributions (KO, sleep-out, etc.).

Important Notes

- This simulator uses simplified physics and approximate parameters. It is not a ground-truth physics engine. Treat results as directional, not definitive.
- The included seed data is intentionally conservative and incomplete. You can and should expand it using `data/parts.json` and optionally `data/combos.json`.
- Network access is disabled in this environment, so the dataset does not attempt to scrape or include third-party sources. You may import your own data files.

Quick Start

1) Create or review `data/parts.json` and optionally `data/combos.json`.
2) Run a quick simulation for two combos:

   `python -m beyblade_sim.cli simulate --combo1 AggroRF --combo2 StaminaS --runs 200`

3) Or specify parts directly:

   `python -m beyblade_sim.cli simulate --b1-metal SampleAttack --b1-track 145 --b1-tip RF --b2-metal SampleDefense --b2-track 145 --b2-tip S --runs 200`

Realtime Visualization (Window)

- Open a top-down BB-10 window with moving beys:
  - `python -m beyblade_sim.visual_rt --combo1 Flame230CS --combo2 Meteo145WD --fps 60 --size 900x700 --seed 7`
  - If Tkinter is missing, install/enable it for your Python distribution.

Project Layout

- `beyblade_sim/parts.py`: Data models for parts and combos + stat aggregation.
- `beyblade_sim/stadium.py`: Stadium geometry and environment parameters (BB-10 approximation).
- `beyblade_sim/physics.py`: Core time-step integrator and collision/friction model.
- `beyblade_sim/simulator.py`: High-level simulation orchestration and Monte Carlo runner.
- `beyblade_sim/io.py`: JSON-based loading for parts and combos.
- `beyblade_sim/cli.py`: Minimal CLI to run simulations and print summaries.
- `data/parts.json`: Seed parameters for a few common tip archetypes and placeholder metal wheels.
- `data/combos.json`: Example combos for quick testing.

Data Model (JSON)

- `tips`: objects with `name`, `mu_static`, `mu_kinetic`, `spin_friction`, `stability`, `shape`.
- `tracks`: objects with `name`, `height_mm`, `scrape_risk`.
- `metal_wheels`: objects with `name`, `mass_g`, `radius_mm`, `attack`, `defense`, `stamina`, `recoil`.

Combos (optional)

Each combo is `{ "name": string, "metal": name, "track": name, "tip": name, "launch_power": 0..1 }`.

Limitations and Assumptions

- Simplified 2D dynamics with aggregate properties (mass, radius, effective friction). No detailed tooth geometry.
- Collisions are inelastic with a tunable restitution factor; recoil and attack ratings modulate impulses and spin loss.
- KO detection is simplified to crossing the stadium boundary with sufficient outward momentum.
- Left- vs right-spin interactions: basic spin-equalization modeling during contact based on grip and wheel properties.
- Destabilization modeled via track height differences and tip stability.
- Life After Death (LAD) modeled via reduced spin-loss and lower stop threshold on high-LAD tips.

License

This project is provided as-is for research and practice. You are responsible for validating any conclusions before competitive use.

Tuning

- `SimParams` in `beyblade_sim/physics.py` controls key realism knobs:
  - `restitution`: lower reduces post-collision bounce.
  - `ko_speed_threshold`: higher makes KOs less frequent at the rim.
  - `base_contact_loss` and `random_spin_jitter`: control spin losses and variance.
- Tip coefficients (`mu_static`, `mu_kinetic`, `spin_friction`, `stability`) strongly affect behavior; refine with local testing.
- Tip `lad` (0..1) increases survival in low-spin wobble/roll conditions.
- Metal wheel `attack/defense/stamina/recoil` are abstract ratings (0..1). Start coarse, then adjust based on trial results.
- Metal wheel `left_spin` and `spin_eq` influence opposite-spin equalization behavior.

Extending Data

- Add MFL-legal parts to `data/parts.json`. Keep naming consistent and avoid duplicates.
- Optionally maintain a legality list at `data/mfl_legality.json` and enable `--validate-mfl` in the CLI.
- Add your preferred stock or custom combos to `data/combos.json` for quick CLI access.
- If you collect measured data (weights, radii), populate `mass_g` and `radius_mm` accordingly for improved accuracy.

GUI Launch

- Easiest: from the repo root (this folder), run one of:
  - `python -m beyblade_sim.gui`
  - `python -m beyblade_sim` (shortcut to the same GUI)
  - `python launch_gui.py` (path-based launcher if `-m` has issues)
- If you see `No module named 'beyblade_sim...'`:
  - You’re probably not in the repo root. `cd` to the folder that contains the `beyblade_sim/` directory and try again.

