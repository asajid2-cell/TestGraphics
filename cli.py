from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

try:
    from .io import PartRegistry, load_combos, resolve_combo_name
    from .parts import Combo
    from .stadium import bb10_default
    from .simulator import run_series
    from .legality import LegalProfile
    from .legality import Catalog as LegalCatalog
except Exception:
    import os, sys as _sys
    _sys.path.append(os.path.dirname(__file__))
    from io import PartRegistry, load_combos, resolve_combo_name
    from parts import Combo
    from stadium import bb10_default
    from simulator import run_series
    from legality import LegalProfile
    from legality import Catalog as LegalCatalog


def make_combo_from_args(args, reg: PartRegistry, prefix: str) -> Combo:
    metal = reg.metals[vars(args)[f"{prefix}_metal"]]
    track = reg.tracks[vars(args)[f"{prefix}_track"]]
    tip = reg.tips[vars(args)[f"{prefix}_tip"]]
    launch_power = float(vars(args).get(f"{prefix}_launch_power", 0.8))
    return Combo(name=None, metal=metal, track=track, tip=tip, launch_power=launch_power)


def cmd_simulate(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Beyblade MFL Simulator")
    p.add_argument("simulate", nargs="?")
    p.add_argument("--parts", default=str(Path("data/parts.json")), help="Path to parts.json")
    p.add_argument("--combos", default=str(Path("data/combos.json")), help="Path to combos.json (optional)")
    p.add_argument("--combo1", default=None, help="Name of combo 1 from combos.json")
    p.add_argument("--combo2", default=None, help="Name of combo 2 from combos.json")
    p.add_argument("--runs", type=int, default=200, help="Number of battles")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--no-ko", action="store_true", help="Disable KOs; sleep-out/time only")
    p.add_argument("--random-launch", action="store_true", help="Randomize launch power for each run")
    p.add_argument("--preset", default="default", help="Tuning preset (e.g., stamina-long)")
    p.add_argument("--preset-json", default=None, help="Advanced preset JSON (stadium/tips/bey/launch)")
    # Optional overrides for quick calibration
    p.add_argument("--spin-loss-scale", type=float, default=None)
    p.add_argument("--lin-fric-scale", type=float, default=None)
    p.add_argument("--slope-scale", type=float, default=None)
    p.add_argument("--contact-loss-scale", type=float, default=None)
    p.add_argument("--launch-spin-scale", type=float, default=None)
    p.add_argument("--vel-launch-scale", type=float, default=None)
    p.add_argument("--min-spin", type=float, default=None)
    p.add_argument("--max-time", type=float, default=None)
    p.add_argument("--wall-tangent-damping", type=float, default=None)
    p.add_argument("--wall-spin-loss", type=float, default=None)
    p.add_argument("--wall-min-speed-keep", type=float, default=None)
    p.add_argument("--wall-reentry-radial", type=float, default=None)
    p.add_argument("--wall-restitution-boost", type=float, default=None)
    p.add_argument("--wall-restitution-vref", type=float, default=None)
    p.add_argument("--wall-torque-coupling", type=float, default=None)
    p.add_argument("--wall-tangent-min-keep", type=float, default=None)
    p.add_argument("--validate-mfl", action="store_true", help="Validate combos against MFL legality profile")
    p.add_argument("--legality", default=str(Path("data/mfl_legality.json")), help="Path to MFL legality JSON")
    p.add_argument("--catalog", default=str(Path("data/catalog_mfl.json")), help="Path to MFL catalog (names + aliases)")
    # Direct specification
    p.add_argument("--b1-metal", dest="b1_metal", default=None)
    p.add_argument("--b1-track", dest="b1_track", default=None)
    p.add_argument("--b1-tip", dest="b1_tip", default=None)
    p.add_argument("--b1-launch-power", dest="b1_launch_power", type=float, default=0.8)
    p.add_argument("--b2-metal", dest="b2_metal", default=None)
    p.add_argument("--b2-track", dest="b2_track", default=None)
    p.add_argument("--b2-tip", dest="b2_tip", default=None)
    p.add_argument("--b2-launch-power", dest="b2_launch_power", type=float, default=0.8)

    args = p.parse_args(argv)

    reg = PartRegistry.from_json(args.parts)

    combo1: Combo
    combo2: Combo

    catalog = None
    if Path(args.catalog).exists():
        try:
            catalog = LegalCatalog.from_json(args.catalog)
        except Exception:
            catalog = None

    if args.combo1 and args.combo2:
        combo1 = combo2 = None
        if Path(args.combos).exists():
            combos = load_combos(args.combos, reg)
            combo1 = combos.get(args.combo1)
            combo2 = combos.get(args.combo2)
        if combo1 is None:
            combo1 = resolve_combo_name(args.combo1, reg, catalog=catalog) or combo1
        if combo2 is None:
            combo2 = resolve_combo_name(args.combo2, reg, catalog=catalog) or combo2
        if combo1 is None:
            raise SystemExit(f"Combo not found: {args.combo1}")
        if combo2 is None:
            raise SystemExit(f"Combo not found: {args.combo2}")
    else:
        required = [args.b1_metal, args.b1_track, args.b1_tip, args.b2_metal, args.b2_track, args.b2_tip]
        if any(x is None for x in required):
            raise SystemExit("Either provide --combo1/--combo2 with combos.json or specify both beys with --b?-metal/track/tip")
        combo1 = make_combo_from_args(args, reg, "b1")
        combo2 = make_combo_from_args(args, reg, "b2")

    # Optional legality validation
    if args.validate_mfl and Path(args.legality).exists():
        profile = LegalProfile.from_json(args.legality)
        errs1 = profile.validate_combo(combo1)
        errs2 = profile.validate_combo(combo2)
        if errs1:
            print("Combo A legality issues:")
            for e in errs1:
                print(" -", e)
        if errs2:
            print("Combo B legality issues:")
            for e in errs2:
                print(" -", e)

    # Sim params (with preset)
    from .tuning import params_preset
    params = params_preset(args.preset)
    if args.no_ko:
        params.allow_ko = False
    # Apply overrides if provided
    if args.spin_loss_scale is not None:
        params.spin_loss_scale = args.spin_loss_scale
    if args.lin_fric_scale is not None:
        params.lin_fric_scale = args.lin_fric_scale
    if args.slope_scale is not None:
        params.slope_scale = args.slope_scale
    if args.contact_loss_scale is not None:
        params.contact_loss_scale = args.contact_loss_scale
    if args.launch_spin_scale is not None:
        params.launch_spin_scale = args.launch_spin_scale
    if args.vel_launch_scale is not None:
        params.vel_launch_scale = args.vel_launch_scale
    if args.min_spin is not None:
        params.min_spin = args.min_spin
    if args.max_time is not None:
        params.max_time = args.max_time
    if args.wall_tangent_damping is not None:
        params.wall_tangent_damping = args.wall_tangent_damping
    if args.wall_spin_loss is not None:
        params.wall_spin_loss = args.wall_spin_loss
    if args.wall_min_speed_keep is not None:
        params.wall_min_speed_keep = args.wall_min_speed_keep
    if args.wall_reentry_radial is not None:
        params.wall_reentry_radial = args.wall_reentry_radial
    if args.wall_restitution_boost is not None:
        params.wall_restitution_boost = args.wall_restitution_boost
    if args.wall_restitution_vref is not None:
        params.wall_restitution_vref = args.wall_restitution_vref
    if args.wall_torque_coupling is not None:
        params.wall_torque_coupling = args.wall_torque_coupling
    if args.wall_tangent_min_keep is not None:
        params.wall_tangent_min_keep = args.wall_tangent_min_keep

    # Advanced preset JSON can override stadium/combos/params
    stadium = bb10_default()
    if args.preset_json:
        from .advanced_preset import apply_advanced_preset
        combo1, combo2, params, stadium = apply_advanced_preset(args.preset_json, combo1, combo2, params, stadium)

    summary = run_series(combo1, combo2, runs=args.runs, stadium=stadium, seed=args.seed, params=params, randomize_launch=args.random_launch)

    # Print concise report
    print("=== Simulation Summary ===")
    print(f"Runs: {summary.runs}")
    print(f"A wins: {summary.a_wins} (KO {summary.ko_wins_a})")
    print(f"B wins: {summary.b_wins} (KO {summary.ko_wins_b})")
    print(f"Draws: {summary.draws}")
    awr = summary.a_wins / summary.runs
    bwr = summary.b_wins / summary.runs
    print(f"Win rate A: {awr:.3f} | B: {bwr:.3f}")
    print(f"Avg battle time: {summary.avg_time:.2f}s")
    return 0


def main():
    return cmd_simulate()


if __name__ == "__main__":
    raise SystemExit(main())
