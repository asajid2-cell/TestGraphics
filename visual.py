from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

from .io import PartRegistry, load_combos, resolve_combo_name
from .parts import Combo
from .stadium import bb10_default
from .physics import SimParams, simulate_duel_steps


def _clear():
    # ANSI clear & home
    print("\x1b[2J\x1b[H", end="")


def _draw_ascii(state, width: int = 80, height: int = 30):
    a = state["a"]
    b = state["b"]
    t = state["t"]

    # Stadium scale
    stadium = bb10_default()
    R = stadium.wall_radius_mm / 1000.0  # meters

    # Build grid
    grid = [[" "] * width for _ in range(height)]

    def to_screen(x: float, y: float):
        # map world (-R..R) to screen (0..w-1, 0..h-1)
        sx = int((x + R) / (2 * R) * (width - 1))
        sy = int((1 - (y + R) / (2 * R)) * (height - 1))
        return sx, sy

    # Draw stadium boundary (approx circle)
    for i in range(height):
        for j in range(width):
            # screen center in world
            wx = (j / (width - 1)) * 2 * R - R
            wy = (1 - i / (height - 1)) * 2 * R - R
            d = math.hypot(wx, wy)
            if abs(d - R) < (R / min(width, height)) * 3.0:
                grid[i][j] = "."

    # Draw beys
    ax, ay = to_screen(a.pos[0], a.pos[1])
    bx, by = to_screen(b.pos[0], b.pos[1])
    if 0 <= ay < height and 0 <= ax < width:
        grid[ay][ax] = "A" if a.alive else "a"
    if 0 <= by < height and 0 <= bx < width:
        grid[by][bx] = "B" if b.alive else "b"

    # Render
    _clear()
    print("BB-10 Visualization | t={:.2f}s".format(t))
    print("A: {} {}{} | spin {:.0f} | v {:.2f} | {}".format(
        a.combo.metal.name,
        a.combo.track.name,
        a.combo.tip.name,
        a.spin,
        math.hypot(a.vel[0], a.vel[1]),
        "KO" if a.ko else ("Alive" if a.alive else "Stopped"),
    ))
    print("B: {} {}{} | spin {:.0f} | v {:.2f} | {}".format(
        b.combo.metal.name,
        b.combo.track.name,
        b.combo.tip.name,
        b.spin,
        math.hypot(b.vel[0], b.vel[1]),
        "KO" if b.ko else ("Alive" if b.alive else "Stopped"),
    ))
    print("=" * width)
    for row in grid:
        print("".join(row))


def _make_combo_from_args(args, reg: PartRegistry, prefix: str) -> Combo:
    metal = reg.metals[getattr(args, f"{prefix}_metal")]
    track = reg.tracks[getattr(args, f"{prefix}_track")]
    tip = reg.tips[getattr(args, f"{prefix}_tip")]
    lp = float(getattr(args, f"{prefix}_launch_power", 0.8))
    return Combo(name=None, metal=metal, track=track, tip=tip, launch_power=lp)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ASCII visualization for a single MFL battle")
    p.add_argument("--parts", default=str(Path("data/parts.json")))
    p.add_argument("--combos", default=str(Path("data/combos.json")))
    p.add_argument("--combo1", default=None)
    p.add_argument("--combo2", default=None)
    p.add_argument("--b1-metal", dest="b1_metal", default=None)
    p.add_argument("--b1-track", dest="b1_track", default=None)
    p.add_argument("--b1-tip", dest="b1_tip", default=None)
    p.add_argument("--b1-launch-power", dest="b1_launch_power", type=float, default=0.8)
    p.add_argument("--b2-metal", dest="b2_metal", default=None)
    p.add_argument("--b2-track", dest="b2_track", default=None)
    p.add_argument("--b2-tip", dest="b2_tip", default=None)
    p.add_argument("--b2-launch-power", dest="b2_launch_power", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=float, default=40.0, help="Playback frames per second")
    p.add_argument("--size", type=str, default="80x30", help="Grid size, e.g., 80x30")

    args = p.parse_args(argv)
    reg = PartRegistry.from_json(args.parts)

    # Build combos
    # Optional catalog for parsing shorthand names (e.g., EarthBull145WD)
    catalog = None
    if Path(args.combos).exists():
        # leave as-is; combos file is optional
        pass

    if args.combo1 and args.combo2:
        c1 = c2 = None
        if Path(args.combos).exists():
            combos = load_combos(args.combos, reg)
            c1 = combos.get(args.combo1)
            c2 = combos.get(args.combo2)
        if c1 is None:
            c1 = resolve_combo_name(args.combo1, reg, catalog=None) or c1
        if c2 is None:
            c2 = resolve_combo_name(args.combo2, reg, catalog=None) or c2
        if c1 is None:
            raise SystemExit(f"Combo not found: {args.combo1}")
        if c2 is None:
            raise SystemExit(f"Combo not found: {args.combo2}")
    else:
        required = [args.b1_metal, args.b1_track, args.b1_tip, args.b2_metal, args.b2_track, args.b2_tip]
        if any(x is None for x in required):
            raise SystemExit("Provide --combo1/--combo2 or all --b?-metal/track/tip fields")
        c1 = _make_combo_from_args(args, reg, "b1")
        c2 = _make_combo_from_args(args, reg, "b2")

    # Parse size
    try:
        w_str, h_str = args.size.lower().split("x")
        width = max(40, int(w_str))
        height = max(20, int(h_str))
    except Exception:
        width, height = 80, 30

    params = SimParams()
    dt_s = 1.0 / max(5.0, args.fps)

    last_draw = 0.0
    for frame in simulate_duel_steps(c1, c2, bb10_default(), params, seed=args.seed):
        # throttle to target fps independent of sim dt
        if frame["t"] - last_draw + 1e-9 >= dt_s:
            _draw_ascii(frame, width=width, height=height)
            last_draw = frame["t"]
            time.sleep(max(0.0, dt_s - params.dt))

    # final frame already drawn; print outcome line
    a = frame["a"]; b = frame["b"]; t = frame["t"]
    if a.ko and not b.ko:
        print("Result: B wins by KO at {:.2f}s".format(t))
    elif b.ko and not a.ko:
        print("Result: A wins by KO at {:.2f}s".format(t))
    elif a.alive and not b.alive:
        print("Result: A wins by sleep-out at {:.2f}s".format(t))
    elif b.alive and not a.alive:
        print("Result: B wins by sleep-out at {:.2f}s".format(t))
    else:
        print("Result: Draw at {:.2f}s".format(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
