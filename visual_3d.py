from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

try:
    import tkinter as tk
except Exception:
    tk = None

try:
    from .io import PartRegistry, load_combos, resolve_combo_name
    from .parts import Combo
    from .stadium import bb10_default, Stadium
    from .physics import SimParams, simulate_duel_steps
    from .tuning import params_preset
except Exception:
    # Fallback to single-folder execution
    import os, sys as _sys, importlib.util as _ilu
    _base = os.path.dirname(__file__)
    if _base not in _sys.path:
        _sys.path.insert(0, _base)
    _spec = _ilu.spec_from_file_location("io_local", os.path.join(_base, "io.py"))
    io_local = _ilu.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(io_local)
    PartRegistry = io_local.PartRegistry
    load_combos = io_local.load_combos
    resolve_combo_name = io_local.resolve_combo_name
    from parts import Combo
    from stadium import bb10_default, Stadium
    from physics import SimParams, simulate_duel_steps
    try:
        from tuning import params_preset
    except Exception:
        from .tuning import params_preset


class Visual3D:
    def __init__(self, width: int, height: int, fps: float, stadium: Stadium, allow_ko: bool = True, bey_size_scale: float = 1.0, trail_length: int = 150):
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.root = tk.Tk()
        self.root.title("MFL Simulator — 3D Stadium View")
        self.w, self.h = width, height
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#0b0b0b")
        self.canvas.pack()
        self.fps = max(5.0, fps)
        self.interval_ms = int(1000 / self.fps)
        self.stadium = stadium
        self.Rm = stadium.wall_radius_mm / 1000.0
        self.allow_ko = allow_ko

        # Camera/projection
        self.cx = width // 2
        self.cy = height // 2 + 40
        self.world_scale = min(width, height) * 0.8 / (2 * self.Rm)
        self.tilt_deg = 45.0
        self.focal = 1.2  # perspective factor (world meters)

        # Trails
        self.trailA = []
        self.trailB = []
        self.max_trail = max(40, int(trail_length))
        self.bey_size_scale = max(0.3, min(2.0, bey_size_scale))

        self.text_info = self.canvas.create_text(
            10, 10, anchor="nw", fill="#e0e0e0", font=("Consolas", 10), text=""
        )
        self.result_text = None

    def bowl_depth(self, r: float) -> float:
        # Approximate BB-10 bowl depth: ~18mm at rim, 0 at center (for visual effect)
        max_depth = 0.018  # meters
        x = max(0.0, min(1.0, r / self.Rm))
        return max_depth * (x ** 2)

    def world_to_camera(self, x: float, y: float, z: float):
        # Rotate around X by tilt, camera looking toward +y
        tilt = math.radians(self.tilt_deg)
        xr = x
        yr = y * math.cos(tilt) - z * math.sin(tilt)
        zr = y * math.sin(tilt) + z * math.cos(tilt)
        return xr, yr, zr

    def project(self, x: float, y: float, z: float):
        xr, yr, zr = self.world_to_camera(x, y, z)
        # Simple perspective
        denom = (self.focal + yr)
        s = self.world_scale * (self.focal / denom) if denom != 0 else self.world_scale
        sx = int(self.cx + xr * s)
        sy = int(self.cy - zr * s)
        return sx, sy, s

    def draw_bowl(self):
        # Draw a few concentric rings to suggest curvature
        for frac, col in [(1.0, "#555"), (0.85, "#2e2e2e"), (0.65, "#262626"), (0.45, "#1f1f1f")]:
            r = self.Rm * frac
            pts = []
            for a in range(0, 360, 6):
                th = math.radians(a)
                x = r * math.cos(th)
                y = r * math.sin(th)
                z = -self.bowl_depth(r)
                sx, sy, _ = self.project(x, y, z)
                pts.append((sx, sy))
            # Draw as polyline
            for i in range(1, len(pts)):
                x0, y0 = pts[i - 1]
                x1, y1 = pts[i]
                self.canvas.create_line(x0, y0, x1, y1, fill=col)

        # pockets (visual lines) at 0/120/240 degrees
        for ang_deg in (0, 120, 240):
            ang = math.radians(ang_deg)
            inner = self.Rm * 0.96
            outer = self.Rm * 1.05
            x0 = inner * math.cos(ang)
            y0 = inner * math.sin(ang)
            z0 = -self.bowl_depth(inner)
            x1 = outer * math.cos(ang)
            y1 = outer * math.sin(ang)
            z1 = -self.bowl_depth(outer)
            sx0, sy0, _ = self.project(x0, y0, z0)
            sx1, sy1, _ = self.project(x1, y1, z1)
            self.canvas.create_line(sx0, sy0, sx1, sy1, fill="#8a7a44", width=2)

    def draw_trail(self, trail, base_color: str, glow: str):
        if len(trail) < 2:
            return
        bg = "#0b0b0b"
        n = len(trail)
        for i in range(1, n):
            (x0, y0, z0) = trail[i - 1]
            (x1, y1, z1) = trail[i]
            sx0, sy0, s0 = self.project(x0, y0, z0)
            sx1, sy1, s1 = self.project(x1, y1, z1)
            age = i / n
            alpha = max(0.0, 1.0 - age)
            alpha = alpha * alpha * 0.9
            # simple blend toward bg
            def blend(c1, c2, a):
                def h2rgb(h):
                    h = h.lstrip('#'); return tuple(int(h[j:j+2],16) for j in (0,2,4))
                def rgb2h(rgb):
                    return '#%02x%02x%02x' % rgb
                r1,g1,b1=h2rgb(c1); r2,g2,b2=h2rgb(c2)
                r=int(r2*(1-a)+r1*a); g=int(g2*(1-a)+g1*a); b=int(b2*(1-a)+b1*a)
                return rgb2h((r,g,b))
            col = blend(glow, bg, alpha)
            self.canvas.create_line(sx0, sy0, sx1, sy1, fill=col, width=1)

    def draw_bey(self, x: float, y: float, z: float, r_m: float, color: str, alive: bool):
        sx, sy, s = self.project(x, y, z)
        rpx = max(2, int(r_m * s * self.bey_size_scale))
        self.canvas.create_oval(sx - rpx, sy - rpx, sx + rpx, sy + rpx, outline=color, fill=color if alive else "", width=2)

    def run(self, c1: Combo, c2: Combo, params: SimParams, seed: Optional[int] = None):
        stepper = simulate_duel_steps(c1, c2, self.stadium, params, seed=seed)
        last_draw = 0.0
        dt_s = 1.0 / self.fps

        def tick():
            nonlocal last_draw
            try:
                frame = next(stepper)
            except StopIteration:
                return
            a = frame["a"]; b = frame["b"]; t = frame["t"]

            # 3D positions (x,y) on bowl, z is negative depth
            ra = math.hypot(a.pos[0], a.pos[1])
            rb = math.hypot(b.pos[0], b.pos[1])
            za = -self.bowl_depth(ra)
            zb = -self.bowl_depth(rb)

            # Trails
            self.trailA.append((a.pos[0], a.pos[1], za))
            self.trailB.append((b.pos[0], b.pos[1], zb))
            if len(self.trailA) > self.max_trail:
                self.trailA.pop(0)
            if len(self.trailB) > self.max_trail:
                self.trailB.pop(0)

            # Draw scene
            self.canvas.delete("all")
            self.draw_bowl()
            self.draw_trail(self.trailA, "#ff4444", "#ff6666")
            self.draw_trail(self.trailB, "#4488ff", "#66aaff")
            self.draw_bey(a.pos[0], a.pos[1], za, a.radius(), "#ff3333", a.alive)
            self.draw_bey(b.pos[0], b.pos[1], zb, b.radius(), "#3388ff", b.alive)

            vA = math.hypot(a.vel[0], a.vel[1])
            vB = math.hypot(b.vel[0], b.vel[1])
            info = f"t={t:5.2f}s | A spin {a.spin:6.1f} v {vA:4.2f} | B spin {b.spin:6.1f} v {vB:4.2f}"
            self.canvas.itemconfigure(self.text_info, text=info)

            # End condition
            if (self.allow_ko and ((a.ko and not b.ko) or (b.ko and not a.ko))) or ((not a.alive) ^ (not b.alive)):
                if self.result_text is None:
                    if self.allow_ko and a.ko and not b.ko:
                        msg = f"B wins by KO at {t:.2f}s"; winner = "B"
                    elif self.allow_ko and b.ko and not a.ko:
                        msg = f"A wins by KO at {t:.2f}s"; winner = "A"
                    elif a.alive and not b.alive:
                        msg = f"A wins by sleep-out at {t:.2f}s"; winner = "A"
                    elif b.alive and not a.alive:
                        msg = f"B wins by sleep-out at {t:.2f}s"; winner = "B"
                    else:
                        msg = f"Draw at {t:.2f}s"; winner = None
                    self._show_win_screen(msg, winner)
                return

            self.root.after(self.interval_ms, tick)

        self.root.after(self.interval_ms, tick)
        self.root.mainloop()

    def _show_win_screen(self, message: str, winner: Optional[str]):
        # Dark overlay + simple confetti
        self.canvas.delete("all")
        self.draw_bowl()
        self.canvas.create_rectangle(0, 0, self.w, self.h, fill="#000000", stipple="gray50")
        self.canvas.create_text(self.cx, 100, text="CONGRATULATIONS!", fill="#f6da55", font=("Consolas", 26, "bold"))
        self.canvas.create_text(self.cx, 140, text=message, fill="#e0e0e0", font=("Consolas", 16))
        import random
        rng = random.Random()
        chars = ["*", "+", "o", "#", "x"]
        colors = ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#f6da55", "#ffffff"]
        particles = []
        for _ in range(140):
            x = rng.randint(10, self.w - 10)
            y = rng.randint(-240, -20)
            vy = rng.uniform(1.8, 4.2)
            ch = rng.choice(chars)
            col = rng.choice(colors)
            particles.append([x, y, vy, ch, col])

        def animate():
            self.canvas.delete("confetti")
            alive = False
            for i in range(len(particles)):
                x, y, vy, ch, col = particles[i]
                y += vy
                particles[i][1] = y
                if y < self.h - 20:
                    alive = True
                self.canvas.create_text(int(x), int(y), text=ch, fill=col, font=("Consolas", 10, "bold"), tags="confetti")
            if alive:
                self.root.after(30, animate)
        animate()


def make_combo_from_args(args, reg: PartRegistry, prefix: str) -> Combo:
    metal = reg.metals[getattr(args, f"{prefix}_metal")]
    track = reg.tracks[getattr(args, f"{prefix}_track")]
    tip = reg.tips[getattr(args, f"{prefix}_tip")]
    lp = float(getattr(args, f"{prefix}_launch_power", 0.8))
    return Combo(name=None, metal=metal, track=track, tip=tip, launch_power=lp)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="3D stadium view (projected) for a single battle")
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
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--size", type=str, default="1000x700")
    p.add_argument("--no-ko", action="store_true")
    p.add_argument("--preset", default="default")
    p.add_argument("--preset-json", default=None, help="Advanced preset JSON (stadium/tips/bey/launch)")
    # Visual/dynamics convenience
    p.add_argument("--speed-boost", type=float, default=None, help="Multiply initial linear and spin launch speeds")
    p.add_argument("--trail-length", type=int, default=150, help="Number of trail points to keep (per bey)")
    p.add_argument("--bey-size-scale", type=float, default=0.8, help="Scale factor for rendered bey size")
    # Optional tuning overrides
    p.add_argument("--wall-tangent-damping", type=float, default=None)
    p.add_argument("--wall-spin-loss", type=float, default=None)
    p.add_argument("--wall-min-speed-keep", type=float, default=None)
    p.add_argument("--wall-reentry-radial", type=float, default=None)
    p.add_argument("--wall-restitution-boost", type=float, default=None)
    p.add_argument("--wall-restitution-vref", type=float, default=None)
    p.add_argument("--wall-torque-coupling", type=float, default=None)
    p.add_argument("--wall-tangent-min-keep", type=float, default=None)
    p.add_argument("--spin-loss-scale", type=float, default=None)
    p.add_argument("--lin-fric-scale", type=float, default=None)
    p.add_argument("--slope-scale", type=float, default=None)
    p.add_argument("--contact-loss-scale", type=float, default=None)
    p.add_argument("--launch-spin-scale", type=float, default=None)
    p.add_argument("--vel-launch-scale", type=float, default=None)
    p.add_argument("--min-spin", type=float, default=None)
    p.add_argument("--max-time", type=float, default=None)
    # Engagement tuning
    p.add_argument("--engage-attraction-gain", type=float, default=None)
    p.add_argument("--engage-attraction-falloff", type=float, default=None)
    p.add_argument("--engage-orbit-gain", type=float, default=None)
    args = p.parse_args(argv)

    # Resolve data paths relative to this script if not found in CWD
    parts_path = Path(args.parts)
    if not parts_path.exists():
        alt = Path(__file__).resolve().parent.parent / 'data' / 'parts.json'
        if alt.exists():
            parts_path = alt
    combos_path = Path(args.combos)
    if not combos_path.exists():
        altc = Path(__file__).resolve().parent.parent / 'data' / 'combos.json'
        if altc.exists():
            combos_path = altc

    reg = PartRegistry.from_json(parts_path)
    if args.combo1 and args.combo2:
        c1 = c2 = None
        if combos_path.exists():
            combos = load_combos(combos_path, reg)
            c1 = combos.get(args.combo1)
            c2 = combos.get(args.combo2)
        if c1 is None:
            c1 = resolve_combo_name(args.combo1, reg) or c1
        if c2 is None:
            c2 = resolve_combo_name(args.combo2, reg) or c2
        if c1 is None or c2 is None:
            raise SystemExit("Combo not found")
    else:
        required = [args.b1_metal, args.b1_track, args.b1_tip, args.b2_metal, args.b2_track, args.b2_tip]
        if any(x is None for x in required):
            raise SystemExit("Provide --combo1/--combo2 or all --b?-metal/track/tip")
        c1 = make_combo_from_args(args, reg, "b1")
        c2 = make_combo_from_args(args, reg, "b2")

    try:
        w_str, h_str = args.size.lower().split("x")
        width = int(w_str); height = int(h_str)
    except Exception:
        width, height = 1000, 700

    params = params_preset(args.preset)
    if args.no_ko:
        params.allow_ko = False
    if args.speed_boost is not None:
        params.vel_launch_scale = (params.vel_launch_scale or 1.0) * max(0.1, args.speed_boost)
        params.launch_spin_scale = (params.launch_spin_scale or 1.0) * max(0.1, args.speed_boost)
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
    if args.engage_attraction_gain is not None:
        params.engage_attraction_gain = args.engage_attraction_gain
    if args.engage_attraction_falloff is not None:
        params.engage_attraction_falloff = args.engage_attraction_falloff
    if args.engage_orbit_gain is not None:
        params.engage_orbit_gain = args.engage_orbit_gain

    stadium = bb10_default()
    # Apply advanced preset JSON before constructing renderer so bowl radii are correct visually
    if args.preset_json:
        from .advanced_preset import apply_advanced_preset
        c1, c2, params, stadium = apply_advanced_preset(args.preset_json, c1, c2, params, stadium)

    vis = Visual3D(width, height, args.fps, stadium, allow_ko=not args.no_ko, bey_size_scale=args.bey_size_scale, trail_length=max(40, int(args.trail_length)))
    vis.run(c1, c2, params=params, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
