from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import random
from typing import Optional

try:
    import tkinter as tk
except Exception as e:  # pragma: no cover
    tk = None

from .io import PartRegistry, load_combos, resolve_combo_name
from .parts import Combo
from .stadium import bb10_default, Stadium
from .physics import SimParams, simulate_duel_steps


class RTVisualizer:
    def __init__(self, width: int, height: int, fps: float, stadium: Stadium, allow_ko: bool = True, charge_launch: bool = False, seed: int | None = None, bey_size_scale: float = 1.0, trail_length: int = 120):
        if tk is None:
            raise RuntimeError("Tkinter is not available in this Python environment.")
        self.root = tk.Tk()
        self.root.title(" MFL Simulator — BB-10 Top-Down\)
        self.width = width
        self.height = height
        self.fps = max(5.0, fps)
        self.interval_ms = int(1000 / self.fps)
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#0b0b0b")
        self.canvas.pack()
        self.stadium = stadium
        # world radius in meters
        self.R = stadium.wall_radius_mm / 1000.0
        # scale to pixels, leave padding
        self.pad = 10
        self.scale = (min(self.width, self.height) - 2 * self.pad) / (2 * self.R)
        self.cx = self.width // 2
        self.cy = self.height // 2

        # Draw static stadium or image background
        self.bg_image = None
        self._draw_stadium()

        # Dynamic items
        self.beyA = None
        self.beyB = None
        self.trailA = []
        self.trailB = []
        self.max_trail = max(30, int(trail_length))
        self.bey_size_scale = max(0.3, min(2.0, bey_size_scale))
        self.text_info = self.canvas.create_text(
            self.width - 10,
            10,
            anchor="ne",
            fill="#e0e0e0",
            font=("Consolas", 10),
            text="",
        )
        self.result_text = None
        self.allow_ko = allow_ko
        self.charge_launch = charge_launch
        self.seed = seed
        self._rng = random.Random(seed)
        self.stepper = None
        self.launch_ready = (not charge_launch)
        self.charging = False
        self.charge_start = 0.0

        # Controls
        self.paused = False
        self.root.bind("<space>", lambda e: self._toggle_pause())
        self.root.bind("q", lambda e: self.root.destroy())
        # Charge controls: hold/release 'l' or 'L'
        self.root.bind("<KeyPress-l>", self._begin_charge)
        self.root.bind("<KeyRelease-l>", self._end_charge)

    def _begin_charge(self, e=None):
        if not self.charge_launch or self.launch_ready:
            return
        if not self.charging:
            self.charging = True
            import time as _t
            self.charge_start = _t.time()

    def _end_charge(self, e=None):
        if not self.charge_launch or self.launch_ready or not self.charging:
            return
        import time as _t
        dur = max(0.0, _t.time() - self.charge_start)
        self.charging = False
        # Map 0..2.0s -> 0.4..1.0
        max_dur = 2.0
        frac = max(0.0, min(1.0, dur / max_dur))
        self.lp_user = 0.4 + 0.6 * frac
        self.lp_other = 0.5 + 0.5 * self._rng.random()
        self.launch_ready = True

    def _toggle_pause(self):
        self.paused = not self.paused

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = int(self.cx + x * self.scale)
        sy = int(self.cy - y * self.scale)
        return sx, sy

    def _draw_stadium(self):
        # Try loading an image background if present
        try:
            img_path_png = Path("data/backgrounds/bb10.png")
            img_path_ppm = Path("data/backgrounds/bb10.ppm")
            img_path = img_path_png if img_path_png.exists() else img_path_ppm
            if img_path.exists():
                self.bg_image = tk.PhotoImage(file=str(img_path))
                self.canvas.create_image(self.cx, self.cy, image=self.bg_image)
        except Exception:
            self.bg_image = None

        rpx = self.R * self.scale
        # Main rim
        self.canvas.create_oval(
            self.cx - rpx,
            self.cy - rpx,
            self.cx + rpx,
            self.cy + rpx,
            outline="#555",
            width=2,
        )
        # Concentric guide rings to suggest slope/ridge
        for frac, col in [(0.85, "#2e2e2e"), (0.65, "#262626"), (0.45, "#1f1f1f")]:
            rr = rpx * frac
            self.canvas.create_oval(self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr, outline=col)
        # Three pockets (approx) at 0, 120, 240 degrees
        for ang_deg in (0, 120, 240):
            ang = math.radians(ang_deg)
            px = self.cx + (rpx - 6) * math.cos(ang)
            py = self.cy - (rpx - 6) * math.sin(ang)
            ex = self.cx + (rpx + 12) * math.cos(ang)
            ey = self.cy - (rpx + 12) * math.sin(ang)
            self.canvas.create_line(px, py, ex, ey, fill="#8a7a44", width=3)
        # Center mark
        self.canvas.create_oval(self.cx - 3, self.cy - 3, self.cx + 3, self.cy + 3, fill="#404040", outline="")

    def _draw_bey(self, x: float, y: float, r_m: float, color: str, alive: bool, ko: bool):
        sx, sy = self.world_to_screen(x, y)
        rpx = max(2, int(r_m * self.scale * self.bey_size_scale))
        outline = color
        fill = color if alive else ""
        if ko:
            outline = "#ffcc00"
        return self.canvas.create_oval(sx - rpx, sy - rpx, sx + rpx, sy + rpx, outline=outline, width=2, fill=fill, stipple="gray50" if not alive else "")

    def _blend_hex(self, fg_hex: str, bg_hex: str, alpha: float) -> str:
        # alpha in [0,1], returns hex string simulating glow via blending
        def _hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        def _rgb_to_hex(rgb):
            return '#%02x%02x%02x' % rgb
        fr, fg, fb = _hex_to_rgb(fg_hex)
        br, bgc, bb = _hex_to_rgb(bg_hex)
        cr = int(br * (1 - alpha) + fr * alpha)
        cg = int(bgc * (1 - alpha) + fg * alpha)
        cb = int(bb * (1 - alpha) + fb * alpha)
        return _rgb_to_hex((max(0, min(255, cr)), max(0, min(255, cg)), max(0, min(255, cb))))

    def _draw_trail(self, trail, base_color: str, glow_color: str = None):
        # draw fading, glowing trail segments
        if len(trail) < 2:
            return
        bg = "#0b0b0b"
        n = len(trail)
        for i in range(1, n):
            (x0, y0) = trail[i - 1]
            (x1, y1) = trail[i]
            sx0, sy0 = self.world_to_screen(x0, y0)
            sx1, sy1 = self.world_to_screen(x1, y1)
            age = i / n
            # stronger at newer segments
            alpha = max(0.0, 1.0 - age)
            alpha = alpha * alpha * 0.9  # quadratic falloff, capped
            color_line = self._blend_hex(glow_color or base_color, bg, alpha)
            self.canvas.create_line(sx0, sy0, sx1, sy1, fill=color_line, width=1)

    def _update_info(self, t: float, a, b):
        vA = math.hypot(a.vel[0], a.vel[1])
        vB = math.hypot(b.vel[0], b.vel[1])
        lines = [
            f"t={t:5.2f}s  fps~{self.fps:.0f}",
            f"A: {a.combo.metal.name} {a.combo.track.name}{a.combo.tip.name}  spin {a.spin:6.1f}  v {vA:4.2f}  {'KO' if a.ko else ('Alive' if a.alive else 'Stopped')}",
            f"B: {b.combo.metal.name} {b.combo.track.name}{b.combo.tip.name}  spin {b.spin:6.1f}  v {vB:4.2f}  {'KO' if b.ko else ('Alive' if b.alive else 'Stopped')}",
            "[space]=pause  q=quit",
        ]
        self.canvas.itemconfigure(self.text_info, text="\n".join(lines))

    def run(self, c1: Combo, c2: Combo, seed: Optional[int] = None, params: Optional[SimParams] = None):
        params = params or SimParams()
        params.allow_ko = self.allow_ko if params is None or params.allow_ko != False else params.allow_ko
        stepper = None

        last_time = time.time()
        next_frame = 0

        def tick():
            nonlocal last_time, next_frame
            if self.paused:
                self.root.after(self.interval_ms, tick)
                return
            # Wait for user launch if charge mode
            if self.charge_launch and not self.launch_ready:
                # Draw static stadium + instruction
                self.canvas.delete("all")
                self._draw_stadium()
                self.canvas.create_text(self.cx, 40, text="Hold 'L' to charge Launch A", fill="#f0f0f0", font=("Consolas", 16, "bold"))
                self.canvas.create_text(self.cx, 64, text="Release to start battle", fill="#c0c0c0", font=("Consolas", 12))
                self.root.after(self.interval_ms, tick)
                return

            nonlocal stepper
            if stepper is None:
                # Set launch powers
                if self.charge_launch:
                    from .parts import Combo as _Combo
                    c1_use = _Combo(name=c1.name, metal=c1.metal, track=c1.track, tip=c1.tip, launch_power=self.lp_user, mass_g=c1.mass_g, radius_mm=c1.radius_mm)
                    c2_use = _Combo(name=c2.name, metal=c2.metal, track=c2.track, tip=c2.tip, launch_power=self.lp_other, mass_g=c2.mass_g, radius_mm=c2.radius_mm)
                else:
                    c1_use = c1; c2_use = c2
                stepper = simulate_duel_steps(c1_use, c2_use, self.stadium, params, seed=seed)
            try:
                frame = next(stepper)
            except StopIteration:
                # Draw final frame again to ensure result visible
                return

            self.canvas.delete("dyn")

            a = frame["a"]
            b = frame["b"]
            t = frame["t"]

            # Append trails
            self.trailA.append((a.pos[0], a.pos[1]))
            self.trailB.append((b.pos[0], b.pos[1]))
            if len(self.trailA) > self.max_trail:
                self.trailA.pop(0)
            if len(self.trailB) > self.max_trail:
                self.trailB.pop(0)

            # Draw trails (glow effects)
            self._draw_trail(self.trailA, base_color="#ff4444", glow_color="#ff6666")
            self._draw_trail(self.trailB, base_color="#4488ff", glow_color="#66aaff")

            # Draw beys
            self._draw_bey(a.pos[0], a.pos[1], a.radius(), "#ff3333", a.alive, a.ko)
            self._draw_bey(b.pos[0], b.pos[1], b.radius(), "#3388ff", b.alive, b.ko)

            # Info overlay
            self._update_info(t, a, b)

            # Check end condition and show win screen
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

            # Outcome text
            if (self.allow_ko and ((a.ko and not b.ko) or (b.ko and not a.ko))) or ((not a.alive) ^ (not b.alive)):
                if self.result_text is None:
                    if self.allow_ko and a.ko and not b.ko:
                        msg = f"B wins by KO at {t:.2f}s"
                    elif self.allow_ko and b.ko and not a.ko:
                        msg = f"A wins by KO at {t:.2f}s"
                    elif a.alive and not b.alive:
                        msg = f"A wins by sleep-out at {t:.2f}s"
                    elif b.alive and not a.alive:
                        msg = f"B wins by sleep-out at {t:.2f}s"
                    else:
                        msg = f"Draw at {t:.2f}s"
                    self.result_text = self.canvas.create_text(
                        self.cx,
                        24,
                        text=msg,
                        fill="#f0f0f0",
                        font=("Consolas", 16, "bold"),
                    )
            self.root.after(self.interval_ms, tick)

        self.root.after(self.interval_ms, tick)
        self.root.mainloop()

    def _show_win_screen(self, message: str, winner: Optional[str]):
        # Overlay and ASCII-like confetti animation drawn on canvas
        self.canvas.delete("all")
        self._draw_stadium()
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill="#000000", stipple="gray50")
        self.canvas.create_text(self.cx, 100, text="CONGRATULATIONS!", fill="#f6da55", font=("Consolas", 26, "bold"))
        self.canvas.create_text(self.cx, 140, text=message, fill="#e0e0e0", font=("Consolas", 16))
        rng = random.Random()
        chars = ["*", "+", "o", "#", "x"]
        colors = ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#f6da55", "#ffffff"]
        particles = []
        for _ in range(140):
            x = rng.randint(10, self.width - 10)
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
                if y < self.height - 20:
                    alive = True
                self.canvas.create_text(int(x), int(y), text=ch, fill=col, font=("Consolas", 10, "bold"), tags="confetti")
            if alive:
                self.root.after(30, animate)
        animate()


def build_combo_from_args(args, reg: PartRegistry, prefix: str) -> Combo:
    metal = reg.metals[getattr(args, f"{prefix}_metal")]
    track = reg.tracks[getattr(args, f"{prefix}_track")]
    tip = reg.tips[getattr(args, f"{prefix}_tip")]
    lp = float(getattr(args, f"{prefix}_launch_power", 0.8))
    return Combo(name=None, metal=metal, track=track, tip=tip, launch_power=lp)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Real-time 2D visualization (Tkinter) for a single MFL battle")
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
    p.add_argument("--fps", type=float, default=60.0, help="Render frames per second")
    p.add_argument("--size", type=str, default="900x700", help="Window size WxH in pixels")
    p.add_argument("--no-ko", action="store_true", help="Disable KOs; sleep-out only")
    p.add_argument("--charge-launch", action="store_true", help="Enable charged launch for A (hold 'L')")
    p.add_argument("--preset", default="default", help="Tuning preset (e.g., stamina-long)")
    p.add_argument("--preset-json", default=None, help="Advanced preset JSON (stadium/tips/bey/launch)")
    # Visual and dynamics quality-of-life
    p.add_argument("--speed-boost", type=float, default=None, help="Multiply initial linear and spin launch speeds")
    p.add_argument("--trail-length", type=int, default=150, help="Number of trail points to keep (per bey)")
    p.add_argument("--bey-size-scale", type=float, default=0.8, help="Scale factor for rendered bey size")
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

    args = p.parse_args(argv)
    reg = PartRegistry.from_json(args.parts)

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
        c1 = build_combo_from_args(args, reg, "b1")
        c2 = build_combo_from_args(args, reg, "b2")

    try:
        w_str, h_str = args.size.lower().split("x")
        width = int(w_str)
        height = int(h_str)
    except Exception:
        width, height = 900, 700

    stadium = bb10_default()
    vis = RTVisualizer(width=width, height=height, fps=args.fps, stadium=stadium, allow_ko=(not args.no_ko), charge_launch=args.charge_launch, seed=args.seed, bey_size_scale=args.bey_size_scale, trail_length=args.trail_length)
    # Build params from preset
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
    # Run with explicit params
    # Apply advanced preset JSON
    if args.preset_json:
        from .advanced_preset import apply_advanced_preset
        c1, c2, params, stadium = apply_advanced_preset(args.preset_json, c1, c2, params, stadium)
        vis.stadium = stadium
    vis.run(c1, c2, seed=args.seed, params=params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
