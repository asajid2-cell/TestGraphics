from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

try:
    from .parts import Combo
    from .stadium import Stadium
except Exception:
    import os, sys as _sys
    _base = os.path.dirname(__file__)
    if _base not in _sys.path:
        _sys.path.insert(0, _base)
    from parts import Combo
    from stadium import Stadium


Vec = Tuple[float, float]


def v_add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def v_mul(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s)


def v_len(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def v_norm(a: Vec) -> Vec:
    l = v_len(a)
    if l == 0:
        return (0.0, 0.0)
    return (a[0] / l, a[1] / l)


@dataclass
class RigidTop:
    combo: Combo
    pos: Vec
    vel: Vec
    spin: float  # rad/s (magnitude)
    spin_dir: int = 1  # +1 right, -1 left
    alive: bool = True
    ko: bool = False
    flower_gain: float = 0.0

    def mass(self) -> float:
        # grams -> kilograms
        return self.combo.total_mass_g() / 1000.0

    def radius(self) -> float:
        return self.combo.effective_radius_mm() / 1000.0

    def moi(self) -> float:
        # Use override if provided, else approximate as solid disk: (1/2) m r^2
        if self.combo.moi_override is not None:
            return self.combo.moi_override
        m = self.mass()
        r = self.radius()
        return 0.5 * m * r * r


@dataclass
class SimParams:
    dt: float = 0.004  # seconds per step
    max_time: float = 60.0  # max sim time seconds
    restitution: float = 0.25  # inelasticity for bey-bey collisions
    base_contact_loss: float = 0.003  # base spin loss per contact (rads)
    random_spin_jitter: float = 0.03  # random noise applied to spin loss/gain
    ko_speed_threshold: float = 1.6  # m/s outward speed to likely KO at rim
    min_spin: float = 3.0  # rad/s treated as stopped
    allow_ko: bool = True  # if False, disable KO; only sleep-outs/time decide
    # Global scaling knobs for calibration
    spin_loss_scale: float = 1.0    # scale rotational spin-down
    lin_fric_scale: float = 1.0     # scale linear kinetic friction
    slope_scale: float = 1.15       # scale bowl slope pull toward center (drift inward)
    launch_spin_scale: float = 1.0  # scale initial spin imparted at launch
    contact_loss_scale: float = 1.0 # scale spin loss on collisions
    vel_launch_scale: float = 1.0   # scale initial linear launch speed
    # Wall behavior
    wall_tangent_damping: float = 0.99  # closer to no tangential loss
    wall_spin_loss: float = 0.15        # much lower spin loss on wall contact (rads)
    wall_min_speed_keep: float = 0.95   # keep most of pre-bounce speed when KOs are off
    wall_tangent_min_keep: float = 0.95 # keep more tangential speed on bounce
    wall_reentry_radial: float = 0.35   # stronger inward nudge after bounce (KO off)
    wall_restitution_boost: float = 0.12 # extra restitution scaled by impact normal speed (KO off)
    wall_restitution_vref: float = 2.5   # m/s reference for restitution boost
    wall_torque_coupling: float = 0.3    # lower shear->spin coupling (less stamina loss on walls)
    # Engagement steering (encourage battling near center)
    engage_attraction_gain: float = 0.8   # base gain toward opponent (m/s^2)
    engage_attraction_falloff: float = 0.6 # 0..inf, higher = weaker at long distance
    engage_orbit_gain: float = 0.25       # small tangential bias around center (m/s^2)


def _apply_engagement(a: RigidTop, b: RigidTop, stadium: Stadium, params: SimParams, dt: float):
    # Steer toward each other with distance falloff
    ab = v_sub(b.pos, a.pos)
    d = v_len(ab)
    if d > 1e-6:
        abn = v_mul(ab, 1.0 / d)
        r = stadium.radius_mm / 1000.0
        fall = 1.0 / (1.0 + (d / max(1e-6, 0.6 * r)) ** (1.0 + params.engage_attraction_falloff))
        acc = params.engage_attraction_gain * fall
        a.vel = v_add(a.vel, v_mul(abn, acc * dt))
        b.vel = v_sub(b.vel, v_mul(abn, acc * dt))
    # Add slight orbiting around center to avoid straight lines
    for top in (a, b):
        rc = v_mul(top.pos, -1.0)
        rl = v_len(rc)
        if rl > 1e-6:
            rcn = v_mul(rc, 1.0 / rl)
            tperp = (-rcn[1], rcn[0])
            top.vel = v_add(top.vel, v_mul(tperp, params.engage_orbit_gain * dt))


def launch_state(combo: Combo, stadium: Stadium, rng: random.Random) -> RigidTop:
    # Randomize launch position near rim and direction inward
    r = stadium.radius_mm / 1000.0 * (0.8 + 0.15 * rng.random())
    theta = 2 * math.pi * rng.random()
    pos = (r * math.cos(theta), r * math.sin(theta))

    # Launch velocity magnitude scaled by launch power and tip friction
    lp = max(0.0, min(1.0, combo.launch_power))
    tip = combo.tip
    base_v = 2.0 + 1.0 * lp
    # Rubber tips trade top-speed for grip; plastics slide more
    speed_scale = max(0.6, 1.1 - 0.6 * tip.mu_kinetic)
    vmag = base_v * speed_scale

    dir_to_center = v_norm(v_mul(pos, -1.0))
    # Banked launch: add tangential component based on bank angle
    tang = (-dir_to_center[1], dir_to_center[0])
    bank = math.radians(combo.bank_deg or 0.0)
    w_in = math.cos(bank)
    w_tan = math.sin(bank)
    dir0 = v_norm((dir_to_center[0] * w_in + tang[0] * w_tan,
                   dir_to_center[1] * w_in + tang[1] * w_tan))
    vel = v_mul(dir0, vmag)

    # Initial spin proportional to launch power and stamina tendency
    spin_base = 220.0 + 120.0 * lp
    # initial spin scaled by preset
    spin = spin_base * (0.9 + 0.2 * combo.stats()["stamina"])  # rad/s
    spin_dir = -1 if combo.metal.left_spin else 1
    flower_gain = combo.flower_arc_gain or 0.0

    return RigidTop(combo=combo, pos=pos, vel=vel, spin=spin, spin_dir=spin_dir, flower_gain=flower_gain)


def apply_floor_friction(top: RigidTop, params: SimParams, stadium: Stadium, rng: random.Random) -> None:
    # Kinetic friction reduces linear speed; spin friction reduces spin
    mu_k = top.combo.tip.mu_kinetic
    m = top.mass()
    g = 9.81
    f = mu_k * m * g * params.lin_fric_scale
    v = v_len(top.vel)
    if v > 0:
        decel = f / m
        new_v = max(0.0, v - decel * params.dt)
        if new_v == 0:
            top.vel = (0.0, 0.0)
        else:
            top.vel = v_mul(v_norm(top.vel), new_v)

    # Spin friction and stability + stamina interplay
    stamina_factor = 1.0 - 0.3 * top.combo.stats().get("stamina", 0.0)
    spin_loss = top.combo.tip.spin_friction * (1.0 + 0.4 * (1.0 - top.combo.tip.stability)) * stamina_factor
    spin_loss *= (1.0 + 0.05 * (v / 2.0))  # moving more costs a bit more spin
    spin_loss *= params.spin_loss_scale
    # Life After Death: reduce loss when wobbling/low-spin
    if top.spin < 30.0:
        spin_loss *= max(0.6, 1.0 - 0.5 * top.combo.tip.lad)
    top.spin = max(0.0, top.spin - spin_loss * params.dt * 60.0)  # scale to per-second


def apply_bowl_slope(top: RigidTop, stadium: Stadium, params: SimParams) -> None:
    # Accelerate towards center due to bowl slope; adds centripetal attraction
    r = v_len(top.pos)
    if r == 0:
        return
    slope_accel = 1.2 * stadium.slope * params.slope_scale  # tuned constant
    dir_in = v_mul(v_norm(top.pos), -1.0)
    top.vel = v_add(top.vel, v_mul(dir_in, slope_accel * params.dt))


def wall_interaction(top: RigidTop, stadium: Stadium, params: SimParams, rng: random.Random) -> None:
    # KO if beyond wall radius with sufficient outward momentum
    r = v_len(top.pos)
    wall_r = stadium.wall_radius_mm / 1000.0
    if r >= wall_r:
        v = v_len(top.vel)
        radial_speed = 0.0
        if v > 0:
            radial_speed = (top.pos[0] * top.vel[0] + top.pos[1] * top.vel[1]) / (r * v)
            radial_speed = max(-1.0, min(1.0, radial_speed))
            radial_speed *= v
        if params.allow_ko and radial_speed > params.ko_speed_threshold:
            top.ko = True
            top.alive = False
            return
        # Bounce back with some loss
        normal = v_norm(top.pos)
        # Decompose velocity
        vn = normal[0] * top.vel[0] + normal[1] * top.vel[1]
        vt_vec = v_sub(top.vel, v_mul(normal, vn))
        pre_v = v_len(top.vel)
        pre_vt_len = v_len(vt_vec)

        # Dynamic restitution when KO is disabled (bouncier at higher impact speeds)
        e = stadium.wall_restitution
        if not params.allow_ko:
            boost = params.wall_restitution_boost * min(1.0, abs(vn) / max(1e-6, params.wall_restitution_vref))
            e = min(0.98, e + boost)

        # Reflect normal; damp tangent but keep a minimum fraction
        vt_damped = v_mul(vt_vec, params.wall_tangent_damping)
        vt_len = v_len(vt_damped)
        min_keep = params.wall_tangent_min_keep * pre_vt_len
        if vt_len < min_keep and pre_vt_len > 1e-9:
            vt_damped = v_mul(vt_vec if pre_vt_len == 0 else v_mul(vt_vec, 1.0 / pre_vt_len), min_keep)

        top.vel = v_add(v_mul(normal, -vn * e), vt_damped)
        # Small spin loss due to wall scrape (tunable)
        # Spin loss: base plus coupling from tangential shear (how much vt was reduced)
        wall_spin_loss = params.wall_spin_loss * (0.3 if not params.allow_ko else 1.0)
        vt_reduction = max(0.0, pre_vt_len - v_len(vt_damped))
        extra_spin = params.wall_torque_coupling * vt_reduction
        top.spin = max(0.0, top.spin - (wall_spin_loss + extra_spin))

        # If KOs are disabled, ensure we keep a minimum portion of pre-bounce speed
        if not params.allow_ko:
            # Ensure minimal inward radial component to avoid sticking and keep flow
            rad = top.vel[0] * normal[0] + top.vel[1] * normal[1]
            if rad > -params.wall_reentry_radial:
                top.vel = v_add(top.vel, v_mul(normal, -(params.wall_reentry_radial + rad)))
            # Ensure overall momentum is not killed
            post_v = v_len(top.vel)
            target = pre_v * max(0.0, min(1.0, params.wall_min_speed_keep))
            if post_v < target:
                scale = target / max(1e-9, post_v)
                top.vel = v_mul(top.vel, scale)
        # Clamp position back just inside the wall to prevent escaping when KOs are disabled
        clearance = max(0.0015, min(0.004, top.radius()))  # 1.5mm..4mm
        limit = max(0.0, wall_r - clearance)
        top.pos = v_mul(normal, limit)


def resolve_collision(a: RigidTop, b: RigidTop, params: SimParams, rng: random.Random) -> None:
    # Simple inelastic collision with recoil modifier
    delta = v_sub(b.pos, a.pos)
    dist = v_len(delta)
    if dist == 0:
        return
    ra = a.radius()
    rb = b.radius()
    if dist > (ra + rb) * 0.98:  # slightly lenient
        return

    normal = v_norm(delta)
    rel_v = v_sub(b.vel, a.vel)
    vel_along_normal = rel_v[0] * normal[0] + rel_v[1] * normal[1]
    if vel_along_normal > 0:
        return

    restitution = params.restitution
    # Recoil and attack affect impulse magnitude
    a_stat = a.combo.stats()["attack"]
    b_stat = b.combo.stats()["attack"]
    a_recoil = a.combo.stats()["recoil"]
    b_recoil = b.combo.stats()["recoil"]
    mod = 1.0 + 0.6 * (a_stat + b_stat) + 0.3 * (a_recoil + b_recoil)
    # Defense reduces effective impulse transfer
    a_def = a.combo.stats()["defense"]
    b_def = b.combo.stats()["defense"]
    def_scale = max(0.6, 1.0 - 0.4 * ((a_def + b_def) * 0.5))
    mod *= def_scale

    inv_mass_a = 1.0 / a.mass()
    inv_mass_b = 1.0 / b.mass()
    # Defense behaves like added effective inertia against linear impulses
    inv_mass_a_eff = inv_mass_a / (1.0 + 0.8 * a_def)
    inv_mass_b_eff = inv_mass_b / (1.0 + 0.8 * b_def)
    j = -(1 + restitution) * vel_along_normal
    j /= (inv_mass_a_eff + inv_mass_b_eff)
    j *= mod

    impulse = v_mul(normal, j)
    a.vel = v_sub(a.vel, v_mul(impulse, inv_mass_a_eff))
    b.vel = v_add(b.vel, v_mul(impulse, inv_mass_b_eff))

    # Spin losses upon contact (baseline)
    loss_a = params.base_contact_loss * (1.0 + 0.8 * b_stat + 0.4 * b_recoil)
    loss_b = params.base_contact_loss * (1.0 + 0.8 * a_stat + 0.4 * a_recoil)

    # Destabilization due to height differences and low stability tips
    h_a = a.combo.track.height_mm
    h_b = b.combo.track.height_mm
    dh = abs(h_a - h_b)
    if dh > 0:
        destab_scale = min(1.4, 1.0 + 0.04 * dh)  # up to ~1.4x around 10mm diff
        if h_a < h_b:
            loss_a *= destab_scale * (1.0 + 0.2 * (1.0 - a.combo.tip.stability))
        else:
            loss_b *= destab_scale * (1.0 + 0.2 * (1.0 - b.combo.tip.stability))

    # Spin equalization for opposite-spin contacts
    if a.spin_dir != b.spin_dir:
        eq_a = a.combo.stats().get("spin_eq", 0.0)
        eq_b = b.combo.stats().get("spin_eq", 0.0)
        grip = 0.5 * (a.combo.tip.mu_static + b.combo.tip.mu_static)
        eq_coeff = 0.25 * (eq_a + eq_b) * (0.6 + 0.8 * grip)
        if a.spin > b.spin:
            delta = eq_coeff * (a.spin - b.spin)
            a.spin = max(0.0, a.spin - delta)
            b.spin = b.spin + delta
        elif b.spin > a.spin:
            delta = eq_coeff * (b.spin - a.spin)
            b.spin = max(0.0, b.spin - delta)
            a.spin = a.spin + delta

    # Apply global scales and randomness
    scale = params.spin_loss_scale * params.contact_loss_scale
    loss_a *= scale * (1.0 + params.random_spin_jitter * (rng.random() * 2 - 1))
    loss_b *= scale * (1.0 + params.random_spin_jitter * (rng.random() * 2 - 1))
    a.spin = max(0.0, a.spin - loss_a)
    b.spin = max(0.0, b.spin - loss_b)


def step(top: RigidTop, stadium: Stadium, params: SimParams, rng: random.Random) -> None:
    if not top.alive:
        return
    # Integrate position
    top.pos = v_add(top.pos, v_mul(top.vel, params.dt))
    # Apply flower curvature (steering) for attackers
    if top.flower_gain != 0.0:
        vmag = v_len(top.vel)
        if vmag > 1e-6:
            perp = (-top.vel[1] / vmag, top.vel[0] / vmag)
            # curvature proportional to speed
            k = top.flower_gain
            top.vel = v_add(top.vel, v_mul(perp, k * params.dt))
    apply_bowl_slope(top, stadium, params)
    apply_floor_friction(top, params, stadium, rng)
    wall_interaction(top, stadium, params, rng)
    # Effective stop threshold reduced by LAD (Life After Death)
    min_spin_eff = params.min_spin * (1.0 - 0.5 * top.combo.tip.lad)
    # Sleep-out mechanics: if spin is very low OR both spin low-ish and linear speed tiny, treat as stopped
    vmag = v_len(top.vel)
    if top.spin <= min_spin_eff or (top.spin <= 1.2 * min_spin_eff and vmag <= 0.05):
        top.alive = False


def simulate_duel(a_combo: Combo, b_combo: Combo, stadium: Stadium, params: SimParams, seed: int | None = None):
    rng = random.Random(seed)
    a = launch_state(a_combo, stadium, rng)
    b = launch_state(b_combo, stadium, rng)

    time = 0.0
    last_contact_t = -999.0
    while time < params.max_time and (a.alive or b.alive):
        # Engagement steering then collision check
        _apply_engagement(a, b, stadium, params, params.dt)
        # Collision check before moving too far
        resolve_collision(a, b, params, rng)
        step(a, stadium, params, rng)
        step(b, stadium, params, rng)
        time += params.dt

        # Prevent tunneling: another quick collision check
        resolve_collision(a, b, params, rng)

        if not a.alive and not b.alive:
            # Both stopped roughly together — tie
            break

    # Determine outcome
    if params.allow_ko:
        if a.ko and not b.ko:
            return {"winner": "B", "method": "KO", "time": time}
        if b.ko and not a.ko:
            return {"winner": "A", "method": "KO", "time": time}
    if a.alive and not b.alive:
        return {"winner": "A", "method": "SO", "time": time}
    if b.alive and not a.alive:
        return {"winner": "B", "method": "SO", "time": time}
    # If both alive at max_time, pick higher spin as stamina win
    if time >= params.max_time:
        if a.spin > b.spin:
            return {"winner": "A", "method": "Time", "time": time}
        elif b.spin > a.spin:
            return {"winner": "B", "method": "Time", "time": time}
        else:
            return {"winner": None, "method": "Draw", "time": time}
    # Both stopped
    return {"winner": None, "method": "Draw", "time": time}


def simulate_duel_steps(a_combo: Combo, b_combo: Combo, stadium: Stadium, params: SimParams, seed: int | None = None):
    """Generator yielding per-step state for visualization.
    Yields dicts with keys: t, a, b. Each of a/b is a RigidTop reference.
    """
    rng = random.Random(seed)
    a = launch_state(a_combo, stadium, rng)
    b = launch_state(b_combo, stadium, rng)
    if params.launch_spin_scale != 1.0:
        a.spin *= params.launch_spin_scale
        b.spin *= params.launch_spin_scale
    if params.vel_launch_scale != 1.0:
        a.vel = v_mul(a.vel, params.vel_launch_scale)
        b.vel = v_mul(b.vel, params.vel_launch_scale)
    t = 0.0
    yield {"t": t, "a": a, "b": b}
    while t < params.max_time and (a.alive or b.alive):
        _apply_engagement(a, b, stadium, params, params.dt)
        resolve_collision(a, b, params, rng)
        step(a, stadium, params, rng)
        step(b, stadium, params, rng)
        t += params.dt
        resolve_collision(a, b, params, rng)
        yield {"t": t, "a": a, "b": b}
