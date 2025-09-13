from __future__ import annotations

try:
    from .physics import SimParams
except Exception:
    import os, sys as _sys
    _base = os.path.dirname(__file__)
    if _base not in _sys.path:
        _sys.path.insert(0, _base)
    from physics import SimParams


def params_preset(name: str | None) -> SimParams:
    """Return a SimParams configured by named preset.

    Presets:
    - default: current defaults from physics
    - stamina-long: target ~4-5 minute stamina endurance on average
    """
    p = SimParams()
    if not name or name == "default":
        return p
    n = name.lower().strip()
    if n == "stamina-long":
        # Reduce linear + spin losses, soften slope, extend time, disable KO
        p.max_time = 900.0
        p.min_spin = 0.8
        p.spin_loss_scale = 0.05
        p.lin_fric_scale = 0.05
        p.slope_scale = 0.5
        p.contact_loss_scale = 0.05
        p.vel_launch_scale = 0.7
        p.random_spin_jitter = 0.005
        p.allow_ko = False
        # Slightly reduce collision energy transfer to avoid rapid decays
        p.restitution = 0.15
        # Keep launch spin comparable
        p.launch_spin_scale = 1.0
        return p
    if n == "stamina-2min":
        # Target around ~2:00 battles on average (no KO)
        p.max_time = 300.0
        p.min_spin = 1.0
        p.spin_loss_scale = 0.02
        p.lin_fric_scale = 0.02
        p.contact_loss_scale = 0.02
        p.slope_scale = 0.5
        p.vel_launch_scale = 0.8
        p.random_spin_jitter = 0.006
        p.allow_ko = False
        p.restitution = 0.18
        p.launch_spin_scale = 1.0
        # Walls keep motion but don't add too much energy
        p.wall_tangent_damping = 0.992
        p.wall_spin_loss = 0.2
        p.wall_min_speed_keep = 0.97
        p.wall_tangent_min_keep = 0.97
        p.wall_reentry_radial = 0.22
        p.wall_restitution_boost = 0.10
        p.wall_restitution_vref = 2.0
        p.wall_torque_coupling = 0.4
        return p
    if n == "stamina-extreme":
        # Very long endurance, near-conservative dynamics
        p.max_time = 1800.0
        p.min_spin = 0.4
        p.spin_loss_scale = 0.005
        p.lin_fric_scale = 0.005
        p.contact_loss_scale = 0.005
        p.slope_scale = 0.3
        p.vel_launch_scale = 0.5
        p.random_spin_jitter = 0.002
        p.allow_ko = False
        p.restitution = 0.12
        p.launch_spin_scale = 1.0
        # Walls preserve momentum strongly
        p.wall_tangent_damping = 0.999
        p.wall_spin_loss = 0.05
        p.wall_min_speed_keep = 0.995
        return p
    # Unknown preset falls back to default
    return p
