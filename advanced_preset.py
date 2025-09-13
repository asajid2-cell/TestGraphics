from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from .physics import SimParams
from .stadium import Stadium
from .parts import Combo, Tip


def apply_advanced_preset(path: str | Path, combo1: Combo, combo2: Combo, params: SimParams, stadium: Stadium) -> Tuple[Combo, Combo, SimParams, Stadium]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # Stadium mapping (Stadium is frozen; construct a new instance with overrides)
    st = data.get("stadium", {})
    dish_r = st.get("dish_radius_m")
    ridge_r = st.get("ridge_radius_m")
    slope_deg = st.get("dish_slope_deg")
    wall_rest = st.get("wall_restitution")

    new_radius_mm = float(ridge_r) * 1000.0 if ridge_r is not None else stadium.radius_mm
    new_wall_radius_mm = float(dish_r) * 1000.0 if dish_r is not None else stadium.wall_radius_mm
    # Map degrees (approx) to our 0..1 slope factor
    if slope_deg is not None:
        slope_factor = max(0.05, min(1.0, float(slope_deg) / 45.0))
    else:
        slope_factor = stadium.slope
    new_wall_restitution = float(wall_rest) if wall_rest is not None else stadium.wall_restitution

    from .stadium import Stadium as _Stadium
    stadium = _Stadium(
        name=stadium.name,
        radius_mm=new_radius_mm,
        wall_radius_mm=new_wall_radius_mm,
        slope=slope_factor,
        wall_restitution=new_wall_restitution,
        center_pit_radius_mm=stadium.center_pit_radius_mm,
    )
    if "wall_tangent_damping" in st:
        params.wall_tangent_damping = float(st["wall_tangent_damping"])

    # Global friction (optional; we mostly use tip frictions)
    fr = data.get("friction", {})
    # not directly used; keep for future floor friction tuning

    # Collision mapping
    coll = data.get("collision", {})
    if "bey_restitution" in coll:
        params.restitution = float(coll["bey_restitution"])
    if "tangential_energy_loss" in coll:
        # map to contact loss scale
        params.contact_loss_scale = max(0.01, min(1.0, float(coll["tangential_energy_loss"])) )

    # Tips mapping (override tip frictions by name; we match by combo tip name if present in mapping)
    tips = data.get("tips", {})
    def override_tip(c: Combo):
        key = None
        # Prefer exact mapping in config by a logical alias (e.g., "CS_worn" if combo.tip.shape=="CS")
        if c.tip and c.tip.shape in tips:
            key = c.tip.shape
        for k in tips.keys():
            if k.lower() in (c.tip.name or "").lower():
                key = k
                break
        if key:
            t = tips[key]
            c.tip = Tip(
                name=c.tip.name,
                mu_static=float(t.get("mu_s", c.tip.mu_static)),
                mu_kinetic=float(t.get("mu_k", c.tip.mu_kinetic)),
                spin_friction=c.tip.spin_friction,
                stability=c.tip.stability,
                shape=c.tip.shape,
                lad=c.tip.lad,
            )
    override_tip(combo1)
    override_tip(combo2)

    # Bey-specific overrides (mass, radius, moi, recoil, bank)
    bstats = data.get("bey_stats", {})
    def match_and_apply(c: Combo):
        # Try direct name; if not, search partial
        key = None
        if c.name and c.name in bstats:
            key = c.name
        else:
            for k in bstats.keys():
                if c.name and k.lower() in c.name.lower():
                    key = k
                    break
                # try metal + track + tip
                label = f"{c.metal.name}{c.track.name}{c.tip.shape}".lower()
                if k.lower() in label:
                    key = k
                    break
        if key:
            b = bstats[key]
            if "mass_g" in b:
                c.mass_g = float(b["mass_g"])
            if "wheel_outer_radius_m" in b:
                c.radius_mm = float(b["wheel_outer_radius_m"]) * 1000.0
            if "wheel_inertia_kgm2" in b:
                c.moi_override = float(b["wheel_inertia_kgm2"])
            if "recoil_factor" in b:
                # map to metal recoil weighting
                c.metal = type(c.metal)(
                    name=c.metal.name,
                    mass_g=c.metal.mass_g,
                    radius_mm=c.metal.radius_mm,
                    attack=c.metal.attack,
                    defense=c.metal.defense,
                    stamina=c.metal.stamina,
                    recoil=float(b["recoil_factor"]),
                    left_spin=c.metal.left_spin,
                    spin_eq=c.metal.spin_eq,
                )
            if "tip" in b and b["tip"] in tips:
                t = tips[b["tip"]]
                c.tip = Tip(
                    name=c.tip.name,
                    mu_static=float(t.get("mu_s", c.tip.mu_static)),
                    mu_kinetic=float(t.get("mu_k", c.tip.mu_kinetic)),
                    spin_friction=c.tip.spin_friction,
                    stability=c.tip.stability,
                    shape=c.tip.shape,
                    lad=c.tip.lad,
                )
        return c
    combo1 = match_and_apply(combo1)
    combo2 = match_and_apply(combo2)

    # Launch mapping: set bank/flower for matching names
    launch = data.get("launch", {})
    def apply_launch(c: Combo):
        key = None
        if c.name and c.name in launch:
            key = c.name
        else:
            for k in launch.keys():
                if c.name and k.lower() in c.name.lower():
                    key = k
                    break
        if key:
            l = launch[key]
            if "bank_deg" in l:
                c.bank_deg = float(l["bank_deg"])
            if "flower_arc_gain" in l:
                c.flower_arc_gain = float(l["flower_arc_gain"])
        return c
    combo1 = apply_launch(combo1)
    combo2 = apply_launch(combo2)

    return combo1, combo2, params, stadium
