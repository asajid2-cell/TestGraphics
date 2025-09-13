from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

try:
    from .parts import Tip, Track, MetalWheel, Combo
except Exception:
    # Single-folder fallback
    import os, sys as _sys
    _base = os.path.dirname(__file__)
    if _base not in _sys.path:
        _sys.path.insert(0, _base)
    from parts import Tip, Track, MetalWheel, Combo


class PartRegistry:
    def __init__(self, tips: Dict[str, Tip], tracks: Dict[str, Track], metals: Dict[str, MetalWheel]):
        self.tips = tips
        self.tracks = tracks
        self.metals = metals

    @classmethod
    def from_json(cls, path: str | Path) -> "PartRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        def to_tip(t: dict) -> Tip:
            return Tip(
                name=t["name"],
                mu_static=t["mu_static"],
                mu_kinetic=t["mu_kinetic"],
                spin_friction=t["spin_friction"],
                stability=t["stability"],
                shape=t.get("shape", ""),
                lad=float(t.get("lad", 0.0)),
            )
        def to_track(tr: dict) -> Track:
            return Track(
                name=tr["name"],
                height_mm=tr["height_mm"],
                scrape_risk=tr.get("scrape_risk", 0.0),
            )
        def to_metal(m: dict) -> MetalWheel:
            return MetalWheel(
                name=m["name"],
                mass_g=m["mass_g"],
                radius_mm=m["radius_mm"],
                attack=m["attack"],
                defense=m["defense"],
                stamina=m["stamina"],
                recoil=m["recoil"],
                left_spin=bool(m.get("left_spin", False)),
                spin_eq=float(m.get("spin_eq", 0.0)),
            )

        tips = {t["name"]: to_tip(t) for t in data.get("tips", [])}
        tracks = {tr["name"]: to_track(tr) for tr in data.get("tracks", [])}
        metals = {m["name"]: to_metal(m) for m in data.get("metal_wheels", [])}
        return cls(tips=tips, tracks=tracks, metals=metals)


def load_combos(path: str | Path, reg: PartRegistry) -> Dict[str, Combo]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result: Dict[str, Combo] = {}
    for c in data.get("combos", []):
        metal = reg.metals[c["metal"]]
        track = reg.tracks[c["track"]]
        tip = reg.tips[c["tip"]]
        combo = Combo(
            name=c.get("name"),
            metal=metal,
            track=track,
            tip=tip,
            launch_power=float(c.get("launch_power", 0.8)),
            mass_g=c.get("mass_g"),
            radius_mm=c.get("radius_mm"),
        )
        result[combo.name or f"combo_{len(result)+1}"] = combo
    return result


# --- Convenience parsing for shorthand combo names, e.g., "Flame230CS" or "Meteo145WD" ---

def resolve_combo_name(name: str, reg: PartRegistry, launch_power: float = 0.8, catalog: "Catalog" | None = None) -> Optional[Combo]:
    """Parse a shorthand combo like Metal+Track+Tip (no separators) and build a Combo.
    Returns None if parsing fails.
    """
    s = name.strip()
    # Find tip by longest suffix match
    tip_match = None
    for tip_name in sorted(reg.tips.keys(), key=len, reverse=True):
        if s.endswith(tip_name):
            tip_match = tip_name
            break
    if not tip_match:
        return None
    s2 = s[: -len(tip_match)]

    # Find track number in the remaining string (assume exact track names used like 90/100/105/145/230)
    track_match = None
    for tr_name in sorted(reg.tracks.keys(), key=len, reverse=True):
        if s2.endswith(tr_name):
            track_match = tr_name
            break
    if not track_match:
        return None
    metal_key = s2[: -len(track_match)]

    # Some names may include a Clear Wheel/Energy Ring in the middle (e.g., EarthBull145WD).
    # If a catalog is provided, try to split the middle token into (metal, ring) and ignore ring for physics.
    if catalog is not None and metal_key:
        # Try split by longest ring suffix
        ring_match = None
        for ring in sorted(catalog.rings, key=len, reverse=True):
            if metal_key.endswith(ring):
                ring_match = ring
                break
        if ring_match:
            metal_candidate = metal_key[: -len(ring_match)]
            if metal_candidate in reg.metals:
                metal_key = metal_candidate
    if not metal_key:
        return None

    # Metal aliasing for common shorthand
    aliases = {
        "Meteo": "MeteoLDrago",
        # Add more shorthand -> registry name mappings here if needed
    }
    metal_name = metal_key
    if metal_name not in reg.metals and metal_name in aliases:
        metal_name = aliases[metal_name]

    if metal_name not in reg.metals or track_match not in reg.tracks or tip_match not in reg.tips:
        return None

    return Combo(
        name=name,
        metal=reg.metals[metal_name],
        track=reg.tracks[track_match],
        tip=reg.tips[tip_match],
        launch_power=launch_power,
    )
