from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stadium:
    name: str
    radius_mm: float  # inner playable bowl radius (approx)
    wall_radius_mm: float  # rim/KO radius
    slope: float  # 0..1 slope factor of bowl (steepness)
    wall_restitution: float  # 0..1 bounce factor on wall hit
    center_pit_radius_mm: float = 0.0


def bb10_default() -> Stadium:
    # Approximate BB-10 Attack-type stadium dimensions and properties
    # Values are approximate and tuned for gameplay feel.
    return Stadium(
        name="BB-10 Attack",
        radius_mm=220.0,
        wall_radius_mm=235.0,
        slope=0.35,
        wall_restitution=0.3,
        center_pit_radius_mm=0.0,
    )

