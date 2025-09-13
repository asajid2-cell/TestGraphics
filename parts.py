from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class Tip:
    name: str
    mu_static: float  # static friction coefficient
    mu_kinetic: float  # kinetic friction coefficient
    spin_friction: float  # spin-down coefficient (arbitrary units)
    stability: float  # resistance to precession/topple (0..1)
    shape: str  # e.g., "RF", "F", "S", "WB"
    lad: float = 0.0  # Life After Death potential (0..1)


@dataclass(frozen=True)
class Track:
    name: str
    height_mm: float
    scrape_risk: float  # 0..1; higher = more likely to scrape


@dataclass(frozen=True)
class MetalWheel:
    name: str
    mass_g: float
    radius_mm: float
    attack: float  # 0..1
    defense: float  # 0..1
    stamina: float  # 0..1
    recoil: float  # 0..1
    left_spin: bool = False  # True for L-Drago family, etc.
    spin_eq: float = 0.0  # inherent spin equalization potential (0..1)


@dataclass
class Combo:
    name: Optional[str]
    metal: MetalWheel
    track: Track
    tip: Tip
    launch_power: float = 0.8  # 0..1
    # Derived/overrides
    mass_g: Optional[float] = None
    radius_mm: Optional[float] = None
    moi_override: Optional[float] = None  # kg*m^2 override
    bank_deg: Optional[float] = None      # launch bank angle (deg)
    flower_arc_gain: Optional[float] = None  # curvature gain for attacker

    def total_mass_g(self) -> float:
        # Approximate added mass for track+tip
        base = self.metal.mass_g
        # Use rough constants to account for non-metal parts; this is simplified
        return self.mass_g if self.mass_g else base + 9.0

    def effective_radius_mm(self) -> float:
        return self.radius_mm if self.radius_mm else self.metal.radius_mm

    def stats(self) -> Dict[str, float]:
        # Aggregate derived stats with simple weighting; can be tuned
        return {
            "attack": self.metal.attack,
            "defense": self.metal.defense,
            "stamina": min(1.0, self.metal.stamina * (0.85 + 0.15 * self.tip.stability)),
            "recoil": self.metal.recoil,
            "spin_eq": min(1.0, 0.5 * self.metal.spin_eq + 0.5 * (self.tip.mu_static)),
        }
