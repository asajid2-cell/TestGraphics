from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .parts import Combo, MetalWheel, Track, Tip
from .stadium import Stadium, bb10_default
from .physics import SimParams, simulate_duel
from .parts import Combo as _Combo


@dataclass
class SimSummary:
    runs: int
    a_wins: int
    b_wins: int
    draws: int
    ko_wins_a: int
    ko_wins_b: int
    avg_time: float


def run_series(a: Combo, b: Combo, runs: int = 200, stadium: Optional[Stadium] = None, params: Optional[SimParams] = None, seed: Optional[int] = None, randomize_launch: bool = False) -> SimSummary:
    stadium = stadium or bb10_default()
    params = params or SimParams()
    rng_seed = seed
    a_wins = b_wins = draws = ko_a = ko_b = 0
    total_time = 0.0
    for i in range(runs):
        if randomize_launch:
            import random as _r
            rr = _r.Random((rng_seed + i) if rng_seed is not None else None)
            def _with_lp(c: _Combo, lp: float) -> _Combo:
                return _Combo(name=c.name, metal=c.metal, track=c.track, tip=c.tip, launch_power=lp, mass_g=c.mass_g, radius_mm=c.radius_mm)
            a_lp = 0.5 + 0.5 * rr.random()
            b_lp = 0.5 + 0.5 * rr.random()
            a_use = _with_lp(a, a_lp)
            b_use = _with_lp(b, b_lp)
        else:
            a_use = a
            b_use = b
        result = simulate_duel(a_use, b_use, stadium, params, seed=(rng_seed + i) if rng_seed is not None else None)
        total_time += result["time"]
        if result["winner"] == "A":
            a_wins += 1
            if result["method"] == "KO":
                ko_a += 1
        elif result["winner"] == "B":
            b_wins += 1
            if result["method"] == "KO":
                ko_b += 1
        else:
            draws += 1
    avg_time = total_time / max(1, runs)
    return SimSummary(
        runs=runs,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        ko_wins_a=ko_a,
        ko_wins_b=ko_b,
        avg_time=avg_time,
    )
