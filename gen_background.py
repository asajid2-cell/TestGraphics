from __future__ import annotations

import math
from pathlib import Path


def clamp(x: float, lo: int = 0, hi: int = 255) -> int:
    return int(max(lo, min(hi, round(x))))


def generate_bb10_ppm(path: str | Path, size: int = 1024) -> None:
    N = size
    cx = cy = N / 2.0
    R = N * 0.46  # leave a bit of padding

    # Pockets angles in radians (approx 0, 120, 240 degrees)
    pockets = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
    # pocket line thickness in pixels
    pocket_thick = max(3, int(N * 0.007))
    pocket_inner = R - N * 0.006
    pocket_outer = R + N * 0.012

    # Precompute pocket segments (x0,y0)-(x1,y1)
    segs = []
    for ang in pockets:
        x0 = cx + pocket_inner * math.cos(ang)
        y0 = cy - pocket_inner * math.sin(ang)
        x1 = cx + pocket_outer * math.cos(ang)
        y1 = cy - pocket_outer * math.sin(ang)
        segs.append((x0, y0, x1, y1))

    def dist_to_segment(px, py, x0, y0, x1, y1):
        # distance from point to segment
        vx = x1 - x0
        vy = y1 - y0
        wx = px - x0
        wy = py - y0
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - x0, py - y0)
        c2 = vx * vx + vy * vy
        if c2 <= 0:
            return math.hypot(px - x0, py - y0)
        t = c1 / c2
        if t >= 1:
            return math.hypot(px - x1, py - y1)
        projx = x0 + t * vx
        projy = y0 + t * vy
        return math.hypot(px - projx, py - projy)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="ascii") as f:
        f.write(f"P3\n{N} {N}\n255\n")
        for j in range(N):
            row = []
            for i in range(N):
                x = i + 0.5
                y = j + 0.5
                dx = x - cx
                dy = y - cy
                r = math.hypot(dx, dy)

                # Base background
                if r > R + 2:
                    row.append("11 11 11")
                    continue

                # Radial shading for bowl
                t = max(0.0, min(1.0, r / R))
                # invert so center is lighter
                shade = (1.0 - t)
                base = 38 + 40 * shade  # 38..78

                # Concentric guide rings
                ring = 0.0
                for frac, amp in [(0.85, 22), (0.65, 16), (0.45, 12)]:
                    rr = R * frac
                    ring += amp * math.exp(-((r - rr) ** 2) / (2 * (N * 0.004) ** 2))

                # Pocket highlight
                pocket_boost = 0.0
                for (x0, y0, x1, y1) in segs:
                    d = dist_to_segment(x, y, x0, y0, x1, y1)
                    if d <= pocket_thick:
                        pocket_boost = 140 * max(0.0, 1.0 - d / pocket_thick)
                        break

                rch = clamp(base + ring + pocket_boost * 0.8)
                gch = clamp(base + ring + pocket_boost * 0.65)
                bch = clamp(base + ring * 0.8 + pocket_boost * 0.2)

                row.append(f"{rch} {gch} {bch}")
            f.write(" ".join(row) + "\n")


def main():
    out = Path("data/backgrounds/bb10.ppm")
    generate_bb10_ppm(out, size=1024)
    print(f"Generated {out}")


if __name__ == "__main__":
    main()

