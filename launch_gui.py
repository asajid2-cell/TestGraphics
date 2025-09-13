from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Ensure repo root is on sys.path
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from beyblade_sim.gui import main as gui_main
    except Exception as e:
        print("Failed to import GUI:", e)
        print("Hint: run this from the repo root (folder containing 'beyblade_sim/').")
        return 2
    return gui_main([])


if __name__ == "__main__":
    raise SystemExit(main())

