from __future__ import annotations

try:
    from .gui import main as gui_main
except Exception:
    # Fallback if relative import fails when run weirdly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from beyblade_sim.gui import main as gui_main


if __name__ == "__main__":
    raise SystemExit(gui_main())

