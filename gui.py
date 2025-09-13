from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json

try:
    # Normal package-relative import
    from .io import PartRegistry, load_combos
except Exception:
    # Fallback if user ran this file directly from inside the package folder
    import sys
    from pathlib import Path as _P
    pkg_root = str(_P(__file__).resolve().parents[1])
    if pkg_root not in sys.path:
        sys.path.append(pkg_root)
    from beyblade_sim.io import PartRegistry, load_combos


class ControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MFL Simulator - Control Panel")
        self.geometry("760x520")

        self.parts_path = tk.StringVar(value=str(Path("data/parts.json")))
        self.combos_path = tk.StringVar(value=str(Path("data/combos.json")))
        self.preset = tk.StringVar(value="default")
        self.preset_json = tk.StringVar(value="")
        self.combo1 = tk.StringVar()
        self.combo2 = tk.StringVar()
        self.no_ko = tk.BooleanVar(value=False)
        self.charge_launch = tk.BooleanVar(value=False)
        self.fps = tk.StringVar(value="60")
        self.size = tk.StringVar(value="900x700")
        self.seed = tk.StringVar(value="")
        self.speed_boost = tk.StringVar(value="")
        self.trail_length = tk.StringVar(value="150")
        self.bey_size_scale = tk.StringVar(value="0.8")
        self.mode = tk.StringVar(value="2D")  # 2D or 3D

        self._build_ui()
        self._ensure_data()
        self._load_combos()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Paths
        row = 0
        ttk.Label(frm, text="Parts JSON:").grid(row=row, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.parts_path, width=60).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="...", command=self._pick_parts).grid(row=row, column=2)
        row += 1
        ttk.Label(frm, text="Combos JSON:").grid(row=row, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.combos_path, width=60).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="...", command=self._pick_combos).grid(row=row, column=2)
        row += 1
        ttk.Label(frm, text="Preset JSON (advanced):").grid(row=row, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.preset_json, width=60).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="...", command=self._pick_preset).grid(row=row, column=2)
        row += 1

        # Combos
        ttk.Label(frm, text="Combo A:").grid(row=row, column=0, sticky="e")
        self.combo1_box = ttk.Combobox(frm, textvariable=self.combo1, width=40)
        self.combo1_box.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(frm, text="Combo B:").grid(row=row, column=0, sticky="e")
        self.combo2_box = ttk.Combobox(frm, textvariable=self.combo2, width=40)
        self.combo2_box.grid(row=row, column=1, sticky="w")
        row += 1

        # Toggles
        ttk.Checkbutton(frm, text="No KO (stamina only)", variable=self.no_ko).grid(row=row, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Charge launch (hold 'L')", variable=self.charge_launch).grid(row=row, column=1, sticky="e")
        row += 1

        # Preset & mode
        ttk.Label(frm, text="Preset:").grid(row=row, column=0, sticky="e")
        ttk.Combobox(frm, textvariable=self.preset, values=["default", "stamina-2min", "stamina-long", "stamina-extreme"], width=20).grid(row=row, column=1, sticky="w")
        ttk.Label(frm, text="Mode:").grid(row=row, column=2, sticky="e")
        ttk.Combobox(frm, textvariable=self.mode, values=["2D", "3D"], width=8).grid(row=row, column=3, sticky="w")
        row += 1

        # Numeric
        grid2 = ttk.Frame(frm)
        grid2.grid(row=row, column=0, columnspan=4, sticky="we", pady=(8, 0))
        ttk.Label(grid2, text="FPS:").grid(row=0, column=0, sticky="e"); ttk.Entry(grid2, textvariable=self.fps, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(grid2, text="Size (WxH):").grid(row=0, column=2, sticky="e"); ttk.Entry(grid2, textvariable=self.size, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(grid2, text="Seed:").grid(row=0, column=4, sticky="e"); ttk.Entry(grid2, textvariable=self.seed, width=10).grid(row=0, column=5, sticky="w")
        ttk.Label(grid2, text="Speed boost:").grid(row=1, column=0, sticky="e"); ttk.Entry(grid2, textvariable=self.speed_boost, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(grid2, text="Trail length:").grid(row=1, column=2, sticky="e"); ttk.Entry(grid2, textvariable=self.trail_length, width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(grid2, text="Bey size scale:").grid(row=1, column=4, sticky="e"); ttk.Entry(grid2, textvariable=self.bey_size_scale, width=6).grid(row=1, column=5, sticky="w")
        row += 1

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=4, pady=10)
        ttk.Button(btns, text="Launch", command=self._launch).grid(row=0, column=0, padx=6)
        ttk.Button(btns, text="Reload combos", command=self._load_combos).grid(row=0, column=1, padx=6)
        ttk.Button(btns, text="Generate (all)", command=self._generate_all_combos).grid(row=0, column=2, padx=6)
        ttk.Button(btns, text="Generate (rules JSON)", command=self._generate_from_rules).grid(row=0, column=3, padx=6)
        ttk.Button(btns, text="Open Builder", command=self._open_builder).grid(row=0, column=4, padx=6)
        ttk.Button(btns, text="Generate combos", command=self._generate_all_combos).grid(row=0, column=2, padx=6)

        for i in range(4):
            frm.columnconfigure(i, weight=1)

    def _pick_parts(self):
        p = filedialog.askopenfilename(title="Select parts.json", filetypes=[("JSON","*.json")])
        if p:
            self.parts_path.set(p)

    def _pick_combos(self):
        p = filedialog.askopenfilename(title="Select combos.json", filetypes=[("JSON","*.json")])
        if p:
            self.combos_path.set(p)
            self._load_combos()

    def _pick_preset(self):
        p = filedialog.askopenfilename(title="Select preset JSON", filetypes=[("JSON","*.json")])
        if p:
            self.preset_json.set(p)

    def _load_combos(self):
        try:
            reg = PartRegistry.from_json(self.parts_path.get())
            if Path(self.combos_path.get()).exists():
                combos = load_combos(self.combos_path.get(), reg)
                names = sorted(combos.keys())
                self.combo1_box["values"] = names
                self.combo2_box["values"] = names
                if names:
                    self.combo1.set(names[0])
                    self.combo2.set(names[1 if len(names) > 1 else 0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load combos: {e}")

    def _launch(self):
        # Build command (single-folder friendly: run script path directly)
        mode = self.mode.get()
        script = (Path(__file__).resolve().parent / ("visual_rt.py" if mode == "2D" else "visual_3d.py"))
        cmd = [sys.executable, str(script),
               "--parts", self.parts_path.get(),
               "--combos", self.combos_path.get(),
               "--combo1", self.combo1.get(),
               "--combo2", self.combo2.get(),
               "--fps", self.fps.get(), "--size", self.size.get()]
        if self.seed.get():
            cmd += ["--seed", self.seed.get()]
        if self.no_ko.get():
            cmd += ["--no-ko"]
        if self.charge_launch.get() and mode == "2D":
            cmd += ["--charge-launch"]
        if self.preset.get():
            cmd += ["--preset", self.preset.get()]
        if self.preset_json.get():
            cmd += ["--preset-json", self.preset_json.get()]
        if self.speed_boost.get():
            cmd += ["--speed-boost", self.speed_boost.get()]
        if self.trail_length.get():
            cmd += ["--trail-length", self.trail_length.get()]
        if self.bey_size_scale.get():
            cmd += ["--bey-size-scale", self.bey_size_scale.get()]

        try:
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch viewer:\n{e}")

    def _ensure_data(self):
        parts_sel = Path(self.parts_path.get())
        combos_sel = Path(self.combos_path.get())
        if parts_sel.exists() and combos_sel.exists():
            return
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        parts_path = data_dir / "parts.json"
        combos_path = data_dir / "combos.json"
        if not parts_path.exists():
            parts_path.write_text('{"tips":[{"name":"RF","mu_static":0.9,"mu_kinetic":0.75,"spin_friction":0.9,"stability":0.55,"shape":"RF"},{"name":"CS","mu_static":0.5,"mu_kinetic":0.35,"spin_friction":0.55,"stability":0.75,"shape":"CS"},{"name":"WD","mu_static":0.2,"mu_kinetic":0.16,"spin_friction":0.38,"stability":0.9,"shape":"WD","lad":0.5},{"name":"MF","mu_static":0.32,"mu_kinetic":0.22,"spin_friction":0.5,"stability":0.6,"shape":"MF","lad":0.1}],"tracks":[{"name":"100","height_mm":10.0,"scrape_risk":0.25},{"name":"145","height_mm":14.5,"scrape_risk":0.1},{"name":"230","height_mm":23.0,"scrape_risk":0.05}],"metal_wheels":[{"name":"Flame","mass_g":29.6,"radius_mm":21.2,"attack":0.35,"defense":0.35,"stamina":0.7,"recoil":0.3},{"name":"MeteoLDrago","mass_g":28.3,"radius_mm":21.0,"attack":0.7,"defense":0.3,"stamina":0.5,"recoil":0.8,"left_spin":true,"spin_eq":0.7},{"name":"Bakushin","mass_g":29.0,"radius_mm":21.0,"attack":0.62,"defense":0.38,"stamina":0.55,"recoil":0.5}]}'
            ,encoding='utf-8')
        if not combos_path.exists():
            combos_path.write_text('{"combos":[{"name":"Flame230CS","metal":"Flame","track":"230","tip":"CS","launch_power":0.85},{"name":"Meteo145WD","metal":"MeteoLDrago","track":"145","tip":"WD","launch_power":0.8},{"name":"Bakushin145MF","metal":"Bakushin","track":"145","tip":"MF","launch_power":0.9}]}'
            ,encoding='utf-8')
        self.parts_path.set(str(parts_path))
        self.combos_path.set(str(combos_path))

    def _generate_all_combos(self):
        """Generate the Cartesian product of current parts into combos.json (no legality filtering)."""
        try:
            parts_path = Path(self.parts_path.get())
            combos_path = Path(self.combos_path.get())
            data = json.loads(parts_path.read_text(encoding='utf-8'))
            metals = [m['name'] for m in data.get('metal_wheels', [])]
            tracks = [t['name'] for t in data.get('tracks', [])]
            tips = [t['name'] for t in data.get('tips', [])]
            combos = []
            if combos_path.exists():
                cur = json.loads(combos_path.read_text(encoding='utf-8'))
                combos = cur.get('combos', [])
            existing = {(c['metal'], c['track'], c['tip']) for c in combos}
            for m in metals:
                for tr in tracks:
                    for tip in tips:
                        key = (m, tr, tip)
                        if key in existing:
                            continue
                        combos.append({
                            'name': f'{m}{tr}{tip}',
                            'metal': m,
                            'track': tr,
                            'tip': tip,
                            'launch_power': 0.85
                        })
            combos_path.write_text(json.dumps({'combos': combos}, indent=2), encoding='utf-8')
            messagebox.showinfo('Combos', f'Generated {len(combos)} combos into\n{combos_path}')
            self._load_combos()
        except Exception as e:
            messagebox.showerror('Error', f'Failed to generate combos:\n{e}')

    def _generate_from_rules(self):
        """Generate combos based on an MFB parts JSON with rules, filtering illegal combos."""
        try:
            # Ask for a rules JSON (or use current parts if it contains rules)
            p = filedialog.askopenfilename(title='Select MFB parts+rules JSON', filetypes=[('JSON','*.json')])
            if not p:
                return
            j = json.loads(Path(p).read_text(encoding='utf-8'))
            # Get parts from this JSON if present, else fall back to current parts
            parts = j if ('metal_wheels' in j and 'tracks' in j and 'tips' in j) else json.loads(Path(self.parts_path.get()).read_text(encoding='utf-8'))
            metals = [m['name'] for m in parts.get('metal_wheels', [])]
            tracks = [t['name'] for t in parts.get('tracks', [])]
            tips = [t['name'] for t in parts.get('tips', [])]

            rules = j.get('rules', j.get('mfl_rules', {}))
            banned_m = set(rules.get('banned_metal_wheels', []))
            banned_tr = set(rules.get('banned_tracks', []))
            banned_tp = set(rules.get('banned_tips', []))
            illegal = rules.get('illegal_pairs', {})
            ill_m_tip = set(tuple(x) for x in illegal.get('metal_wheel_tip', []))
            ill_m_tr = set(tuple(x) for x in illegal.get('metal_wheel_track', []))
            ill_tr_tip = set(tuple(x) for x in illegal.get('track_tip', []))
            # Optional whitelist
            legal_m = set(rules.get('legal_metal_wheels', metals))
            legal_tr = set(rules.get('legal_tracks', tracks))
            legal_tp = set(rules.get('legal_tips', tips))

            def is_legal(m, tr, tip):
                if m in banned_m or tr in banned_tr or tip in banned_tp:
                    return False
                if m not in legal_m or tr not in legal_tr or tip not in legal_tp:
                    return False
                if (m, tip) in ill_m_tip:
                    return False
                if (m, tr) in ill_m_tr:
                    return False
                if (tr, tip) in ill_tr_tip:
                    return False
                return True

            combos = []
            for m in metals:
                for tr in tracks:
                    for tip in tips:
                        if not is_legal(m, tr, tip):
                            continue
                        combos.append({
                            'name': f'{m}{tr}{tip}',
                            'metal': m,
                            'track': tr,
                            'tip': tip,
                            'launch_power': 0.85
                        })
            # Limit if requested in rules
            mx = rules.get('max_combos')
            if isinstance(mx, int) and mx > 0:
                combos = combos[:mx]
            combos_path = Path(self.combos_path.get())
            combos_path.write_text(json.dumps({'combos': combos}, indent=2), encoding='utf-8')
            messagebox.showinfo('Combos', f'Generated {len(combos)} legal combos into\n{combos_path}')
            self._load_combos()
        except Exception as e:
            messagebox.showerror('Error', f'Failed to generate combos from rules:\n{e}')

    def _open_builder(self):
        # Minimal builder window with part selectors and Add button
        try:
            parts = json.loads(Path(self.parts_path.get()).read_text(encoding='utf-8'))
        except Exception as e:
            messagebox.showerror('Error', f'Failed to read parts JSON:\n{e}')
            return

        def get_list(keys):
            for k in keys:
                if k in parts and isinstance(parts[k], list):
                    return [x['name'] for x in parts[k] if 'name' in x]
            return []

        faces = get_list(['faces', 'face_bolts', 'bolts'])
        rings = get_list(['energy_rings', 'clear_wheels', 'rings'])
        metals = get_list(['metal_wheels', 'fusion_wheels']) or get_list(['warrior_wheels','chrome_wheels'])
        tracks = get_list(['tracks', 'spin_tracks'])
        tips = get_list(['tips', 'performance_tips'])

        win = tk.Toplevel(self)
        win.title('Beybuilder')
        win.geometry('640x480')
        frm = ttk.Frame(win, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(frm, text='Name:').grid(row=row, column=0, sticky='e'); name = tk.StringVar(value='CustomBey'); ttk.Entry(frm, textvariable=name, width=30).grid(row=row, column=1, sticky='w'); row+=1
        if faces:
            ttk.Label(frm, text='Face/Bolt:').grid(row=row, column=0, sticky='e'); face = tk.StringVar(); ttk.Combobox(frm, textvariable=face, values=faces, width=30).grid(row=row, column=1, sticky='w'); row+=1
        else:
            face = tk.StringVar(value='')
        if rings:
            ttk.Label(frm, text='Ring/Clear Wheel:').grid(row=row, column=0, sticky='e'); ring = tk.StringVar(); ttk.Combobox(frm, textvariable=ring, values=rings, width=30).grid(row=row, column=1, sticky='w'); row+=1
        else:
            ring = tk.StringVar(value='')
        ttk.Label(frm, text='Metal/Fusion Wheel:').grid(row=row, column=0, sticky='e'); metal = tk.StringVar(); ttk.Combobox(frm, textvariable=metal, values=metals, width=30).grid(row=row, column=1, sticky='w'); row+=1
        ttk.Label(frm, text='Track:').grid(row=row, column=0, sticky='e'); track = tk.StringVar(); ttk.Combobox(frm, textvariable=track, values=tracks, width=30).grid(row=row, column=1, sticky='w'); row+=1
        ttk.Label(frm, text='Tip:').grid(row=row, column=0, sticky='e'); tip = tk.StringVar(); ttk.Combobox(frm, textvariable=tip, values=tips, width=30).grid(row=row, column=1, sticky='w'); row+=1
        ttk.Label(frm, text='Launch power (0..1):').grid(row=row, column=0, sticky='e'); lp = tk.StringVar(value='0.85'); ttk.Entry(frm, textvariable=lp, width=10).grid(row=row, column=1, sticky='w'); row+=1

        # Diagram placeholder
        canvas = tk.Canvas(frm, width=240, height=160, bg='#111111')
        canvas.grid(row=row, column=0, columnspan=2, pady=10)
        canvas.create_oval(20,20,220,140, outline='#555555')
        canvas.create_text(120,80, text='BB-10', fill='#888888')
        row+=1

        rules_path = tk.StringVar(value='')
        ttk.Label(frm, text='Rules JSON (optional):').grid(row=row, column=0, sticky='e'); ttk.Entry(frm, textvariable=rules_path, width=30).grid(row=row, column=1, sticky='w'); ttk.Button(frm, text='...', command=lambda: rules_path.set(filedialog.askopenfilename(title='Select rules JSON', filetypes=[('JSON','*.json')]) or rules_path.get())).grid(row=row, column=2); row+=1

        def add_combo():
            try:
                c_name = name.get().strip() or 'CustomBey'
                c_metal = metal.get().strip()
                c_track = track.get().strip()
                c_tip = tip.get().strip()
                if not (c_metal and c_track and c_tip):
                    messagebox.showerror('Error', 'Select metal, track, and tip')
                    return
                # legality filter if rules provided
                def legal_ok(m,tr,tp):
                    rp = rules_path.get().strip()
                    if not rp:
                        return True
                    try:
                        rj = json.loads(Path(rp).read_text(encoding='utf-8'))
                    except Exception:
                        return True
                    rules = rj.get('rules', rj.get('mfl_rules', {}))
                    banned_m = set(rules.get('banned_metal_wheels', []))
                    banned_tr = set(rules.get('banned_tracks', []))
                    banned_tp = set(rules.get('banned_tips', []))
                    illegal = rules.get('illegal_pairs', {})
                    ill_m_tip = set(tuple(x) for x in illegal.get('metal_wheel_tip', []))
                    ill_m_tr = set(tuple(x) for x in illegal.get('metal_wheel_track', []))
                    ill_tr_tip = set(tuple(x) for x in illegal.get('track_tip', []))
                    legal_m = set(rules.get('legal_metal_wheels', [m]))
                    legal_tr = set(rules.get('legal_tracks', [tr]))
                    legal_tp = set(rules.get('legal_tips', [tp]))
                    if m in banned_m or tr in banned_tr or tp in banned_tp:
                        return False
                    if (m,tr) in ill_m_tr or (m,tp) in ill_m_tip or (tr,tp) in ill_tr_tip:
                        return False
                    if (m not in legal_m) or (tr not in legal_tr) or (tp not in legal_tp):
                        return False
                    return True

                if not legal_ok(c_metal, c_track, c_tip):
                    messagebox.showerror('Illegal', 'This combination is not legal per the rules JSON')
                    return
                combos_path = Path(self.combos_path.get())
                combos = []
                if combos_path.exists():
                    combos = json.loads(combos_path.read_text(encoding='utf-8')).get('combos', [])
                combos.append({
                    'name': c_name,
                    'metal': c_metal,
                    'track': c_track,
                    'tip': c_tip,
                    'launch_power': float(lp.get() or 0.85)
                })
                combos_path.write_text(json.dumps({'combos': combos}, indent=2), encoding='utf-8')
                messagebox.showinfo('Added', f'Added {c_name} to\n{combos_path}')
                self._load_combos()
            except Exception as e:
                messagebox.showerror('Error', f'Failed to add combo:\n{e}')

        ttk.Button(frm, text='Add Combo', command=add_combo).grid(row=row, column=0, columnspan=2, pady=8)

    def _generate_all_combos(self):
        """Generate all combinations of the sample metals/tracks/tips into combos.json."""
        try:
            parts_path = Path(self.parts_path.get())
            combos_path = Path(self.combos_path.get())
            data = json.loads(parts_path.read_text(encoding='utf-8'))
            metals = [m['name'] for m in data.get('metal_wheels', [])]
            tracks = [t['name'] for t in data.get('tracks', [])]
            tips = [t['name'] for t in data.get('tips', [])]
            combos = []
            if combos_path.exists():
                cur = json.loads(combos_path.read_text(encoding='utf-8'))
                combos = cur.get('combos', [])
            existing = { (c['metal'], c['track'], c['tip']) for c in combos }
            for m in metals:
                for tr in tracks:
                    for tip in tips:
                        key = (m, tr, tip)
                        if key in existing:
                            continue
                        name = f"{m}{tr}{tip}"
                        combos.append({
                            'name': name,
                            'metal': m,
                            'track': tr,
                            'tip': tip,
                            'launch_power': 0.85
                        })
            combos_path.write_text(json.dumps({'combos': combos}, indent=2), encoding='utf-8')
            messagebox.showinfo('Combos', f'Generated {len(combos)} combos into\n{combos_path}')
            self._load_combos()
        except Exception as e:
            messagebox.showerror('Error', f'Failed to generate combos:\n{e}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MFL Simulator GUI")
    _ = parser.parse_args(argv)
    app = ControlPanel()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
