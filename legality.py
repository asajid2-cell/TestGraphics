from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Set

from .parts import Combo


@dataclass
class LegalProfile:
    legal_metals: Set[str]
    legal_tracks: Set[str]
    legal_tips: Set[str]
    banned_metals: Set[str]
    banned_tracks: Set[str]
    banned_tips: Set[str]

    @classmethod
    def from_json(cls, path: str | Path) -> "LegalProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            legal_metals=set(data.get("legal_metals", [])),
            legal_tracks=set(data.get("legal_tracks", [])),
            legal_tips=set(data.get("legal_tips", [])),
            banned_metals=set(data.get("banned_metals", [])),
            banned_tracks=set(data.get("banned_tracks", [])),
            banned_tips=set(data.get("banned_tips", [])),
        )

    def validate_combo(self, c: Combo) -> list[str]:
        errs: list[str] = []
        if self.banned_metals and c.metal.name in self.banned_metals:
            errs.append(f"Banned metal wheel: {c.metal.name}")
        if self.legal_metals and c.metal.name not in self.legal_metals:
            errs.append(f"Illegal metal wheel: {c.metal.name}")
        if self.banned_tracks and c.track.name in self.banned_tracks:
            errs.append(f"Banned track: {c.track.name}")
        if self.legal_tracks and c.track.name not in self.legal_tracks:
            errs.append(f"Illegal track: {c.track.name}")
        if self.banned_tips and c.tip.name in self.banned_tips:
            errs.append(f"Banned tip: {c.tip.name}")
        if self.legal_tips and c.tip.name not in self.legal_tips:
            errs.append(f"Illegal tip: {c.tip.name}")
        return errs


@dataclass
class Catalog:
    metals_legal: Set[str]
    rings: Set[str]
    tracks_legal: Set[str]
    tips_legal: Set[str]
    metals_banned: Set[str]
    tracks_banned: Set[str]
    tips_banned: Set[str]
    aliases: Dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path) -> "Catalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            metals_legal=set(data.get("metals_legal", [])),
            rings=set(data.get("rings", [])),
            tracks_legal=set(data.get("tracks_legal", [])),
            tips_legal=set(data.get("tips_legal", [])),
            metals_banned=set(data.get("metals_banned", [])),
            tracks_banned=set(data.get("tracks_banned", [])),
            tips_banned=set(data.get("tips_banned", [])),
            aliases=dict(data.get("aliases", {})),
        )
