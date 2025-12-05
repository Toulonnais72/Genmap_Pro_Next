from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

try:
    from auth import has_role
except Exception:  # pragma: no cover
    def has_role(role: str) -> bool:
        return False


LOG_PATH = Path("logs/feedback.log")


def _read_last_lines(n: int = 200) -> List[Dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    lines: List[Dict[str, Any]] = []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lines.append(json.loads(raw))
                except Exception:
                    continue
    except Exception:
        return []
    return lines[-n:]


def render_admin_panel() -> None:
    if not has_role("admin"):
        return
    with st.expander("Feedback admin", expanded=False):
        st.caption("Derniers retours utilisateurs (max 200)")
        records = _read_last_lines(200)
        if not records:
            st.info("Aucun feedback enregistré pour le moment.")
            return
        # order by ts if present
        try:
            records = sorted(records, key=lambda r: r.get("ts", ""))
        except Exception:
            pass
        # Display a compact table
        cols = ["ts", "user", "step", "category", "message"]
        data = [{c: rec.get(c, "") for c in cols} for rec in records]
        st.dataframe(data, use_container_width=True)
