from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

try:
    # Optional imports; we don’t hard-couple to these modules at import time
    from auth import current_user
except Exception:  # pragma: no cover
    def current_user() -> Optional[str]:  # fallback if auth is absent
        return None

try:
    from Genmap_modules.status_manager import get_session_status
except Exception:  # pragma: no cover
    def get_session_status(default: str = "Idle") -> str:
        return default


LOG_PATH = Path("logs/feedback.log")


def _append_log(entry: Dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Never break UI on logging issues
        pass


def submit_feedback(message: str, category: str = "general", extra: Optional[Dict[str, Any]] = None) -> bool:
    message = (message or "").strip()
    if not message:
        return False
    entry: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user": current_user() or "anonymous",
        "step": get_session_status(),
        "category": category,
        "message": message,
    }
    if extra:
        entry.update(extra)
    _append_log(entry)
    return True


def render_feedback_box(sidebar: bool = True) -> None:
    container = st.sidebar if sidebar else st
    with container:
        st.subheader("Feedback")
        txt = st.text_area("Décrivez votre problème ou suggestion", key="feedback_text", height=120)
        col1, col2 = st.columns([1, 1])
        with col1:
            category = st.selectbox("Catégorie", ["general", "bug", "idée"], key="feedback_category")
        with col2:
            include_context = st.checkbox("Inclure le contexte", value=True, key="feedback_ctx")
        if st.button("Envoyer le feedback", type="primary"):
            extra: Dict[str, Any] = {}
            if include_context:
                # capture a bit de contexte utile
                extra = {
                    "session_keys": list(st.session_state.keys()),
                }
            ok = submit_feedback(txt, category=category, extra=extra)
            if ok:
                st.success("Merci pour votre retour !")
                st.session_state["feedback_text"] = ""
            else:
                st.warning("Le message est vide — rien n’a été envoyé.")

