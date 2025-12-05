from __future__ import annotations

from typing import List

import streamlit as st

WARNING_KEY = "warning_messages"
STATUS_KEY = "session_status"


def reset_warnings() -> None:
    """Clear the warning list at the start of a new run."""
    st.session_state[WARNING_KEY] = []


def add_warning(message: str) -> None:
    """Store a warning message for display in the sidebar panel."""
    if not message:
        return
    warnings = st.session_state.setdefault(WARNING_KEY, [])
    if message not in warnings:
        warnings.append(message)


def get_warnings() -> List[str]:
    """Return the current warnings as a new list."""
    return list(st.session_state.get(WARNING_KEY, []))


def set_session_status(status: str) -> None:
    """Record the current step or status label."""
    st.session_state[STATUS_KEY] = status


def get_session_status(default: str = "Idle") -> str:
    """Fetch the most recently stored session status."""
    return st.session_state.get(STATUS_KEY, default)