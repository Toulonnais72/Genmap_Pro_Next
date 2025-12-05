import os
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


load_dotenv()  # loads .env at project root


# ---------------------------
# Config & user store loading
# ---------------------------
def _load_yaml_config() -> Dict[str, Any]:
    cfg_path = Path("auth_config.yaml")
    if cfg_path.exists() and yaml is not None:
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _build_user_store() -> Dict[str, Dict[str, Any]]:
    """Return a dict: username -> {hash: str, roles: [str]}.

    YAML-only: if no valid users are configured, authentication is locked.
    """
    cfg = _load_yaml_config()
    users_cfg = cfg.get("users") if isinstance(cfg, dict) else None
    store: Dict[str, Dict[str, Any]] = {}

    if isinstance(users_cfg, list):
        for u in users_cfg:
            if not isinstance(u, dict):
                continue
            username = str(u.get("username", "")).strip()
            pwd_hash = str(u.get("hash", "")).strip()
            roles = u.get("roles", [])
            if username and pwd_hash:
                store[username] = {"hash": pwd_hash, "roles": list(roles) if isinstance(roles, list) else []}

    return store


_USER_STORE = _build_user_store()


def _get_settings() -> Dict[str, Any]:
    cfg = _load_yaml_config()
    settings = {}
    if isinstance(cfg, dict):
        settings = cfg.get("settings", {}) or {}
    # Defaults
    return {
        "session_timeout_minutes": int(settings.get("session_timeout_minutes", 30)),
        "audit_log_path": str(settings.get("audit_log_path", "logs/auth_audit.log")),
        "ip_header_env": str(settings.get("ip_header_env", "X_FORWARDED_FOR")),
    }


# ---------------------------
# Helpers: IP + audit logging
# ---------------------------
def _get_client_ip() -> str:
    candidates = [
        os.getenv("X_FORWARDED_FOR"),
        os.getenv("HTTP_X_FORWARDED_FOR"),
        os.getenv("X_REAL_IP"),
        os.getenv("REMOTE_ADDR"),
    ]
    for c in candidates:
        if c:
            # X-Forwarded-For may be a list
            return c.split(",")[0].strip()
    # Streamlit typically runs behind localhost; IP often not exposed
    return "127.0.0.1"


def _log_event(event: str, username: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        settings = _get_settings()
        log_path = Path(settings["audit_log_path"]).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "user": username or st.session_state.get("username"),
            "ip": _get_client_ip(),
        }
        if extra:
            entry.update(extra)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Never block the app on logging errors
        pass


# ---------------------------
# Auth core
# ---------------------------
def _credentials_ok(username: str, password: str) -> bool:
    user = _USER_STORE.get(username)
    if not user:
        return False
    pwd_hash = user.get("hash", "").encode()
    if not pwd_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode(), pwd_hash)
    except Exception:
        return False


def _clear_auth_state():
    for k in ("auth_ok", "username", "roles", "last_activity"):
        if k in st.session_state:
            del st.session_state[k]


def _touch_activity():
    st.session_state["last_activity"] = datetime.now(timezone.utc).isoformat()


def _is_session_expired() -> bool:
    settings = _get_settings()
    timeout_min = int(settings.get("session_timeout_minutes", 30))
    last = st.session_state.get("last_activity")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(str(last))
    except Exception:
        return False
    return datetime.now(timezone.utc) - last_dt > timedelta(minutes=timeout_min)


def has_role(role: str) -> bool:
    roles = st.session_state.get("roles") or []
    return role in roles


def current_user() -> Optional[str]:
    return st.session_state.get("username")


def _render_logged_in_controls() -> None:
    with st.sidebar:
        user = current_user() or "?"
        roles = ", ".join(st.session_state.get("roles", [])) or "no-role"
        st.caption(f"Signed in as {user} ({roles})")
        if st.button("Logout"):
            _log_event("logout", username=user)
            _clear_auth_state()
            st.rerun()


def login() -> bool:
    """Render a login form in the sidebar and return True if authenticated."""
    # Expire session if needed
    if st.session_state.get("auth_ok") is True and _is_session_expired():
        _log_event("session_timeout", username=st.session_state.get("username"))
        _clear_auth_state()

    if st.session_state.get("auth_ok") is True:
        _touch_activity()
        _render_logged_in_controls()
        return True

    with st.sidebar:
        if not _USER_STORE:
            st.error("Authentication not configured. Please create auth_config.yaml with users.")
            return False
        st.subheader("Sign in")
        u = st.text_input("User", value="", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign in", type="primary"):
            if _credentials_ok(u, p):
                st.session_state["auth_ok"] = True
                st.session_state["username"] = u
                st.session_state["roles"] = _USER_STORE.get(u, {}).get("roles", [])
                _touch_activity()
                _log_event("login_success", username=u)
                st.rerun()
            else:
                _log_event("login_failure", username=u)
                st.error("Invalid credentials")
    return False


def require_auth():
    """Stop the app until the user is authenticated. Also renders logout."""
    if not login():
        st.stop()

