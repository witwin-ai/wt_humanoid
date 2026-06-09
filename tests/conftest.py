"""Test harness for the wt_humanoid plugin.

The library item registration imports ``witwin.Library`` at package import time in
the real host. Unit tests stub that narrow surface so adapter/component tests do
not require the full application runtime.
"""
import sys
import types
from pathlib import Path


PLUGINS_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = Path(__file__).resolve().parents[3]
SERVER_DIR = REPO_DIR / "server"
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

try:
    import witwin  # noqa: F401
    _HAS_WITWIN = True
except Exception:
    _HAS_WITWIN = False

if "witwin" not in sys.modules and not _HAS_WITWIN:
    witwin = types.ModuleType("witwin")

    class _Library:
        registered = []
        categories = []

        @classmethod
        def register_category(cls, *args, **kwargs):
            cls.categories.append((args, kwargs))

        @classmethod
        def register_item(cls, *args, **kwargs):
            cls.registered.append((args, kwargs))

    witwin.Library = _Library
    sys.modules["witwin"] = witwin
