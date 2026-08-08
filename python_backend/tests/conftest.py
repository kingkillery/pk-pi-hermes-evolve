# Ensure pytest discovers `python_backend/tests` without needing an install.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# The backend module defines dspy.Signature subclasses at module scope, so
# `import pk_pi_hermes_evolve.backend` fails hard when dspy isn't installed.
# Collect-time skip lets the same suite run in a lightweight CI job without the
# heavyweight optional dep while still running locally when dspy is available.
# We only skip tests that actually import backend; parity checks that read
# source files directly still run.
try:
    import dspy  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - CI-only branch
    collect_ignore = ["test_backend.py"]
