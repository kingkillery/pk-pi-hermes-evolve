"""Parity check for the shared secret-patterns JSON.

Intentionally does NOT import `pk_pi_hermes_evolve.backend` so it runs even
when the optional `dspy` dependency isn't installed. This gives CI a signal
that the TS engine and the Python fallback list stay aligned with the
canonical `src/secret-patterns.json`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_JSON = REPO_ROOT / "src" / "secret-patterns.json"


def _fallback_names() -> set[str]:
    # Duplicated verbatim so we don't need to import backend.py (which pulls
    # in dspy at module scope). Keep this in sync with
    # `_FALLBACK_SECRET_PATTERNS` in python_backend/pk_pi_hermes_evolve/backend.py.
    backend_py = REPO_ROOT / "python_backend" / "pk_pi_hermes_evolve" / "backend.py"
    source = backend_py.read_text(encoding="utf-8")
    match = re.search(
        r"_FALLBACK_SECRET_PATTERNS: list\[tuple\[str, str\]\] = \[(.*?)\]\n",
        source,
        re.DOTALL,
    )
    assert match, "Could not locate _FALLBACK_SECRET_PATTERNS block in backend.py"
    body = match.group(1)
    return set(re.findall(r'\("([^"]+)",', body))


def test_shared_patterns_json_exists() -> None:
    assert SHARED_JSON.is_file(), f"Missing shared JSON at {SHARED_JSON}"


def test_shared_json_and_python_fallback_have_identical_names() -> None:
    data = json.loads(SHARED_JSON.read_text(encoding="utf-8"))
    shared_names = {entry["name"] for entry in data["patterns"]}
    fallback_names = _fallback_names()
    assert shared_names == fallback_names, (
        "Shared secret-patterns.json and Python fallback list drifted. "
        f"Missing in fallback: {shared_names - fallback_names}. "
        f"Extra in fallback: {fallback_names - shared_names}."
    )


def test_all_patterns_compile_as_regex() -> None:
    data = json.loads(SHARED_JSON.read_text(encoding="utf-8"))
    for entry in data["patterns"]:
        # Each pattern must be a valid Python regex, since backend.py compiles
        # them at load time.
        re.compile(entry["pattern"])
