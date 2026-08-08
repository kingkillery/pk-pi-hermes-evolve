"""Minimal pytest suite for the Python DSPy backend sidecar.

These tests exercise pure helpers only (no dspy, no subprocess, no network), so
they run in CI without the optional heavyweight deps and give us a signal on
regressions in `backend.py` before users see them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pk_pi_hermes_evolve import backend


# ── Secret scanning ──────────────────────────────────────────────────────────

class TestSecretScanning:
    def test_detects_anthropic_key(self) -> None:
        text = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        found = backend.scan_for_secrets(text)
        names = {entry["pattern"] for entry in found}
        assert "anthropic-key" in names

    def test_detects_env_var_references(self) -> None:
        text = "export ANTHROPIC_API_KEY=xxx and DATABASE_URL=yyy"
        names = {entry["pattern"] for entry in backend.scan_for_secrets(text)}
        assert "env-anthropic" in names
        assert "env-database" in names

    def test_clean_text_yields_no_findings(self) -> None:
        assert backend.scan_for_secrets("hello world, nothing secret here") == []

    def test_strip_secrets_replaces_matches(self) -> None:
        text = "token=abcdefghijklmnop and normal text"
        cleaned = backend.strip_secrets(text)
        assert "abcdefghijklmnop" not in cleaned
        assert "[REDACTED]" in cleaned

    def test_patterns_load_from_shared_json(self) -> None:
        # Locate the shared JSON file that TS and Python both consume.
        shared = backend._find_shared_patterns_file()
        assert shared is not None, "src/secret-patterns.json should be discoverable"
        data = json.loads(shared.read_text(encoding="utf-8"))
        shared_names = {entry["name"] for entry in data["patterns"]}
        loaded_names = {name for name, _ in backend.SECRET_PATTERNS}
        assert shared_names == loaded_names, (
            "Loaded SECRET_PATTERNS must match the shared JSON exactly. "
            f"Missing in loaded: {shared_names - loaded_names}. "
            f"Extra in loaded: {loaded_names - shared_names}."
        )


# ── Rubric selection ─────────────────────────────────────────────────────────

class TestRubricPresets:
    @pytest.mark.parametrize("kind", ["skill", "prompt", "instructions"])
    def test_all_expected_presets_present(self, kind: str) -> None:
        assert kind in backend.RUBRIC_PRESETS
        assert backend.RUBRIC_PRESETS[kind].strip(), "preset must not be empty"

    def test_detect_type_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "review"
        skill_dir.mkdir(parents=True)
        target = skill_dir / "SKILL.md"
        target.write_text("# skill\n", encoding="utf-8")
        assert backend._detect_type(target) == "skill"

    def test_detect_type_prompt(self, tmp_path: Path) -> None:
        # Prompt detection matches `/.pi/prompts/` or `/.agents/prompts/` in the path.
        prompt_dir = tmp_path / ".pi" / "prompts"
        prompt_dir.mkdir(parents=True)
        target = prompt_dir / "template.md"
        target.write_text("prompt\n", encoding="utf-8")
        assert backend._detect_type(target) == "prompt"

    def test_detect_type_instructions_default(self, tmp_path: Path) -> None:
        target = tmp_path / "AGENTS.md"
        target.write_text("agents\n", encoding="utf-8")
        assert backend._detect_type(target) == "instructions"


# ── Pure helpers ─────────────────────────────────────────────────────────────

class TestPureHelpers:
    def test_slugify_basic(self) -> None:
        assert backend._slugify("Hello World!") == "hello-world"

    def test_slugify_truncates(self) -> None:
        long = "a" * 200
        assert len(backend._slugify(long)) <= 48

    def test_clamp_score_bounds(self) -> None:
        assert backend._clamp_score(1.5) == 1.0
        assert backend._clamp_score(-0.2) == 0.0
        # Non-numeric input falls back to a neutral 0.5 (see backend._clamp_score).
        assert backend._clamp_score("bogus") == 0.5
        assert backend._clamp_score(0.42) == pytest.approx(0.42)

    def test_split_and_reassemble_frontmatter(self) -> None:
        original = "---\ntitle: X\n---\nbody line 1\nbody line 2\n"
        fm, body = backend._split_frontmatter(original)
        assert fm is not None
        assert "title: X" in fm
        assert body.startswith("body line 1")
        recombined = backend._reassemble(fm, body)
        assert "title: X" in recombined
        assert "body line 1" in recombined

    def test_split_frontmatter_none_when_absent(self) -> None:
        fm, body = backend._split_frontmatter("no frontmatter here\n")
        assert fm is None
        assert body.startswith("no frontmatter")

    def test_extract_placeholders(self) -> None:
        text = "Use {{name}} and {{  value  }} and {{name}} again."
        found = backend._extract_placeholders(text)
        # Backend returns the raw {{...}} spans deduplicated and sorted.
        assert "{{name}}" in found
        assert any("value" in item for item in found)

    def test_max_bytes_grows_with_original(self) -> None:
        assert backend._max_bytes(1000) > 1000
        assert backend._max_bytes(0) > 0

    def test_extract_json_payload_plain(self) -> None:
        payload = backend._extract_json_payload('{"a": 1}')
        assert payload == {"a": 1}

    def test_extract_json_payload_fenced(self) -> None:
        wrapped = "```json\n{\"a\": 2}\n```"
        payload = backend._extract_json_payload(wrapped)
        assert payload == {"a": 2}
