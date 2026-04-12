from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pk_pi_hermes_evolve.backend import doctor, run_backend  # noqa: E402


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "doctor"

    try:
        if command == "doctor":
            print(json.dumps(doctor()))
            return 0
        if command == "run":
            payload = json.loads(sys.stdin.read() or "{}")
            print(json.dumps(run_backend(payload)))
            return 0
        print(json.dumps({"error": f"Unknown command: {command}"}))
        return 2
    except Exception as exc:  # pragma: no cover - CLI surface
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
