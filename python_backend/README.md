# pk-pi-hermes-evolve Python backend

This optional backend adds a real Python-side DSPy/GEPA path to the npm package.

Install manually in a Python environment:

```bash
cd python_backend
pip install -e .
```

This environment now also includes the OpenTelemetry packages used by `scripts/ralph_otel.py`, the traced Ralph loop for Hermes-parity gap closure work.

The TypeScript extension can then detect and invoke `run_backend.py` automatically when the Python dependencies are available.
