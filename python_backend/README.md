# pk-pi-hermes-evolve Python backend

This optional backend adds a real Python-side DSPy/GEPA path to the npm package.

Install manually in a Python environment:

```bash
cd python_backend
pip install -e .
```

The TypeScript extension can then detect and invoke `run_backend.py` automatically when the Python dependencies are available.
