"""The science layer.

Every function here is deterministic and computed - no trained models, no
predicted endpoints. If Winnow reports a number, that number is reproducible
from the input structure and a named literature rule.

This package is deliberately free of FastAPI imports. It must be usable from a
worker process, a notebook, or a CLI without an app instance.
"""
