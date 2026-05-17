"""
examples/_path.py — add the project root to ``sys.path``.

Importing this module makes ``chocospdc`` importable regardless of
where the script is launched from (project root, examples/, anywhere).
Each example does ``import _path  # noqa`` near the top.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
