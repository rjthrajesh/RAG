import sys
import types
from pathlib import Path

# Ensure `app` is importable when running pytest from the backend/ directory
sys.path.insert(0, str(Path(__file__).parent))

# Stub heavy ML libs that aren't installed in the test environment.
# These are replaced by MagicMock/object stubs so imports resolve without wheels.
if "sentence_transformers" not in sys.modules:
    _st = types.ModuleType("sentence_transformers")
    _st.SentenceTransformer = object  # type: ignore[attr-defined]
    _st.CrossEncoder = object  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = _st
