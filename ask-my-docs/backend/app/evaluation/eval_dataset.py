from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class EvalEntry(TypedDict):
    question: str
    ground_truth: str
    relevant_pages: list[int]


def load_eval_dataset(path: str) -> list[EvalEntry]:
    """Load golden dataset from JSON, skipping any entries with TODO ground_truth."""
    raw: list[EvalEntry] = json.loads(Path(path).read_text())

    ready = [e for e in raw if not e["ground_truth"].strip().upper().startswith("TODO")]
    skipped = len(raw) - len(ready)
    if skipped:
        logger.warning(
            "Skipped %d/%d dataset entries — ground_truth not yet populated (TODO marker)",
            skipped,
            len(raw),
        )
    if not ready:
        raise ValueError(
            f"No eval-ready entries in {path}. "
            "Populate ground_truth fields in golden_dataset.json first."
        )

    logger.info("Loaded %d eval entries from %s", len(ready), path)
    return ready
