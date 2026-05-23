"""
CI eval gate — reads eval_report.json produced by ragas_runner.py and asserts thresholds.

Run order in CI:
  1. python -m app.evaluation.ragas_runner   →  writes eval_report.json
  2. pytest tests/eval/test_ragas.py         →  this file asserts on the report
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings

# Runner writes the report relative to its working directory (backend/)
_REPORT = Path(__file__).parent.parent.parent / "eval_report.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not _REPORT.exists():
        pytest.skip("eval_report.json not found — run 'python -m app.evaluation.ragas_runner' first")
    return json.loads(_REPORT.read_text())


def test_faithfulness_meets_threshold(report: dict) -> None:
    score = report["faithfulness"]
    threshold = settings.eval_faithfulness_threshold
    assert score >= threshold, (
        f"Faithfulness {score:.3f} below threshold {threshold}. "
        "Check retrieval quality and citation enforcement."
    )


def test_answer_relevancy_meets_threshold(report: dict) -> None:
    score = report["answer_relevancy"]
    threshold = settings.eval_answer_relevancy_threshold
    assert score >= threshold, (
        f"Answer relevancy {score:.3f} below threshold {threshold}. "
        "Check prompt construction and question coverage."
    )


def test_eval_overall_passed(report: dict) -> None:
    assert report["passed"], (
        f"Evaluation did not pass all thresholds:\n{json.dumps(report, indent=2)}"
    )


def test_context_precision_nonzero(report: dict) -> None:
    assert report["context_precision"] > 0.0, (
        "Context precision is 0 — retrieval may be returning irrelevant chunks"
    )


def test_context_recall_nonzero(report: dict) -> None:
    assert report["context_recall"] > 0.0, (
        "Context recall is 0 — check that the document was ingested correctly"
    )
