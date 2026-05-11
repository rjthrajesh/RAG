from __future__ import annotations

from app.retrieval.vector_store import RetrievalResult

_CHUNK_PREVIEW_LIMIT = 600

_SYSTEM = """\
You are an expert assistant for the book "AI Engineering: Building Applications \
with Foundation Models" by Chip Huyen.

Answer the user's question using ONLY the provided context passages below.
You MUST cite every factual claim using the format [p.{page_number}].
If the answer cannot be found in the context, say exactly:
"I cannot find information about this in the provided context."
Do NOT use any knowledge outside the provided passages.\
"""


class PromptBuilder:
    def build_rag_prompt(self, query: str, results: list[RetrievalResult]) -> str:
        context_lines: list[str] = ["CONTEXT PASSAGES:"]
        for i, r in enumerate(results, start=1):
            title = r.section_title or "General"
            text = r.text[:_CHUNK_PREVIEW_LIMIT]
            context_lines.append(f"[{i}] (Page {r.page_number}, {title}):\n{text}")

        context_block = "\n\n".join(context_lines)

        return (
            f"{_SYSTEM}\n\n"
            f"{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER (cite every claim with [p.X]):"
        )
