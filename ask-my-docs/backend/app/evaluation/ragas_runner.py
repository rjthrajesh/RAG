from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from app.config import settings
from app.evaluation.eval_dataset import load_eval_dataset
from app.generation.llm_client import GroqClient, OllamaClient
from app.generation.prompt_builder import PromptBuilder
from app.ingestion.embedder import Embedder
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_REPORT_FILENAME = "eval_report.json"
_DATASET_PATH = (
    Path(__file__).parent.parent.parent / "tests" / "eval" / "golden_dataset.json"
)


@dataclass
class EvalReport:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    passed: bool


# ---------------------------------------------------------------------------
# Custom LangChain chat model backed by completion_sync (plain httpx.Client)
#
# langchain_openai.ChatOpenAI uses the openai SDK's anyio/httpcore async stack,
# which fails with ConnectError inside RAGAS's background thread on macOS.
# Using completion_sync (httpx.Client) avoids this entirely: BaseChatModel's
# default _agenerate runs _generate via run_in_executor (thread pool), so
# RAGAS's async executor gets a proper awaitable without touching anyio.
# ---------------------------------------------------------------------------

class _SyncChatModel(BaseChatModel):
    """LangChain chat model that calls completion_sync in a thread pool."""

    api_key: str
    model: str
    provider: str  # "groq" | "ollama"
    ollama_base_url: str = ""

    @property
    def _llm_type(self) -> str:
        return f"{self.provider}-sync"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}
        chat_messages = [
            {"role": _ROLE_MAP.get(m.type, "user"), "content": str(m.content)}
            for m in messages
        ]
        if self.provider == "groq":
            text = GroqClient(api_key=self.api_key, model=self.model).completion_sync_messages(chat_messages)
        else:
            text = OllamaClient(base_url=self.ollama_base_url, model=self.model).completion_sync_messages(chat_messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _make_ragas_llm() -> LangchainLLMWrapper:
    if settings.llm_provider == "groq":
        # Use judge model (larger, better at structured JSON) rather than the fast app model
        chat = _SyncChatModel(
            api_key=settings.groq_api_key,
            model=settings.groq_judge_model,
            provider="groq",
        )
    else:
        chat = _SyncChatModel(
            api_key="",
            model=settings.ollama_model,
            provider="ollama",
            ollama_base_url=settings.ollama_base_url,
        )
    return LangchainLLMWrapper(chat)


def _make_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Use the same local BGE model as retrieval — no external API needed."""
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embed_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf_embeddings)


def _make_generation_client() -> GroqClient | OllamaClient:
    if settings.llm_provider == "groq":
        return GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)
    return OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model)


class RAGASRunner:
    def __init__(self) -> None:
        embedder = Embedder(model_name=settings.embed_model_name)
        bm25_store = BM25Store()
        bm25_path = Path(settings.bm25_index_path)
        if bm25_path.exists():
            bm25_store.load(str(bm25_path))
        else:
            logger.warning("BM25 index not found at %s — sparse retrieval will be empty", bm25_path)

        self._retriever = HybridRetriever(
            vector_store=VectorStore(),
            bm25_store=bm25_store,
            embedder=embedder,
        )
        self._reranker = CrossEncoderReranker()
        self._llm = _make_generation_client()
        self._prompt_builder = PromptBuilder()

    def _run_pipeline(self, question: str) -> tuple[str, list[str]]:
        results = self._retriever.retrieve(question, top_k=settings.top_k_retrieve)
        reranked = self._reranker.rerank(question, results, top_k=settings.top_k_rerank)
        prompt = self._prompt_builder.build_rag_prompt(question, reranked)
        answer = self._llm.completion_sync(prompt)
        contexts = [r.text for r in reranked]
        return answer, contexts

    def run_eval(self, dataset_path: str = str(_DATASET_PATH)) -> EvalReport:
        entries = load_eval_dataset(dataset_path)
        logger.info("Running RAGAS evaluation on %d entries", len(entries))
        t0 = time.time()

        questions: list[str] = []
        answers: list[str] = []
        contexts_list: list[list[str]] = []
        ground_truths: list[str] = []

        for i, entry in enumerate(entries, start=1):
            logger.info("[%d/%d] %s", i, len(entries), entry["question"][:70])
            try:
                answer, contexts = self._run_pipeline(entry["question"])
            except Exception:
                logger.exception("Pipeline failed on entry %d — skipping", i)
                continue
            questions.append(entry["question"])
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(entry["ground_truth"])

        if not questions:
            raise RuntimeError("All pipeline runs failed — check ChromaDB and LLM connectivity")

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        })

        ragas_llm = _make_ragas_llm()
        ragas_embeddings = _make_ragas_embeddings()

        # max_workers=1 keeps Groq free-tier rate limits safe; timeout=180 for 70b latency
        run_config = RunConfig(max_workers=1, timeout=180)

        logger.info("Scoring %d answers with RAGAS metrics (max_workers=1)…", len(questions))
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )

        elapsed = round(time.time() - t0, 1)
        logger.info("RAGAS evaluation complete in %.1fs — %s", elapsed, result)

        faith = float(result["faithfulness"])
        relevancy = float(result["answer_relevancy"])
        precision = float(result["context_precision"])
        recall = float(result["context_recall"])

        passed = (
            faith >= settings.eval_faithfulness_threshold
            and relevancy >= settings.eval_answer_relevancy_threshold
        )

        report = EvalReport(
            faithfulness=faith,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
            passed=passed,
        )

        report_path = Path(_REPORT_FILENAME)
        report_path.write_text(json.dumps(asdict(report), indent=2))
        logger.info(
            "Report written to %s — passed=%s  faith=%.3f  relevancy=%.3f  precision=%.3f  recall=%.3f",
            report_path,
            report.passed,
            faith,
            relevancy,
            precision,
            recall,
        )
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    runner = RAGASRunner()
    report = runner.run_eval()

    print("\n=== RAGAS Evaluation Results ===")
    print(f"  Faithfulness:      {report.faithfulness:.3f}  (threshold ≥ {settings.eval_faithfulness_threshold})")
    print(f"  Answer Relevancy:  {report.answer_relevancy:.3f}  (threshold ≥ {settings.eval_answer_relevancy_threshold})")
    print(f"  Context Precision: {report.context_precision:.3f}")
    print(f"  Context Recall:    {report.context_recall:.3f}")
    print(f"  Result: {'PASSED ✓' if report.passed else 'FAILED ✗'}")
