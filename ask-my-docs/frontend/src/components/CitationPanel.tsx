"use client";

import type { Source } from "@/lib/types";

const SOURCE_COLORS: Record<Source["retrieval_source"], string> = {
  dense: "bg-blue-100 text-blue-700",
  sparse: "bg-green-100 text-green-700",
  both: "bg-purple-100 text-purple-700",
};

interface CitationPanelProps {
  sources: Source[];
  activePageNumber: number | null;
  onSourceSelect: (pageNumber: number | null) => void;
}

export function CitationPanel({
  sources,
  activePageNumber,
  onSourceSelect,
}: CitationPanelProps) {
  if (sources.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm px-6 text-center">
        Sources will appear here after you ask a question.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 p-4 overflow-y-auto h-full scrollbar-thin">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
        Sources ({sources.length})
      </h2>
      {sources.map((source, i) => {
        const isActive = activePageNumber === source.page_number;
        const scoreBarWidth =
          source.rerank_score !== null
            ? Math.min(100, Math.max(0, ((source.rerank_score + 10) / 20) * 100))
            : 0;

        return (
          <button
            key={i}
            onClick={() => onSourceSelect(isActive ? null : source.page_number)}
            className={`text-left rounded-lg border p-3 transition-all ${
              isActive
                ? "border-indigo-400 bg-indigo-50 shadow-sm"
                : "border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm"
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-mono font-semibold text-gray-700">
                p.{source.page_number}
              </span>
              <span
                className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                  SOURCE_COLORS[source.retrieval_source]
                }`}
              >
                {source.retrieval_source}
              </span>
            </div>
            {source.section_title && (
              <p className="text-xs font-medium text-gray-800 mb-1 truncate">
                {source.section_title}
              </p>
            )}
            <p className="text-xs text-gray-500 line-clamp-3 leading-relaxed">
              {source.text_preview}
            </p>
            {source.rerank_score !== null && (
              <div className="mt-2">
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-xs text-gray-400">Rerank score</span>
                  <span className="text-xs text-gray-500 font-mono">
                    {source.rerank_score.toFixed(3)}
                  </span>
                </div>
                <div className="h-1 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-400 rounded-full transition-all duration-300"
                    style={{ width: `${scoreBarWidth}%` }}
                  />
                </div>
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}
