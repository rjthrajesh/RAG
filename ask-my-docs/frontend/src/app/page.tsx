"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";
import { useChat } from "@/hooks/useChat";
import { ChatWindow } from "@/components/ChatWindow";
import { CitationPanel } from "@/components/CitationPanel";
import { IngestionUploader } from "@/components/IngestionUploader";
import type { Source } from "@/lib/types";

export default function Home() {
  const [docsReady, setDocsReady] = useState(false);
  const [backendDown, setBackendDown] = useState(false);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const { messages, sendMessage, isLoading, error } = useChat();

  useEffect(() => {
    getHealth().catch(() => setBackendDown(true));
  }, []);

  const handleSourceClick = (pageNumber: number) => {
    setActiveSource((prev) => (prev === pageNumber ? null : pageNumber));
  };

  const lastAssistantMsg = [...messages]
    .reverse()
    .find((m) => m.role === "assistant");
  const sources: Source[] = lastAssistantMsg?.sources ?? [];

  return (
    <div className="h-full flex flex-col">
      {backendDown && (
        <div className="shrink-0 bg-red-50 border-b border-red-200 px-4 py-2 text-xs text-red-600 text-center">
          Backend unreachable — make sure the API server is running on port 8000.
        </div>
      )}

      {!docsReady ? (
        <div className="flex-1 overflow-hidden">
          <IngestionUploader onComplete={() => setDocsReady(true)} />
        </div>
      ) : (
        <div className="flex flex-1 overflow-hidden">
          {/* Left panel — 60% */}
          <div className="w-3/5 flex flex-col border-r border-gray-200 bg-gray-50 overflow-hidden">
            <header className="shrink-0 px-5 py-3 border-b border-gray-200 bg-white flex items-center justify-between">
              <span className="font-semibold text-gray-800">Ask My Docs</span>
              <button
                onClick={() => {
                  setDocsReady(false);
                  setActiveSource(null);
                }}
                className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
              >
                Upload new doc
              </button>
            </header>
            <div className="flex-1 overflow-hidden">
              <ChatWindow
                messages={messages}
                isLoading={isLoading}
                error={error}
                onSend={sendMessage}
                activeSource={activeSource}
                onSourceClick={handleSourceClick}
              />
            </div>
          </div>

          {/* Right panel — 40% */}
          <div className="w-2/5 flex flex-col bg-white overflow-hidden">
            <header className="shrink-0 px-5 py-3 border-b border-gray-200">
              <span className="font-semibold text-gray-800 text-sm">Sources</span>
            </header>
            <div className="flex-1 overflow-hidden">
              <CitationPanel
                sources={sources}
                activePageNumber={activeSource}
                onSourceSelect={setActiveSource}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
