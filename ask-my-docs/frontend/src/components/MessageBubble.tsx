"use client";

import { Fragment } from "react";
import type { Message } from "@/lib/types";
import { SourceChip } from "./SourceChip";

interface MessageBubbleProps {
  message: Message;
  isStreaming: boolean;
  activeSource: number | null;
  onSourceClick: (pageNumber: number) => void;
}

function parseContent(
  content: string,
  activeSource: number | null,
  onSourceClick: (pageNumber: number) => void
): React.ReactNode[] {
  const parts = content.split(/(\[p\.\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[p\.(\d+)\]$/);
    if (match) {
      const page = parseInt(match[1], 10);
      return (
        <SourceChip
          key={i}
          pageNumber={page}
          isActive={activeSource === page}
          onClick={() => onSourceClick(page)}
        />
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

export function MessageBubble({
  message,
  isStreaming,
  activeSource,
  onSourceClick,
}: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%]">
        {!message.citation_valid && !isStreaming && message.content && (
          <div className="mb-1.5 flex items-center gap-1.5 rounded-md bg-yellow-50 px-3 py-1.5 text-xs text-yellow-700 border border-yellow-200">
            <span>&#9888;</span>
            <span>Some claims could not be verified against the source documents.</span>
          </div>
        )}
        <div className="rounded-2xl rounded-tl-sm bg-white border border-gray-200 px-4 py-2.5 text-sm text-gray-800 shadow-sm">
          {message.content ? (
            <span className="leading-relaxed whitespace-pre-wrap">
              {parseContent(message.content, activeSource, onSourceClick)}
            </span>
          ) : !isStreaming ? (
            <span className="text-gray-400 italic">No response.</span>
          ) : null}
          {isStreaming && (
            <span className="inline-block w-2 h-4 bg-gray-400 ml-0.5 animate-pulse rounded-sm align-middle" />
          )}
        </div>
      </div>
    </div>
  );
}
