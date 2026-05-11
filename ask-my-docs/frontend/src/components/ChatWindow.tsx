"use client";

import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import type { Message } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  onSend: (question: string) => void;
  activeSource: number | null;
  onSourceClick: (pageNumber: number) => void;
}

export function ChatWindow({
  messages,
  isLoading,
  error,
  onSend,
  activeSource,
  onSourceClick,
}: ChatWindowProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const q = input.trim();
    if (!q || isLoading) return;
    setInput("");
    onSend(q);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-400 text-sm">Ask a question about your document.</p>
          </div>
        )}
        {messages.map((msg, i) => {
          const isLastAssistant =
            msg.role === "assistant" && i === messages.length - 1;
          return (
            <MessageBubble
              key={i}
              message={msg}
              isStreaming={isLastAssistant && isLoading}
              activeSource={activeSource}
              onSourceClick={onSourceClick}
            />
          );
        })}
        <div ref={bottomRef} />
      </div>

      {error && (
        <div className="px-4 py-2 bg-red-50 border-t border-red-100 text-xs text-red-600 shrink-0">
          {error}
        </div>
      )}

      <div className="border-t border-gray-200 bg-white px-4 py-3 flex items-end gap-2 shrink-0">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          rows={1}
          className="flex-1 resize-none rounded-xl border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-300 scrollbar-thin"
          style={{ maxHeight: "8rem", overflowY: "auto" }}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
          className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
        >
          {isLoading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
