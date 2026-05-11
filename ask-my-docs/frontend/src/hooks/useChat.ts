"use client";

import { useCallback, useState } from "react";
import { createParser, type ParsedEvent } from "eventsource-parser";
import { queryStream } from "@/lib/api";
import type { Message, Source } from "@/lib/types";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async (question: string) => {
    if (isLoading || !question.trim()) return;

    setError(null);
    setIsLoading(true);

    // Append user message and an empty assistant placeholder
    const userMsg: Message = {
      role: "user",
      content: question,
      sources: [],
      citation_valid: true,
    };
    const assistantMsg: Message = {
      role: "assistant",
      content: "",
      sources: [],
      citation_valid: true,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    try {
      const res = await queryStream(question);
      if (!res.ok || !res.body) {
        throw new Error(`Server error: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      const parser = createParser((event: ParsedEvent | { type: "reconnect-interval"; value: number }) => {
        if (event.type !== "event") return;
        try {
          const data = JSON.parse((event as ParsedEvent).data);

          if (data.type === "delta") {
            setMessages((prev) => {
              const next = [...prev];
              const last = { ...next[next.length - 1] };
              last.content += data.text as string;
              next[next.length - 1] = last;
              return next;
            });
          } else if (data.type === "sources") {
            setMessages((prev) => {
              const next = [...prev];
              const last = { ...next[next.length - 1] };
              last.sources = data.sources as Source[];
              next[next.length - 1] = last;
              return next;
            });
          } else if (data.type === "done") {
            setMessages((prev) => {
              const next = [...prev];
              const last = { ...next[next.length - 1] };
              last.citation_valid = data.citation_valid as boolean;
              next[next.length - 1] = last;
              return next;
            });
          } else if (data.type === "error") {
            setError(data.message as string);
          }
        } catch {
          // ignore malformed SSE data lines
        }
      });

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        parser.feed(decoder.decode(value, { stream: true }));
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      setMessages((prev) => {
        const next = [...prev];
        const last = { ...next[next.length - 1] };
        last.content = `Error: ${msg}`;
        next[next.length - 1] = last;
        return next;
      });
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  return { messages, sendMessage, isLoading, error };
}
