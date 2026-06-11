"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { postIngest, getIngestStatus } from "@/lib/api";
import type { IngestStatus } from "@/lib/types";

interface IngestionUploaderProps {
  onComplete: () => void;
}

export function IngestionUploader({ onComplete }: IngestionUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onCompleteRef = useRef(onComplete);
  useEffect(() => { onCompleteRef.current = onComplete; });

  const handleFile = useCallback(async (f: File) => {
    if (!f.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }
    setFile(f);
    setError(null);
    setStatus(null);
    setUploading(true);
    try {
      const { job_id } = await postIngest(f);
      setJobId(job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
      setUploading(false);
    }
  }, []);

  useEffect(() => {
    if (!jobId) return;

    pollRef.current = setInterval(async () => {
      try {
        const s = await getIngestStatus(jobId);
        setStatus(s);
        if (s.status === "done") {
          clearInterval(pollRef.current!);
          setUploading(false);
          setTimeout(() => onCompleteRef.current(), 1200);
        } else if (s.status === "failed") {
          clearInterval(pollRef.current!);
          setUploading(false);
          setError(s.error ?? "Ingestion failed.");
        }
      } catch {
        // transient poll error — keep trying
      }
    }, 1000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const progressPct = Math.round((status?.progress ?? 0) * 100);

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Ask My Docs</h1>
        <p className="mt-2 text-sm text-gray-500">
          Upload a PDF and ask questions about its contents.
        </p>
      </div>

      {!uploading && status?.status !== "done" && (
        <div
          onDrop={onDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => inputRef.current?.click()}
          className={`w-full max-w-sm rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition-colors select-none ${
            dragOver
              ? "border-indigo-500 bg-indigo-50"
              : "border-gray-300 hover:border-indigo-400 hover:bg-gray-50"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />
          <p className="text-3xl mb-3">&#128196;</p>
          <p className="text-sm font-medium text-gray-700">
            {file ? file.name : "Drop a PDF here or click to browse"}
          </p>
          <p className="text-xs text-gray-400 mt-1">Max 50 MB</p>
        </div>
      )}

      {uploading && (
        <div className="w-full max-w-sm space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-700 font-medium truncate max-w-[70%]">
              {file?.name}
            </span>
            <span className="text-gray-500 text-xs tabular-nums">
              {status ? `${progressPct}%` : "Uploading..."}
            </span>
          </div>
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <p className="text-xs text-gray-400">
            {status?.status === "processing"
              ? [
                  "Processing",
                  status.pages ? `${status.pages} pages` : null,
                  status.chunks ? `${status.chunks} chunks` : null,
                ]
                  .filter(Boolean)
                  .join(" · ")
              : "Preparing..."}
          </p>
        </div>
      )}

      {status?.status === "done" && (
        <div className="text-center space-y-1">
          <p className="text-lg font-semibold text-green-600">&#10003; Ready!</p>
          <p className="text-xs text-gray-500">
            {[
              status.pages ? `${status.pages} pages` : null,
              status.chunks ? `${status.chunks} chunks` : null,
              status.duration_seconds
                ? `${status.duration_seconds.toFixed(1)}s`
                : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </p>
        </div>
      )}

      {error && (
        <p className="text-sm text-red-600 text-center max-w-sm">{error}</p>
      )}
    </div>
  );
}
