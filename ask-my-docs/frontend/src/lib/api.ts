import type { HealthStatus, IngestStatus } from "./types";

const BASE = "/api";

export async function getHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function postIngest(file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Ingest failed: ${res.status}`);
  }
  return res.json();
}

export async function getIngestStatus(jobId: string): Promise<IngestStatus> {
  const res = await fetch(`${BASE}/ingest/status/${jobId}`);
  if (!res.ok) throw new Error("Status check failed");
  return res.json();
}

export function queryStream(
  question: string,
  topKRetrieve = 20,
  topKRerank = 5,
): Promise<Response> {
  return fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k_retrieve: topKRetrieve,
      top_k_rerank: topKRerank,
    }),
  });
}
