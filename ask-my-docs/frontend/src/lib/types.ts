export interface Source {
  page_number: number;
  section_title: string | null;
  text_preview: string;
  rerank_score: number | null;
  retrieval_source: "dense" | "sparse" | "both";
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  citation_valid: boolean;
}

export interface IngestStatus {
  status: "processing" | "done" | "failed";
  progress: number;
  pages?: number;
  chunks?: number;
  duration_seconds?: number;
  error?: string;
}

export interface HealthStatus {
  status: string;
  ollama: boolean;
  chroma: boolean;
}
