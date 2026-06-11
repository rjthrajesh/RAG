export const dynamic = "force-dynamic";

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function GET(
  _req: Request,
  { params }: { params: { jobId: string } },
) {
  try {
    const upstream = await fetch(`${BACKEND}/ingest/status/${params.jobId}`);
    const data = await upstream.json();
    return Response.json(data, { status: upstream.status });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "upstream error";
    return Response.json({ error: msg }, { status: 502 });
  }
}
