export const dynamic = "force-dynamic";

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const NO_CACHE = { "Cache-Control": "no-store" };

export async function GET(
  _req: Request,
  { params }: { params: { jobId: string } },
) {
  try {
    const upstream = await fetch(`${BACKEND}/ingest/status/${params.jobId}`, {
      cache: "no-store",
    });
    const data = await upstream.json();
    return Response.json(data, { status: upstream.status, headers: NO_CACHE });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "upstream error";
    return Response.json({ error: msg }, { status: 502, headers: NO_CACHE });
  }
}
