export const dynamic = "force-dynamic";

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function POST(req: Request) {
  const form = await req.formData();
  const upstream = await fetch(`${BACKEND}/ingest`, {
    method: "POST",
    body: form,
  });
  const data = await upstream.json();
  return Response.json(data, { status: upstream.status });
}
