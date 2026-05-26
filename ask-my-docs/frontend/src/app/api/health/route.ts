const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function GET() {
  const upstream = await fetch(`${BACKEND}/health`);
  const data = await upstream.json();
  return Response.json(data, { status: upstream.status });
}
