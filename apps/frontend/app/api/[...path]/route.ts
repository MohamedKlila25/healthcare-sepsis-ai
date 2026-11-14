import { NextRequest } from 'next/server';

const allowed = /^(health|api\/v1\/(patients(?:\/DEMO-\d{3}\/timeline)?|predict|explain|agent\/query|models|evidence(?:\/pubmed)?|documents\/analyze))$/;

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const endpoint = path.join('/');
  if (!allowed.test(endpoint)) return Response.json({ detail: 'Unknown endpoint' }, { status: 404 });
  const base = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (process.env.HEALTHCARE_API_KEY) headers['X-API-Key'] = process.env.HEALTHCARE_API_KEY;
  try {
    const response = await fetch(`${base}/${endpoint}${request.nextUrl.search}`, {
      method: request.method, headers, cache: 'no-store', signal: AbortSignal.timeout(25000),
      ...(request.method === 'POST' ? { body: await request.text() } : {}),
    });
    return new Response(await response.text(), { status: response.status, headers: { 'Content-Type': 'application/json' } });
  } catch {
    return Response.json({ detail: 'Backend unavailable. Start the Healthcare SepsisAI API on port 8000.' }, { status: 503 });
  }
}
export const GET = proxy;
export const POST = proxy;
