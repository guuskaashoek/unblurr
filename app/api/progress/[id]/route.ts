import { NextRequest, NextResponse } from "next/server";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id?: string }> }
) {
  const { id } = await params;
  if (!id) {
    return NextResponse.json({ error: "Missing job id" }, { status: 400 });
  }
  const jobId = id;
  const backendUrl = `http://localhost:5000/api/progress/${encodeURIComponent(jobId)}`;

  try {
    const response = await fetch(backendUrl, {
      method: "GET",
      cache: "no-store",
    });

    const body = await response.text();
    const headers = new Headers();
    headers.set("Content-Type", response.headers.get("content-type") ?? "application/json");

    return new NextResponse(body, {
      status: response.status,
      headers,
    });
  } catch (error) {
    console.error("Progress proxy error:", error);
    return NextResponse.json(
      { error: "Progress service unavailable" },
      { status: 502 }
    );
  }
}
