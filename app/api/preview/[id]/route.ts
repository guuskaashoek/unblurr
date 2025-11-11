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
  const backendUrl = `http://localhost:5000/api/preview/${encodeURIComponent(jobId)}`;

  try {
    const response = await fetch(backendUrl, {
      method: "GET",
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "Preview not available" },
        { status: response.status }
      );
    }

    const blob = await response.blob();
    const headers = new Headers();
    headers.set("Content-Type", response.headers.get("content-type") ?? "image/jpeg");
    headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
    headers.set("Pragma", "no-cache");
    headers.set("Expires", "0");

    return new NextResponse(blob, {
      status: 200,
      headers,
    });
  } catch (error) {
    console.error("Preview proxy error:", error);
    return NextResponse.json(
      { error: "Preview service unavailable" },
      { status: 502 }
    );
  }
}

