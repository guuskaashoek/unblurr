import { randomUUID } from "crypto";
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const model = (formData.get("model") as string) ?? "detail";
  const jobId = (formData.get("jobId") as string) ?? randomUUID();

    if (!file) {
      return NextResponse.json(
        { error: "No file provided" },
        { status: 400 }
      );
    }

    // Get original filename
    const originalName = file.name;
    
    console.log(
      `Forwarding image to Python backend: ${originalName} (model: ${model}, job: ${jobId})`
    );

    // Create form data to send to Python backend
    const pythonFormData = new FormData();
    pythonFormData.append("file", file);
    pythonFormData.append("model", model);
    pythonFormData.append("jobId", jobId);

    // Call Python Flask backend
    const response = await fetch("http://localhost:5000/api/unblur", {
      method: "POST",
      body: pythonFormData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || "Python backend failed");
    }

    const blob = await response.blob();
    
    console.log(`Successfully processed: ${originalName}`);

    // Return the processed image
    return new NextResponse(blob, {
      headers: {
        "Content-Type": file.type,
        "Content-Disposition": `attachment; filename="${originalName}"`,
        "Cache-Control": "no-cache",
      },
    });
  } catch (error) {
    console.error("Unblur error:", error);
    return NextResponse.json(
      { error: `Failed to process image: ${error instanceof Error ? error.message : "Unknown error"}` },
      { status: 500 }
    );
  }
}
