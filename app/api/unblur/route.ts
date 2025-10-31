import { randomUUID } from "crypto";
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const model = (formData.get("model") as string) ?? "detail";
    const mode = (formData.get("mode") as string) ?? "upscale";
    const jobId = (formData.get("jobId") as string) ?? randomUUID();
    const cropDimensionsStr = formData.get("cropDimensions") as string | null;

    if (!file) {
      return NextResponse.json(
        { error: "No file provided" },
        { status: 400 }
      );
    }

    // Get original filename
    const originalName = file.name;
    
    console.log(
      `Forwarding image to Python backend: ${originalName} (model: ${model}, mode: ${mode}, job: ${jobId})`
    );

    // Create form data to send to Python backend
    const pythonFormData = new FormData();
    pythonFormData.append("file", file);
    pythonFormData.append("model", model);
    pythonFormData.append("mode", mode);
    pythonFormData.append("jobId", jobId);
    
    if (cropDimensionsStr) {
      pythonFormData.append("cropDimensions", cropDimensionsStr);
    }

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

    // Return the processed image or zip file
    const contentType = blob.type;
    const fileName = mode === "crop" 
      ? `${originalName.replace(/\.[^/.]+$/, "")}_crops.zip`
      : originalName;

    return new NextResponse(blob, {
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": `attachment; filename="${fileName}"`,
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
