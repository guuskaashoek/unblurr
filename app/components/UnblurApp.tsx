"use client";

import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, Loader2, Sparkles, Upload, X, Image, Scissors, Zap, FileText, Trash2, Plus } from "lucide-react";

type ModelKey = "detail" | "text";
type Mode = "upscale" | "crop";

interface ModelOption {
  title: string;
  description: string;
  accent: string;
}

interface TileProgress {
  completed: number;
  total: number;
}

interface CropDimension {
  id: string;
  width: number;
  height: number;
}

interface UploadedFile {
  id: string;
  file: File;
  status: "pending" | "processing" | "completed" | "error";
  error?: string;
  preview?: string;
  livePreview?: string;
  model: ModelKey;
  mode: Mode;
  cropDimensions?: CropDimension[];
  progress: number;
  tiles?: TileProgress;
  currentTile?: number;
  currentTileProgress?: number;
  backendStatus?: "pending" | "processing" | "completed" | "error";
  backendMessage?: string;
}

interface ProgressResponse {
  status: UploadedFile["backendStatus"];
  progress?: number;
  completed?: number;
  total?: number;
  current_tile?: number;
  current_tile_progress?: number;
  message?: string | null;
}

const MODEL_OPTIONS: Record<ModelKey, ModelOption> = {
  detail: {
    title: "Detail Enhancer",
    description: "Best for portraits, product shots, and natural photos.",
    accent: "from-blue-500 to-cyan-500",
  },
  text: {
    title: "Text & Logos",
    description: "Dialed-in for UI, scanned documents, and crisp lettering.",
    accent: "from-purple-500 to-pink-500",
  },
};

const PROGRESS_POLL_INTERVAL = 400;
const PROGRESS_EPSILON = 0.005;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function UnblurApp() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelKey>("detail");
  const [mode, setMode] = useState<Mode>("upscale");
  const [cropDimensions, setCropDimensions] = useState<CropDimension[]>([
    { id: "1", width: 1200, height: 628 },
    { id: "2", width: 1200, height: 1200 },
  ]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  const addFiles = (incoming: FileList | File[]) => {
    const fileArray = Array.from(incoming);
    const imageFiles = fileArray.filter((file) => file.type.startsWith("image/"));
    const timestamp = Date.now();

    const queued = imageFiles.map<UploadedFile>((file, index) => {
      const preview = URL.createObjectURL(file);
      return {
        id: `${file.name}-${timestamp}-${index}`,
        file,
        status: "pending",
        preview,
        model: selectedModel,
        mode: mode,
        cropDimensions: mode === "crop" ? cropDimensions : undefined,
        progress: 0,
        tiles: undefined,
        backendStatus: "pending",
        backendMessage: undefined,
        error: undefined,
      };
    });

    if (queued.length === 0) {
      return;
    }

    setFiles((previous) => [...previous, ...queued]);
    queued.forEach((file) => void processFile(file));
  };

  const processFile = async (uploadedFile: UploadedFile) => {
    setFiles((previous) =>
      previous.map((file) =>
        file.id === uploadedFile.id
          ? {
              ...file,
              status: "processing",
              backendStatus: "processing",
              progress: file.progress ?? 0,
            }
          : file,
      ),
    );

    let pollingActive = true;

    const startProgressPolling = async () => {
      // slight delay to allow the backend to register the job
      await sleep(150);
      while (pollingActive) {
        await sleep(PROGRESS_POLL_INTERVAL);
        try {
          const response = await fetch(`/api/progress/${uploadedFile.id}`, {
            cache: "no-store",
          });

          if (!response.ok) {
            if (response.status === 404) {
              continue;
            }
            break;
          }

          const data = (await response.json()) as ProgressResponse;
          setFiles((previous) => {
            let mutated = false;
            const next = previous.map((file) => {
              if (file.id !== uploadedFile.id) {
                return file;
              }

              const rawProgress = data.progress !== undefined ? Math.min(1, Math.max(0, data.progress)) : file.progress;
              let nextProgress = rawProgress;
              let nextStatus: UploadedFile["status"] = file.status;
              let nextError = file.error;

              if (data.status === "error") {
                nextStatus = "error";
                nextError = data.message ?? file.error ?? "Processing failed";
              } else if (data.status === "completed") {
                nextStatus = "completed";
                nextProgress = 1;
              }

              let nextTiles = file.tiles;
              if (data.completed !== undefined && data.total !== undefined) {
                const completed = data.completed;
                const total = data.total;
                if (!file.tiles || file.tiles.completed !== completed || file.tiles.total !== total) {
                  nextTiles = { completed, total };
                }
              }

              const nextCurrentTile = data.current_tile ?? file.currentTile;
              const nextCurrentTileProgress = data.current_tile_progress ?? file.currentTileProgress;

              const nextBackendStatus = data.status ?? file.backendStatus;
              const nextBackendMessage = data.message ?? file.backendMessage;
              
              // Set live preview URL if processing
              if (data.status === "processing" && nextCurrentTile && nextCurrentTile > 0) {
                // Use direct URL with timestamp to prevent caching
                const previewUrl = `/api/preview/${uploadedFile.id}?t=${Date.now()}`;
                setFiles((prev) =>
                  prev.map((f) =>
                    f.id === uploadedFile.id
                      ? { ...f, livePreview: previewUrl }
                      : f
                  )
                );
              }

              const progressChanged =
                typeof nextProgress === "number" && typeof file.progress === "number"
                  ? Math.abs(nextProgress - file.progress) > PROGRESS_EPSILON
                  : nextProgress !== file.progress;
              const tilesChanged = nextTiles !== file.tiles;
              const currentTileChanged = nextCurrentTile !== file.currentTile;
              const currentTileProgressChanged = 
                typeof nextCurrentTileProgress === "number" && typeof file.currentTileProgress === "number"
                  ? Math.abs(nextCurrentTileProgress - file.currentTileProgress) > PROGRESS_EPSILON
                  : nextCurrentTileProgress !== file.currentTileProgress;
              const backendStatusChanged = nextBackendStatus !== file.backendStatus;
              const backendMessageChanged = nextBackendMessage !== file.backendMessage;
              const statusChanged = nextStatus !== file.status;
              const errorChanged = nextError !== file.error;

              if (
                !progressChanged &&
                !tilesChanged &&
                !currentTileChanged &&
                !currentTileProgressChanged &&
                !backendStatusChanged &&
                !backendMessageChanged &&
                !statusChanged &&
                !errorChanged
              ) {
                return file;
              }

              mutated = true;
              return {
                ...file,
                progress: typeof nextProgress === "number" ? nextProgress : file.progress,
                tiles: nextTiles,
                currentTile: nextCurrentTile,
                currentTileProgress: nextCurrentTileProgress,
                backendStatus: nextBackendStatus,
                backendMessage: nextBackendMessage,
                status: nextStatus,
                error: nextError,
              };
            });

            return mutated ? next : previous;
          });

          if (data.status === "completed" || data.status === "error") {
            break;
          }
        } catch (err) {
          break;
        }
      }
    };

    void startProgressPolling();

    try {
      const formData = new FormData();
      formData.append("file", uploadedFile.file);
      formData.append("model", uploadedFile.model);
      formData.append("mode", uploadedFile.mode);
      formData.append("jobId", uploadedFile.id);

      if (uploadedFile.mode === "crop" && uploadedFile.cropDimensions) {
        formData.append("cropDimensions", JSON.stringify(uploadedFile.cropDimensions));
      }

      const response = await fetch("/api/unblur", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || "Failed to process image");
      }

      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = uploadedFile.mode === "crop" 
        ? `${uploadedFile.file.name.replace(/\.[^/.]+$/, "")}_crops.zip`
        : uploadedFile.file.name;
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();

      setTimeout(() => {
        document.body.removeChild(link);
        URL.revokeObjectURL(downloadUrl);
      }, 150);

      setFiles((previous) =>
        previous.map((file) =>
          file.id === uploadedFile.id
            ? {
                ...file,
                status: "completed",
                progress: 1,
                backendStatus: "completed",
              }
            : file,
        ),
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unexpected processing error";
      setFiles((previous) =>
        previous.map((file) =>
          file.id === uploadedFile.id
            ? { ...file, status: "error", error: message }
            : file,
        ),
      );
    } finally {
      pollingActive = false;
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDragEnter = (event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    if (event.target === dropZoneRef.current) {
      setIsDragging(false);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    const droppedFiles = event.dataTransfer.files;
    if (droppedFiles.length > 0) {
      addFiles(droppedFiles);
    }
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files;
    if (selected) {
      addFiles(selected);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const removeFile = (id: string) => {
    setFiles((previous) => {
      const file = previous.find((entry) => entry.id === id);
      if (file?.preview) {
        URL.revokeObjectURL(file.preview);
      }
      return previous.filter((entry) => entry.id !== id);
    });
  };

  const clearAll = () => {
    files.forEach((file) => file.preview && URL.revokeObjectURL(file.preview));
    setFiles([]);
  };

  const addCropDimension = () => {
    const newId = String(Math.max(0, ...cropDimensions.map(d => parseInt(d.id, 10))) + 1);
    setCropDimensions([...cropDimensions, { id: newId, width: 800, height: 600 }]);
  };

  const removeCropDimension = (id: string) => {
    if (cropDimensions.length > 1) {
      setCropDimensions(cropDimensions.filter(d => d.id !== id));
    }
  };

  const updateCropDimension = (id: string, field: "width" | "height", value: number) => {
    setCropDimensions(cropDimensions.map(d => 
      d.id === id ? { ...d, [field]: value } : d
    ));
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-slate-50 via-white to-blue-50/30">
      <motion.div
        aria-hidden
        className="pointer-events-none absolute -top-32 left-1/2 w-[60rem] h-[60rem] -translate-x-1/2 rounded-full bg-gradient-to-br from-blue-200/40 via-purple-200/30 to-pink-200/20 blur-3xl"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.4, ease: "easeOut" }}
      />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-7xl flex-col gap-8 px-6 pb-12 pt-8 md:pt-12">
        <header className="flex items-center justify-between">
          <motion.div
            className="flex items-center gap-2"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500 shadow-lg">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-slate-900">Unblurr</h1>
          </motion.div>
          {files.length > 0 && (
            <motion.button
              onClick={clearAll}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-2 rounded-xl border-2 border-red-200 bg-white px-4 py-2 text-sm font-semibold text-red-600 shadow-sm transition-all hover:border-red-300 hover:bg-red-50"
            >
              <Trash2 className="h-4 w-4" />
              Clear All
            </motion.button>
          )}
        </header>

        <section className="flex flex-col gap-6 rounded-3xl border border-slate-200/80 bg-white/80 p-6 shadow-xl shadow-slate-200/50 backdrop-blur-xl md:p-8">
          {/* Mode Selector */}
          <div className="grid grid-cols-2 gap-4">
            {(["upscale", "crop"] as Mode[]).map((m) => {
              const isActive = mode === m;
              return (
                <motion.button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`relative flex items-center gap-3 overflow-hidden rounded-2xl border-2 px-5 py-4 transition-all ${
                    isActive
                      ? "border-transparent text-white shadow-lg"
                      : "border-slate-200 bg-slate-50/50 text-slate-700 hover:border-slate-300 hover:bg-slate-100/70 hover:shadow-md"
                  }`}
                  whileTap={{ scale: 0.98 }}
                  whileHover={!isActive ? { scale: 1.01, y: -2 } : undefined}
                >
                    {isActive && (
                      <motion.span
                        layoutId="mode-accent"
                        className={`absolute inset-0 -z-10 ${
                          m === "upscale"
                            ? "bg-blue-500"
                            : "bg-amber-500"
                        }`}
                        initial={false}
                        transition={{ type: "spring", stiffness: 250, damping: 30 }}
                      />
                    )}
                  {m === "upscale" ? (
                    <Zap className={`h-5 w-5 ${isActive ? "text-white" : "text-blue-600"}`} />
                  ) : (
                    <Scissors className={`h-5 w-5 ${isActive ? "text-white" : "text-amber-600"}`} />
                  )}
                  <span className="text-sm font-bold">
                    {m === "upscale" ? "Upscale" : "Crop"}
                  </span>
                </motion.button>
              );
            })}
          </div>

          {/* Crop Dimensions Editor */}
          {mode === "crop" && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-4 rounded-2xl border border-amber-200/60 bg-amber-50/50 p-5 shadow-inner"
            >
              <div className="flex items-center gap-2">
                <Image className="h-4 w-4 text-amber-700" />
                <span className="text-sm font-bold text-slate-900">Dimensions</span>
              </div>
              <div className="space-y-3">
                {cropDimensions.map((dim) => (
                  <div key={dim.id} className="flex gap-3 items-center">
                    <input
                      type="number"
                      value={dim.width}
                      onChange={(e) => updateCropDimension(dim.id, "width", parseInt(e.target.value) || 0)}
                      placeholder="Width"
                      className="flex-1 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-900 shadow-sm transition-all focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200"
                      min="100"
                      max="4000"
                    />
                    <span className="text-slate-400">×</span>
                    <input
                      type="number"
                      value={dim.height}
                      onChange={(e) => updateCropDimension(dim.id, "height", parseInt(e.target.value) || 0)}
                      placeholder="Height"
                      className="flex-1 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-900 shadow-sm transition-all focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200"
                      min="100"
                      max="4000"
                    />
                    <button
                      onClick={() => removeCropDimension(dim.id)}
                      disabled={cropDimensions.length === 1}
                      className="rounded-xl border-2 border-red-200 bg-white p-2.5 text-red-600 shadow-sm transition-all hover:border-red-300 hover:bg-red-50 hover:shadow disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-white"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
              <button
                onClick={addCropDimension}
                className="flex w-full items-center justify-center gap-2 rounded-xl border-2 border-dashed border-amber-300 bg-white/60 px-4 py-2.5 text-sm font-semibold text-amber-700 transition-all hover:border-amber-400 hover:bg-white hover:shadow-md"
              >
                <Plus className="h-4 w-4" />
                Add Size
              </button>
            </motion.div>
          )}

          <div className="grid grid-cols-2 gap-4">
            {(Object.keys(MODEL_OPTIONS) as ModelKey[]).map((key) => {
              const option = MODEL_OPTIONS[key];
              const isActive = selectedModel === key;
              return (
                <motion.button
                  key={key}
                  onClick={() => setSelectedModel(key)}
                  className={`relative flex items-center gap-3 overflow-hidden rounded-2xl border-2 px-5 py-4 transition-all ${
                    isActive
                      ? "border-transparent text-white shadow-lg"
                      : "border-slate-200 bg-slate-50/50 text-slate-700 hover:border-slate-300 hover:bg-slate-100/70 hover:shadow-md"
                  }`}
                  whileTap={{ scale: 0.98 }}
                  whileHover={!isActive ? { scale: 1.01, y: -2 } : undefined}
                >
                    {isActive && (
                      <motion.span
                        layoutId="model-accent"
                        className={`absolute inset-0 -z-10 ${
                          key === "detail" ? "bg-blue-500" : "bg-purple-500"
                        }`}
                        initial={false}
                        transition={{ type: "spring", stiffness: 250, damping: 30 }}
                      />
                    )}
                  {key === "detail" ? (
                    <Image className={`h-5 w-5 ${isActive ? "text-white" : "text-blue-600"}`} />
                  ) : (
                    <FileText className={`h-5 w-5 ${isActive ? "text-white" : "text-purple-600"}`} />
                  )}
                  <span className="text-sm font-bold">
                    {key === "detail" ? "Detail" : "Text"}
                  </span>
                </motion.button>
              );
            })}
          </div>

          <motion.div
            ref={dropZoneRef}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative flex flex-col items-center justify-center gap-6 rounded-3xl border-2 border-dashed px-8 py-16 text-center transition-all duration-300 md:px-16 md:py-20 ${
              isDragging
                ? "border-blue-400 bg-blue-50 shadow-2xl shadow-blue-200/50"
                : "border-slate-300 bg-white hover:border-blue-300 hover:bg-blue-50/50 hover:shadow-xl"
            }`}
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.15 }}
          >
            <motion.div
              animate={{
                scale: isDragging ? 1.1 : 1,
                rotate: isDragging ? 5 : 0,
              }}
              transition={{ type: "spring", stiffness: 220, damping: 15 }}
              className={`flex h-24 w-24 items-center justify-center rounded-2xl shadow-lg ${
                isDragging
                  ? "bg-blue-500"
                  : "bg-blue-400"
              }`}
            >
              <Upload className="h-12 w-12 text-white" />
            </motion.div>
            <motion.button
              onClick={() => fileInputRef.current?.click()}
              className="rounded-xl border-2 border-blue-500 bg-blue-500 px-8 py-3 text-sm font-bold text-white shadow-lg transition-all hover:scale-105 hover:shadow-xl hover:shadow-blue-200/50"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              {isDragging ? "Drop Here" : "Upload Images"}
            </motion.button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
            />
          </motion.div>
        </section>

        <section className="flex-1">
          {files.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-slate-900">
                  Queue ({files.length})
                </h3>
              </div>

              <AnimatePresence initial={false}>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {files.map((file) => {
                    // Bereken precieze progress: gebruik file.progress direct (die is al gebaseerd op chunks)
                    let progressValue = file.status === "completed" ? 1 : (file.progress ?? 0);
                    
                    // Gebruik de nieuwe backend progress data voor preciezere chunk progress
                    let currentChunkProgress = 0;
                    let currentChunkNumber = 0;
                    if (file.tiles && file.tiles.total > 0 && file.status === "processing") {
                      // Gebruik de nieuwe backend velden voor precieze progress
                      if (file.currentTile !== undefined && file.currentTileProgress !== undefined) {
                        currentChunkNumber = file.currentTile;
                        currentChunkProgress = file.currentTileProgress;
                      } else {
                        // Fallback naar oude berekening
                        const exactChunkPosition = (file.progress ?? 0) * file.tiles.total;
                        const completedChunks = Math.floor(exactChunkPosition);
                        currentChunkNumber = completedChunks + 1;
                        currentChunkProgress = exactChunkPosition - completedChunks;
                      }
                      
                      // Gebruik de preciezere progress
                      progressValue = file.progress ?? 0;
                    }
                    
                    // Converteer naar percentage met 1 decimaal voor precisie
                    const percent = Math.min(100, Math.max(0, Math.round(progressValue * 1000) / 10));
                    const currentChunkPercent = Math.round(currentChunkProgress * 100);
                    
                    const progressLabel =
                      file.backendStatus === "completed"
                        ? "Finalizing download"
                        : file.backendMessage ??
                          (file.tiles && currentChunkNumber > 0
                            ? `Chunk ${currentChunkNumber} / ${file.tiles.total} (${currentChunkPercent}%)`
                            : file.tiles
                            ? `Chunk ${file.tiles.completed} / ${file.tiles.total}`
                            : `${percent}%`);

                    return (
                      <motion.div
                        key={file.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ type: "spring", stiffness: 200, damping: 20 }}
                        className="group flex flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-lg transition-all hover:shadow-xl"
                      >
                        {(file.preview || file.livePreview) && (
                          <div className="relative h-48 w-full overflow-hidden bg-slate-100 flex items-center justify-center">
                            {file.livePreview && file.status === "processing" ? (
                              <img
                                src={file.livePreview}
                                alt={`${file.file.name} (processing)`}
                                className="max-h-full max-w-full object-contain transition-opacity duration-300"
                              />
                            ) : file.preview ? (
                              <img
                                src={file.preview}
                                alt={file.file.name}
                                className="max-h-full max-w-full object-contain transition-transform duration-500 group-hover:scale-[1.03]"
                              />
                            ) : null}
                            {file.status === "processing" && !file.livePreview && (
                              <div className="absolute inset-0 flex items-center justify-center bg-white/80 backdrop-blur-sm">
                                <Loader2 className="h-10 w-10 animate-spin text-blue-500" />
                              </div>
                            )}
                          </div>
                        )}
                        <div className="flex flex-1 flex-col gap-5 p-6">
                          <div className="space-y-3">
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0 flex-1">
                                <p className="truncate text-base font-bold text-slate-900">
                                  {file.file.name}
                                </p>
                                <p className="mt-1 text-xs text-slate-500">
                                  {(file.file.size / 1024).toFixed(1)} KB · {MODEL_OPTIONS[file.model].title}
                                </p>
                              </div>
                              <div className="flex flex-wrap gap-2">
                                <span className="rounded-lg border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-semibold uppercase tracking-wide text-blue-700">
                                  {file.model === "detail" ? "Detail" : "Text"}
                                </span>
                                <span className="rounded-lg border border-purple-200 bg-purple-50 px-2.5 py-1 text-xs font-semibold uppercase tracking-wide text-purple-700">
                                  {file.mode === "upscale" ? "Upscale" : "Crop"}
                                </span>
                              </div>
                            </div>
                          </div>

                          <div className="space-y-4">
                            {file.status === "pending" && (
                              <span className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-600">
                                <motion.span
                                  className="inline-block h-2 w-2 rounded-full bg-slate-400"
                                  animate={{ opacity: [0.3, 1, 0.3] }}
                                  transition={{ repeat: Infinity, duration: 1.5 }}
                                />
                                Queued
                              </span>
                            )}
                            {file.status === "processing" && (
                              <span className="inline-flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700">
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                Processing
                              </span>
                            )}
                            {file.status === "completed" && (
                              <span className="inline-flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700">
                                <Check className="h-3.5 w-3.5" />
                                Downloaded
                              </span>
                            )}
                            {file.status === "error" && (
                              <div className="space-y-2">
                                <span className="inline-flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-1.5 text-xs font-semibold text-red-700">
                                  <X className="h-3.5 w-3.5" /> Error
                                </span>
                                {file.error && (
                                  <p className="text-xs text-red-600">{file.error}</p>
                                )}
                              </div>
                            )}

                            {file.status !== "pending" && file.status !== "error" && (
                              <div className="space-y-3">
                                <div className="space-y-1.5">
                                  <div className="flex items-center justify-between text-xs font-semibold text-slate-600">
                                    <span>
                                      {file.tiles 
                                        ? `Chunk ${file.tiles.completed} / ${file.tiles.total}`
                                        : progressLabel}
                                    </span>
                                    <span className="text-slate-700">{file.status === "completed" ? "100%" : `${percent.toFixed(1)}%`}</span>
                                  </div>
                                  <div className="h-4 w-full overflow-hidden rounded-full bg-slate-200 shadow-inner">
                                    <motion.div
                                      className="h-full rounded-full bg-blue-500 shadow-sm"
                                      initial={{ width: 0 }}
                                      animate={{ width: `${percent}%` }}
                                      transition={{ duration: 0.3, ease: "easeOut" }}
                                    />
                                  </div>
                                  {file.tiles && file.tiles.total > 0 && (
                                    <div className="flex gap-1">
                                      {Array.from({ length: file.tiles.total }).map((_, idx) => {
                                        const chunkIndex = idx + 1;
                                        const isCompleted = chunkIndex < file.tiles!.completed;
                                        const isCurrent = chunkIndex === file.tiles!.completed && file.status === "processing";
                                        const isNext = chunkIndex === file.tiles!.completed + 1 && file.status === "processing";
                                        
                                        // Bereken progress voor huidige chunk
                                        let chunkProgress = 0;
                                        if (isCurrent && currentChunkProgress > 0) {
                                          chunkProgress = currentChunkProgress;
                                        }
                                        
                                        return (
                                          <div key={idx} className="flex-1 relative h-1.5 rounded-full bg-slate-200 overflow-hidden">
                                            {isCompleted && (
                                              <div className="absolute inset-0 bg-blue-500 rounded-full" />
                                            )}
                                            {isCurrent && chunkProgress > 0 && (
                                              <motion.div
                                                className="absolute inset-0 bg-blue-500 rounded-full"
                                                initial={{ width: 0 }}
                                                animate={{ width: `${chunkProgress * 100}%` }}
                                                transition={{ duration: 0.2 }}
                                              />
                                            )}
                                            {isNext && (
                                              <div className="absolute inset-0 bg-blue-300 animate-pulse rounded-full" />
                                            )}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>

                          <button
                            onClick={() => removeFile(file.id)}
                            className="mt-auto flex w-full items-center justify-center gap-2 rounded-xl border-2 border-slate-200 bg-white py-2.5 text-sm font-semibold text-slate-700 transition-all hover:border-red-300 hover:bg-red-50 hover:text-red-600 hover:shadow-md"
                          >
                            <X className="h-4 w-4" />
                            Remove
                          </button>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </AnimatePresence>
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}
