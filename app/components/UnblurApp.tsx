"use client";

import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, Loader2, Sparkles, Upload, X } from "lucide-react";

type ModelKey = "detail" | "text";

interface ModelOption {
  title: string;
  description: string;
  accent: string;
}

interface TileProgress {
  completed: number;
  total: number;
}

interface UploadedFile {
  id: string;
  file: File;
  status: "pending" | "processing" | "completed" | "error";
  error?: string;
  preview?: string;
  model: ModelKey;
  progress: number;
  tiles?: TileProgress;
  backendStatus?: "pending" | "processing" | "completed" | "error";
  backendMessage?: string;
}

interface ProgressResponse {
  status: UploadedFile["backendStatus"];
  progress?: number;
  completed?: number;
  total?: number;
  message?: string | null;
}

const MODEL_OPTIONS: Record<ModelKey, ModelOption> = {
  detail: {
    title: "Detail Enhancer",
    description: "Best for portraits, product shots, and natural photos.",
    accent: "from-blue-500/70 to-cyan-400/60",
  },
  text: {
    title: "Text & Logos",
    description: "Dialed-in for UI, scanned documents, and crisp lettering.",
    accent: "from-purple-500/70 to-pink-500/60",
  },
};

const PROGRESS_POLL_INTERVAL = 400;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function UnblurApp() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelKey>("detail");
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
          setFiles((previous) =>
            previous.map((file) => {
              if (file.id !== uploadedFile.id) {
                return file;
              }

              const next: UploadedFile = {
                ...file,
                progress: data.progress !== undefined ? Math.min(1, Math.max(0, data.progress)) : file.progress,
                tiles:
                  data.completed !== undefined && data.total !== undefined
                    ? { completed: data.completed, total: data.total }
                    : file.tiles,
                backendStatus: data.status ?? file.backendStatus,
                backendMessage: data.message ?? file.backendMessage,
              };

              if (data.status === "error") {
                next.status = "error";
                next.error = data.message ?? file.error ?? "Processing failed";
              } else if (data.status === "completed") {
                next.status = "completed";
                next.progress = 1;
              }

              return next;
            }),
          );

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
      formData.append("jobId", uploadedFile.id);

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
      link.download = uploadedFile.file.name;
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

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#05050b] text-slate-100">
      <motion.div
        aria-hidden
        className="pointer-events-none absolute -top-32 left-1/2 w-[60rem] h-[60rem] -translate-x-1/2 rounded-full bg-gradient-to-br from-purple-800/70 via-indigo-800/40 to-blue-700/30 blur-3xl"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.4, ease: "easeOut" }}
      />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col gap-10 px-6 pb-16 pt-20">
        <header className="flex flex-col gap-6 text-center md:text-left">
          <motion.div
            className="inline-flex items-center gap-2 self-center rounded-full border border-white/10 bg-white/5 px-4 py-1 text-xs uppercase tracking-[0.32em] text-slate-300 backdrop-blur md:self-start"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Sparkles className="h-3 w-3" /> AI Restoration Suite
          </motion.div>
          <motion.h1
            className="text-4xl font-semibold leading-tight md:text-5xl lg:text-6xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
          >
            Bring your blurry images back to life with pixel-perfect clarity.
          </motion.h1>
          <motion.p
            className="max-w-3xl text-base text-slate-300 md:text-lg"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
          >
            Upload entire batches, pick the neural network that fits the job, and
            download enhanced images with preserved filenames and dimensions.
          </motion.p>
        </header>

        <section className="flex flex-col gap-6 rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl md:p-8">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">Model selector</h2>
              <p className="text-sm text-slate-300/80">
                Swap between general detail restoration and a text-focused model without leaving the flow.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {(Object.keys(MODEL_OPTIONS) as ModelKey[]).map((key) => {
                const option = MODEL_OPTIONS[key];
                const isActive = selectedModel === key;
                return (
                  <motion.button
                    key={key}
                    onClick={() => setSelectedModel(key)}
                    className={`relative overflow-hidden rounded-xl border px-4 py-3 text-left transition-colors ${
                      isActive
                        ? "border-transparent text-white"
                        : "border-white/10 text-slate-300 hover:border-white/30"
                    }`}
                    whileTap={{ scale: 0.97 }}
                    whileHover={!isActive ? { scale: 1.02 } : undefined}
                  >
                    {isActive && (
                      <motion.span
                        layoutId="model-accent"
                        className={`absolute inset-0 -z-10 bg-gradient-to-br ${option.accent}`}
                        initial={false}
                        transition={{ type: "spring", stiffness: 250, damping: 30 }}
                      />
                    )}
                    <span className="text-sm font-semibold uppercase tracking-wide text-white/80">
                      {option.title}
                    </span>
                    <p className="mt-1 text-xs text-white/70">{option.description}</p>
                  </motion.button>
                );
              })}
            </div>
          </div>

          <motion.div
            ref={dropZoneRef}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative flex flex-col items-center justify-center gap-4 rounded-2xl border border-dashed border-white/20 px-8 py-16 text-center transition-all duration-200 md:px-16 md:py-20 ${
              isDragging
                ? "border-white/60 bg-white/10 shadow-[0_0_60px_-20px_rgba(255,255,255,0.45)]"
                : "bg-black/30 hover:border-white/40 hover:bg-black/25"
            }`}
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.15 }}
          >
            <motion.div
              animate={{
                scale: isDragging ? 1.05 : 1,
                rotate: isDragging ? -2 : 0,
              }}
              transition={{ type: "spring", stiffness: 220, damping: 15 }}
              className="flex h-24 w-24 items-center justify-center rounded-full bg-white/10"
            >
              <Upload className="h-12 w-12 text-slate-100" />
            </motion.div>
            <div>
              <h3 className="text-2xl font-semibold">
                {isDragging ? "Release to upload" : "Drop your images or click to browse"}
              </h3>
              <p className="mt-2 text-sm text-slate-300/80">
                Supports PNG, JPG, and WebP. We keep your filenames intact on download.
              </p>
            </div>
            <motion.button
              onClick={() => fileInputRef.current?.click()}
              className="rounded-full border border-white/30 bg-white/10 px-6 py-2 text-sm font-medium uppercase tracking-wide text-white/80 transition hover:border-white/70 hover:bg-white/20"
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              Select images
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
            <div className="space-y-6">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h3 className="text-base font-semibold text-white">
                    Processing queue · {files.length} file{files.length === 1 ? "" : "s"}
                  </h3>
                  <p className="text-sm text-slate-300/70">
                    Downloads start automatically the moment a file finishes.
                  </p>
                </div>
                <button
                  onClick={clearAll}
                  className="self-start rounded-full border border-white/10 px-4 py-1 text-xs uppercase tracking-wide text-slate-300 transition hover:border-red-400/50 hover:text-red-300"
                >
                  Clear queue
                </button>
              </div>

              <AnimatePresence initial={false}>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {files.map((file) => {
                    const progressValue = file.status === "completed" ? 1 : (file.progress ?? 0);
                    const percent = Math.min(100, Math.max(0, Math.round(progressValue * 100)));
                    const progressLabel =
                      file.backendStatus === "completed"
                        ? "Finalizing download"
                        : file.backendMessage ??
                          (file.tiles ? `${file.tiles.completed}/${file.tiles.total} tiles` : `${percent}%`);

                    return (
                      <motion.div
                        key={file.id}
                        layout
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ type: "spring", stiffness: 200, damping: 20 }}
                        className="group flex flex-col overflow-hidden rounded-2xl border border-white/10 bg-white/5 backdrop-blur"
                      >
                        {file.preview && (
                          <div className="relative h-44 w-full overflow-hidden bg-black/60">
                            <img
                              src={file.preview}
                              alt={file.file.name}
                              className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.02]"
                            />
                            {file.status === "processing" && (
                              <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                                <Loader2 className="h-9 w-9 animate-spin text-cyan-300" />
                              </div>
                            )}
                          </div>
                        )}
                        <div className="flex flex-1 flex-col gap-4 p-5">
                          <div className="space-y-2">
                            <div className="flex items-start justify-between gap-2">
                              <div>
                                <p className="truncate text-base font-semibold text-white">
                                  {file.file.name}
                                </p>
                                <p className="text-xs text-slate-300/70">
                                  {(file.file.size / 1024).toFixed(1)} KB · {MODEL_OPTIONS[file.model].title}
                                </p>
                              </div>
                              <span className="rounded-full border border-white/10 px-3 py-1 text-xs uppercase tracking-wide text-slate-200/80">
                                {file.model === "detail" ? "Detail" : "Text"}
                              </span>
                            </div>
                          </div>

                          <div className="space-y-3">
                            {file.status === "pending" && (
                              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-1 text-xs text-slate-200/80">
                                <motion.span
                                  className="inline-block h-2 w-2 rounded-full bg-slate-200/80"
                                  animate={{ opacity: [0.3, 1, 0.3] }}
                                  transition={{ repeat: Infinity, duration: 1.5 }}
                                />
                                Queued
                              </span>
                            )}
                            {file.status === "processing" && (
                              <span className="inline-flex items-center gap-2 rounded-full border border-cyan-400/40 px-3 py-1 text-xs text-cyan-300">
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                Processing
                              </span>
                            )}
                            {file.status === "completed" && (
                              <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/40 px-3 py-1 text-xs text-emerald-300">
                                <Check className="h-3.5 w-3.5" />
                                Downloaded
                              </span>
                            )}
                            {file.status === "error" && (
                              <div className="space-y-1">
                                <span className="inline-flex items-center gap-2 rounded-full border border-red-400/40 px-3 py-1 text-xs text-red-300">
                                  <X className="h-3.5 w-3.5" /> Error
                                </span>
                                {file.error && (
                                  <p className="text-xs text-red-200/80">{file.error}</p>
                                )}
                              </div>
                            )}

                            {file.status !== "pending" && file.status !== "error" && (
                              <div className="space-y-2">
                                <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                                  <div
                                    className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 transition-all duration-200"
                                    style={{ width: `${percent}%` }}
                                  />
                                </div>
                                <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-slate-300/70">
                                  <span>{progressLabel}</span>
                                  <span>{file.status === "completed" ? "100%" : `${percent}%`}</span>
                                </div>
                              </div>
                            )}
                          </div>

                          <button
                            onClick={() => removeFile(file.id)}
                            className="mt-auto w-full rounded-xl border border-white/10 bg-black/20 py-2 text-sm font-medium text-slate-200 transition hover:border-red-400/50 hover:text-red-200"
                          >
                            Remove from queue
                          </button>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </AnimatePresence>
            </div>
          ) : (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-10 text-center text-slate-300/80 backdrop-blur">
              <p className="text-sm md:text-base">
                Your queue is empty. Drag in a batch of assets or tap Select images to begin.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
