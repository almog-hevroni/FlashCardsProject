"use client";

import { useRouter } from "next/navigation";
import {
  useId,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
  type RefObject,
} from "react";
import { motion } from "framer-motion";
import { useMutation } from "@tanstack/react-query";
import { createExamFromUpload } from "@/lib/api/client";
import { mapHomeApiError } from "@/lib/api/ui-error";
import { MagicSparklesBackground } from "@/components/home/magic-sparkles-background";
import { InlineError } from "@/components/common/inline-error";
import { MagicUploadProgress } from "@/components/home/magic-upload-progress";
import { useGuestSession } from "@/lib/session/guest-session";

const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"];
const sectionMotion = {
  hidden: { opacity: 0, y: 28 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: "easeOut" as const },
  },
};

function buildFileKey(file: File) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function extensionTone(fileName: string) {
  const lowered = fileName.toLowerCase();
  if (lowered.endsWith(".pdf")) {
    return "pdf";
  }
  if (lowered.endsWith(".docx")) {
    return "docx";
  }
  if (lowered.endsWith(".txt")) {
    return "txt";
  }
  return "default";
}

type SelectedUpload = {
  id: string;
  file: File;
};

type UploadExamFormProps = {
  id?: string;
  sectionRef?: RefObject<HTMLElement | null>;
  onFocusWithin?: () => void;
};

export function UploadExamForm({
  id,
  sectionRef,
  onFocusWithin,
}: UploadExamFormProps) {
  const router = useRouter();
  const titleInputId = useId();
  const filesInputRef = useRef<HTMLInputElement | null>(null);
  const uploadIdRef = useRef(0);
  const [isDragActive, setIsDragActive] = useState(false);
  const [selectedUploads, setSelectedUploads] = useState<SelectedUpload[]>([]);
  const [title, setTitle] = useState("");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const uploadErrorTimeoutRef = useRef<number | null>(null);
  const { userId } = useGuestSession();

  const uploadMutation = useMutation({
    mutationFn: createExamFromUpload,
    onSuccess: (result) => {
      router.push(`/exams/${result.exam_id}`);
    },
  });

  const selectedSummary = useMemo(() => {
    if (selectedUploads.length === 0) {
      return "No files yet. The stage is politely waiting.";
    }

    if (selectedUploads.length === 1) {
      return `Ready to study: ${selectedUploads[0].file.name}`;
    }

    return `${selectedUploads.length} files queued for a very productive glow-up`;
  }, [selectedUploads]);
  const titleValue = title.trim();
  const hasFiles = selectedUploads.length > 0;
  const hasTitle = titleValue.length > 0;

  useEffect(() => {
    return () => {
      if (uploadErrorTimeoutRef.current !== null) {
        window.clearTimeout(uploadErrorTimeoutRef.current);
      }
    };
  }, []);

  function clearUploadError() {
    if (uploadErrorTimeoutRef.current !== null) {
      window.clearTimeout(uploadErrorTimeoutRef.current);
      uploadErrorTimeoutRef.current = null;
    }
    setUploadError(null);
  }

  function showUploadError(message: string) {
    if (uploadErrorTimeoutRef.current !== null) {
      window.clearTimeout(uploadErrorTimeoutRef.current);
    }
    setUploadError(message);
    uploadErrorTimeoutRef.current = window.setTimeout(() => {
      setUploadError(null);
      uploadErrorTimeoutRef.current = null;
    }, 4_500);
  }

  function isSupportedUpload(file: File) {
    const loweredName = file.name.toLowerCase();
    return SUPPORTED_EXTENSIONS.some((extension) => loweredName.endsWith(extension));
  }

  function updateFiles(files: FileList | null) {
    if (!files || files.length === 0) {
      return;
    }

    const droppedFiles = Array.from(files);
    const validFiles = droppedFiles.filter(isSupportedUpload);
    const rejectedCount = droppedFiles.length - validFiles.length;

    const nextUploads = validFiles.map((file) => {
      uploadIdRef.current += 1;
      return {
        id: `${buildFileKey(file)}:${uploadIdRef.current}`,
        file,
      };
    });

    if (nextUploads.length > 0) {
      setSelectedUploads((previous) => [...previous, ...nextUploads]);
      uploadMutation.reset();
      if (rejectedCount === 0) {
        clearUploadError();
      }
    }

    if (rejectedCount > 0) {
      showUploadError("Some files were ignored. Only PDF, DOCX, and TXT files are supported.");
    }
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    setIsDragActive(false);
    updateFiles(event.dataTransfer.files);
  }

  function onDragOver(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    if (!isDragActive) {
      setIsDragActive(true);
    }
  }

  function onDragLeave(event: DragEvent<HTMLElement>) {
    if (event.currentTarget.contains(event.relatedTarget as Node)) {
      return;
    }
    setIsDragActive(false);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (selectedUploads.length === 0 || titleValue.length === 0) {
      return;
    }

    try {
      await uploadMutation.mutateAsync({
        userId,
        title: titleValue,
        files: selectedUploads.map((upload) => upload.file),
        mode: "mastery",
      });
    } catch {
      // Error is already reflected in mutation state for UI messaging.
    }
  }

  return (
    <section
      id={id}
      ref={sectionRef}
      className="home-upload-flow home-upload-flow--single"
      tabIndex={-1}
      onFocus={onFocusWithin}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      aria-label="Create exam from files"
    >
      <MagicSparklesBackground />

      <motion.form
        className="home-upload-flow__form home-upload-flow__form--single"
        onSubmit={onSubmit}
        initial="hidden"
        animate="visible"
      >
        <motion.div
          variants={sectionMotion}
          className="home-upload-composer"
          aria-label="Create your study deck"
        >
          <header className="home-upload-composer__hero">
            <h1 className="home-upload-card__title">FlashCards</h1>
            <p className="home-upload-card__subtitle">
              Upload your material and get a clean, sourced study deck in one
              simple move.
            </p>
          </header>

          <section className="home-upload-card__panel home-upload-card__panel--animated home-upload-composer__panel">
            <div className="home-upload-composer__title-row">
              <label className="home-upload-card__label" htmlFor={titleInputId}>
                Deck name
              </label>
              <input
                className="home-upload-card__input"
                id={titleInputId}
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="What are we studying today?"
              />
            </div>

            <div
              className={`home-upload-card__dropzone home-upload-card__dropzone--composer${
                isDragActive ? " home-upload-card__dropzone--active" : ""
              }`}
              role="button"
              tabIndex={0}
              onClick={() => filesInputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  filesInputRef.current?.click();
                }
              }}
              aria-label="Drop files here or click to choose files"
            >
              <p className="home-upload-card__dropzone-title">
                {isDragActive
                  ? "Drop them here"
                  : "Add your study files"}
              </p>
              <p className="home-upload-card__dropzone-hint">
                Drag in PDFs, DOCX, or TXT files, or choose them from your
                device.
              </p>
              {selectedUploads.length === 0 ? (
                <div
                  className="home-upload-card__doc-preview home-upload-card__doc-preview--compact"
                  aria-label="Supported file examples"
                >
                  <article className="home-upload-card__doc-card home-upload-card__doc-card--pdf">
                    <span className="home-upload-card__doc-badge">PDF</span>
                    <span className="home-upload-card__doc-icon" aria-hidden="true">📄</span>
                    <span className="home-upload-card__doc-label">Slides</span>
                  </article>
                  <article className="home-upload-card__doc-card home-upload-card__doc-card--docx">
                    <span className="home-upload-card__doc-badge">DOCX</span>
                    <span className="home-upload-card__doc-icon" aria-hidden="true">📝</span>
                    <span className="home-upload-card__doc-label">Notes</span>
                  </article>
                  <article className="home-upload-card__doc-card home-upload-card__doc-card--txt">
                    <span className="home-upload-card__doc-badge">TXT</span>
                    <span className="home-upload-card__doc-icon" aria-hidden="true">✦</span>
                    <span className="home-upload-card__doc-label">Summaries</span>
                  </article>
                </div>
              ) : (
                <ul
                  className="home-upload-card__file-list home-upload-card__file-list--composer"
                  aria-label="Selected files"
                >
                  {selectedUploads.map((upload) => (
                    <li key={upload.id} className="home-upload-card__file-item">
                      <span
                        className={`home-upload-card__file-icon home-upload-card__file-icon--${extensionTone(upload.file.name)}`}
                      >
                        {upload.file.name.split(".").pop()?.toUpperCase() ??
                          "FILE"}
                      </span>
                      <div className="home-upload-card__file-meta">
                        <span className="home-upload-card__file-name">
                          {upload.file.name}
                        </span>
                        <span className="home-upload-card__file-kind">
                          Study material
                        </span>
                      </div>
                      <button
                        className="home-upload-card__file-remove"
                        type="button"
                        aria-label={`Remove ${upload.file.name}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelectedUploads((previous) =>
                            previous.filter((item) => item.id !== upload.id),
                          );
                        }}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <button
                className="home-upload-card__pick"
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  filesInputRef.current?.click();
                }}
                disabled={uploadMutation.isPending}
              >
                Choose files
              </button>
            </div>

            <div className="home-upload-card__selected-wrap" aria-live="polite">
              <p className="home-upload-card__selected">{selectedSummary}</p>
              {selectedUploads.length > 0 ? (
                <button
                  className="home-upload-card__clear"
                  type="button"
                  onClick={() => setSelectedUploads([])}
                >
                  Clear the guest list
                </button>
              ) : null}
            </div>

            {uploadError ? (
              <p className="home-upload-card__error" role="alert">
                {uploadError}
              </p>
            ) : null}

              {uploadMutation.isError ? (
                <InlineError
                  message={mapHomeApiError(uploadMutation.error, "home.upload.create_exam").message}
                  messageClassName="home-upload-card__error"
                />
              ) : null}

              {uploadMutation.isPending ? <MagicUploadProgress /> : null}

              {!uploadMutation.isPending ? (
                <div className="home-upload-card__actions home-upload-card__actions--composer">
                  <button
                    className={`home-upload-card__action home-upload-card__action--primary home-upload-card__magic-button home-upload-card__magic-button--large${
                      hasFiles && hasTitle ? " is-ready" : ""
                    }`}
                    type="submit"
                    disabled={!hasFiles || !hasTitle}
                  >
                    <span className="home-upload-card__magic-button-text">Create my cards</span>
                  </button>
                </div>
              ) : null}
          </section>

          <input
            ref={filesInputRef}
            type="file"
            hidden
            multiple
            onChange={(event) => {
              updateFiles(event.target.files);
              event.currentTarget.value = "";
            }}
            accept={SUPPORTED_EXTENSIONS.join(",")}
          />
        </motion.div>
      </motion.form>
    </section>
  );
}
