type UiErrorContext =
  | "home.upload.create_exam"
  | "home.sidebar.list_recent_exams"
  | "exam.workspace.load_exam"
  | "exam.workspace.load_session"
  | "exam.workspace.next_card"
  | "exam.workspace.review_card"
  | "exam.progress.load";

type BackendErrorInfo = {
  code: string | null;
  message: string | null;
};

export type UiErrorMessage = {
  message: string;
  canRetry: boolean;
};

function parseJsonObject(value: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(value);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return null;
  }
  return null;
}

function extractBackendErrorInfo(responseText: string): BackendErrorInfo {
  const payload = parseJsonObject(responseText);
  if (!payload) {
    return { code: null, message: null };
  }

  const payloadError = payload.error;
  const payloadMessage = typeof payload.message === "string" ? payload.message : null;
  if (typeof payloadError === "string") {
    return {
      code: payloadError,
      message: payloadMessage,
    };
  }

  if (typeof payloadError === "object" && payloadError !== null) {
    const nestedError = payloadError as Record<string, unknown>;
    return {
      code: typeof nestedError.code === "string" ? nestedError.code : null,
      message: typeof nestedError.message === "string" ? nestedError.message : payloadMessage,
    };
  }

  return { code: null, message: payloadMessage };
}

function normalizeStatus(error: unknown): number | null {
  if (typeof error !== "object" || error === null || !("status" in error)) {
    return null;
  }
  const status = (error as { status?: unknown }).status;
  return typeof status === "number" ? status : null;
}

function normalizeResponseText(error: unknown): string {
  if (typeof error !== "object" || error === null || !("responseText" in error)) {
    return "";
  }
  const responseText = (error as { responseText?: unknown }).responseText;
  return typeof responseText === "string" ? responseText : "";
}

function isNetworkError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  return /failed to fetch|networkerror|network request failed|load failed/i.test(
    error.message,
  );
}

function mapCreateExamError(status: number | null, backendCode: string | null): UiErrorMessage {
  if (backendCode === "unsupported_document_type") {
    return {
      message: "One or more files are not supported. Please upload PDF, DOCX, or TXT files.",
      canRetry: true,
    };
  }
  if (backendCode === "diagnostic_bootstrap_failed") {
    return {
      message: "We could not build your exam from these files. Please try with clearer or more complete material.",
      canRetry: true,
    };
  }
  if (backendCode === "immutable_exam") {
    return {
      message: "This exam cannot be updated right now. Please create a new exam.",
      canRetry: true,
    };
  }

  if (status === 400 || status === 422) {
    return {
      message: "Please review your files and title, then try again.",
      canRetry: true,
    };
  }
  if (status === 401 || status === 403) {
    return {
      message: "You do not have permission to create this exam.",
      canRetry: false,
    };
  }
  if (status === 404) {
    return {
      message: "The requested resource was not found. Please refresh and try again.",
      canRetry: true,
    };
  }
  if (status === 409) {
    return {
      message: "This exam is in a locked state. Please create a new exam.",
      canRetry: true,
    };
  }
  if (status === 429) {
    return {
      message: "Too many requests right now. Please wait a moment and try again.",
      canRetry: true,
    };
  }
  if (status !== null && status >= 500) {
    return {
      message: "Our server is having trouble right now. Please try again in a moment.",
      canRetry: true,
    };
  }

  return {
    message: "Something went wrong while creating your exam. Please try again.",
    canRetry: true,
  };
}

function mapListRecentExamsError(status: number | null): UiErrorMessage {
  if (status === 401 || status === 403) {
    return {
      message: "You do not have permission to view these exams.",
      canRetry: false,
    };
  }
  if (status === 404) {
    return {
      message: "We could not find your recent exams.",
      canRetry: true,
    };
  }
  if (status === 429) {
    return {
      message: "Too many requests right now. Please wait a moment and retry.",
      canRetry: true,
    };
  }
  if (status !== null && status >= 500) {
    return {
      message: "Our server is having trouble loading your exams. Please retry shortly.",
      canRetry: true,
    };
  }

  return {
    message: "Something went wrong while loading your exams. Please try again.",
    canRetry: true,
  };
}

function mapExamLoadError(status: number | null): UiErrorMessage {
  if (status === 401 || status === 403) {
    return {
      message: "You do not have permission to access this exam.",
      canRetry: false,
    };
  }
  if (status === 404) {
    return {
      message: "We could not find this exam.",
      canRetry: false,
    };
  }
  if (status === 429) {
    return {
      message: "Too many requests right now. Please wait a moment and retry.",
      canRetry: true,
    };
  }
  if (status !== null && status >= 500) {
    return {
      message: "Our server is having trouble loading this exam. Please try again shortly.",
      canRetry: true,
    };
  }
  return {
    message: "Could not load this exam. Please try again.",
    canRetry: true,
  };
}

function mapExamReviewError(status: number | null): UiErrorMessage {
  if (status === 400 || status === 422) {
    return {
      message: "Could not save your rating. Please try again.",
      canRetry: true,
    };
  }
  if (status === 401 || status === 403) {
    return {
      message: "You do not have permission to rate this card.",
      canRetry: false,
    };
  }
  if (status === 404) {
    return {
      message: "This card is no longer available. Please load the next card.",
      canRetry: true,
    };
  }
  if (status === 429) {
    return {
      message: "Too many requests right now. Please wait a moment and rate again.",
      canRetry: true,
    };
  }
  if (status !== null && status >= 500) {
    return {
      message: "Our server is having trouble saving your rating. Please try again shortly.",
      canRetry: true,
    };
  }
  return {
    message: "Could not save your rating. Please try again.",
    canRetry: true,
  };
}

function mapExamProgressError(status: number | null): UiErrorMessage {
  if (status === 401 || status === 403) {
    return {
      message: "You do not have permission to view this progress.",
      canRetry: false,
    };
  }
  if (status === 404) {
    return {
      message: "Progress data is not available for this exam yet.",
      canRetry: true,
    };
  }
  if (status === 429) {
    return {
      message: "Too many requests right now. Please wait a moment and retry.",
      canRetry: true,
    };
  }
  if (status !== null && status >= 500) {
    return {
      message: "Our server is having trouble loading your progress. Please retry shortly.",
      canRetry: true,
    };
  }
  return {
    message: "Could not load progress. Please try again.",
    canRetry: true,
  };
}

function mapNetworkErrorByContext(context: UiErrorContext): UiErrorMessage {
  if (context === "exam.workspace.review_card") {
    return {
      message: "We could not reach the server to save your rating. Please check your connection and try again.",
      canRetry: true,
    };
  }
  if (context === "exam.progress.load") {
    return {
      message: "We could not reach the server to load your progress. Please check your connection and retry.",
      canRetry: true,
    };
  }
  if (context.startsWith("exam.workspace")) {
    return {
      message: "We could not reach the server for this exam. Please check your connection and try again.",
      canRetry: true,
    };
  }
  return {
    message: "We could not reach the server. Please check your connection and try again.",
    canRetry: true,
  };
}

export function mapApiError(error: unknown, context: UiErrorContext): UiErrorMessage {
  if (isNetworkError(error)) {
    return mapNetworkErrorByContext(context);
  }

  const status = normalizeStatus(error);
  const responseText = normalizeResponseText(error);
  const backend = extractBackendErrorInfo(responseText);

  if (context === "home.upload.create_exam") {
    return mapCreateExamError(status, backend.code);
  }
  if (context === "home.sidebar.list_recent_exams") {
    return mapListRecentExamsError(status);
  }
  if (
    context === "exam.workspace.load_exam" ||
    context === "exam.workspace.load_session" ||
    context === "exam.workspace.next_card"
  ) {
    return mapExamLoadError(status);
  }
  if (context === "exam.workspace.review_card") {
    return mapExamReviewError(status);
  }
  return mapExamProgressError(status);
}

export function mapHomeApiError(
  error: unknown,
  context: Extract<UiErrorContext, "home.upload.create_exam" | "home.sidebar.list_recent_exams">,
): UiErrorMessage {
  return mapApiError(error, context);
}
