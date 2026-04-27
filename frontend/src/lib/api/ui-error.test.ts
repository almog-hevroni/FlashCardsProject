import { describe, expect, it } from "vitest";
import { mapApiError } from "@/lib/api/ui-error";

describe("mapApiError", () => {
  it("maps home upload 422 to validation guidance", () => {
    const error = {
      status: 422,
      responseText: JSON.stringify({
        error: "diagnostic_bootstrap_failed",
      }),
    };
    const mapped = mapApiError(error, "home.upload.create_exam");
    expect(mapped.message).toContain("could not build your exam");
    expect(mapped.canRetry).toBe(true);
  });

  it("maps exam load 404 to non-retry message", () => {
    const error = {
      status: 404,
      responseText: JSON.stringify({
        error: "Exam not found",
      }),
    };
    const mapped = mapApiError(error, "exam.workspace.load_session");
    expect(mapped.message).toBe("We could not find this exam.");
    expect(mapped.canRetry).toBe(false);
  });

  it("maps review 403 to permission message without retry", () => {
    const error = {
      status: 403,
      responseText: JSON.stringify({
        error: "Exam does not belong to user",
      }),
    };
    const mapped = mapApiError(error, "exam.workspace.review_card");
    expect(mapped.message).toBe("You do not have permission to rate this card.");
    expect(mapped.canRetry).toBe(false);
  });

  it("maps progress network failure to connectivity guidance", () => {
    const error = new Error("Failed to fetch");
    const mapped = mapApiError(error, "exam.progress.load");
    expect(mapped.message).toContain("check your connection");
    expect(mapped.canRetry).toBe(true);
  });
});
