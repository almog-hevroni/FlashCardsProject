import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { AnchorHTMLAttributes, ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ExamHistorySidebar } from "@/components/home/exam-history-sidebar";

const pushMock = vi.fn();
const listRecentExamsMock = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
  }),
}));

vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    ...props
  }: {
    href: string;
    children: ReactNode;
  } & AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

vi.mock("@/lib/api/client", () => ({
  listRecentExams: (...args: unknown[]) => listRecentExamsMock(...args),
}));

vi.mock("@/lib/session/guest-session", () => ({
  useGuestSession: () => ({ userId: "guest", mode: "guest" }),
}));

function renderSidebar() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <ExamHistorySidebar onNewExam={vi.fn()} />
    </QueryClientProvider>,
  );
}

describe("ExamHistorySidebar", () => {
  beforeEach(() => {
    listRecentExamsMock.mockReset();
    pushMock.mockReset();
  });

  it("renders empty state when there are no exams", async () => {
    listRecentExamsMock.mockResolvedValue([]);
    renderSidebar();

    await waitFor(() => {
      expect(screen.getByText("No recent exams yet.")).toBeInTheDocument();
    });
  });

  it("navigates to selected exam from recent list", async () => {
    listRecentExamsMock.mockResolvedValue([
      {
        exam_id: "exam-9",
        title: "Physics chapter 1",
        updated_at: new Date().toISOString(),
        created_at: new Date().toISOString(),
        mode: "mastery",
        info: {},
      },
    ]);
    renderSidebar();

    const examButton = await screen.findByRole("button", { name: /Physics chapter 1/i });
    fireEvent.click(examButton);

    expect(pushMock).toHaveBeenCalledWith("/exams/exam-9");
  });

  it("shows a friendly error message when recent exams fail", async () => {
    listRecentExamsMock.mockRejectedValue(
      new Error('API request failed (422): {"error":"diagnostic_bootstrap_failed"}'),
    );
    renderSidebar();

    await waitFor(
      () => {
        expect(
          screen.getByText("Something went wrong while loading your exams. Please try again."),
        ).toBeInTheDocument();
      },
      { timeout: 3_000 },
    );
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(screen.queryByText(/API request failed/i)).not.toBeInTheDocument();
  });
});
