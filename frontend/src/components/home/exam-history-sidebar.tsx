"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { InlineError } from "@/components/common/inline-error";
import { listRecentExams } from "@/lib/api/client";
import { mapHomeApiError } from "@/lib/api/ui-error";
import { useGuestSession } from "@/lib/session/guest-session";

type ExamHistorySidebarProps = {
  className?: string;
  onNewExam: () => void;
  onNavigate?: () => void;
};

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Updated recently";
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function ExamHistorySidebar({ className, onNewExam, onNavigate }: ExamHistorySidebarProps) {
  const router = useRouter();
  const accountRef = useRef<HTMLDivElement | null>(null);
  const [isAccountMenuOpen, setIsAccountMenuOpen] = useState(false);
  const { userId } = useGuestSession();

  const {
    data: exams,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["recent-exams", userId],
    queryFn: () => listRecentExams(userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    function onDocumentPointerDown(event: MouseEvent) {
      if (accountRef.current?.contains(event.target as Node)) {
        return;
      }
      setIsAccountMenuOpen(false);
    }

    if (isAccountMenuOpen) {
      document.addEventListener("mousedown", onDocumentPointerDown);
      return () => {
        document.removeEventListener("mousedown", onDocumentPointerDown);
      };
    }
  }, [isAccountMenuOpen]);

  const recentExams = useMemo(() => exams ?? [], [exams]);
  const recentExamsError = isError
    ? mapHomeApiError(error, "home.sidebar.list_recent_exams")
    : null;

  function handleExamSelect(examId: string) {
    onNavigate?.();
    router.push(`/exams/${examId}`);
  }

  return (
    <aside id="home-sidebar" className={className} aria-label="Exam history sidebar">
      <button className="home-sidebar__new-exam" type="button" onClick={onNewExam}>
        + New exam
      </button>

      <section>
        <h2 className="home-sidebar__section-label">Recent exams</h2>

        {isLoading ? <p className="home-sidebar__hint">Loading recent exams...</p> : null}

        {isError ? (
          <InlineError
            message={recentExamsError?.message ?? "Something went wrong while loading your exams. Please try again."}
            onRetry={recentExamsError?.canRetry ? () => void refetch() : undefined}
          />
        ) : null}

        {!isLoading && !isError && recentExams.length === 0 ? (
          <p className="home-sidebar__hint">No recent exams yet.</p>
        ) : null}

        {!isLoading && !isError && recentExams.length > 0 ? (
          <ul className="home-sidebar__recent-list">
            {recentExams.map((exam) => (
              <li key={exam.exam_id}>
                <button
                  className="home-sidebar__recent-item"
                  type="button"
                  onClick={() => handleExamSelect(exam.exam_id)}
                >
                  <span className="home-sidebar__recent-title">{exam.title || "Untitled exam"}</span>
                  <span className="home-sidebar__recent-meta">{formatTimestamp(exam.updated_at)}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      <div className="home-sidebar__spacer" />

      <div ref={accountRef} className="home-sidebar__account-wrap">
        {isAccountMenuOpen ? (
          <div className="home-sidebar__account-menu" role="menu" aria-label="Account options">
            <Link className="home-sidebar__account-link" href="/profile" role="menuitem" onClick={onNavigate}>
              Profile
            </Link>
            <Link className="home-sidebar__account-link" href="/settings" role="menuitem" onClick={onNavigate}>
              Settings
            </Link>
          </div>
        ) : null}

        <button
          className="home-sidebar__account-button"
          type="button"
          aria-haspopup="menu"
          aria-expanded={isAccountMenuOpen}
          onClick={() => setIsAccountMenuOpen((prev) => !prev)}
        >
          <span>Account</span>
          <span aria-hidden="true">{isAccountMenuOpen ? "^" : "v"}</span>
        </button>
      </div>
    </aside>
  );
}
