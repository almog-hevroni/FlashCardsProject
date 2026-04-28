"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { InlineError } from "@/components/common/inline-error";
import { listRecentExams } from "@/lib/api/client";
import { mapHomeApiError } from "@/lib/api/ui-error";
import { useGuestSession } from "@/lib/session/guest-session";

type ExamHistorySidebarProps = {
  className?: string;
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

export function ExamHistorySidebar({ className, onNavigate }: ExamHistorySidebarProps) {
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
    <motion.aside
      id="home-sidebar"
      className={className}
      aria-label="Exam history sidebar"
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <section className="home-sidebar__history">
        <h2 className="home-sidebar__section-label">Recent decks</h2>

        {isLoading ? <p className="home-sidebar__hint">Finding your latest brain snacks...</p> : null}

        {isError ? (
          <InlineError
            message={recentExamsError?.message ?? "Something went wrong while loading your exams. Please try again."}
            onRetry={recentExamsError?.canRetry ? () => void refetch() : undefined}
          />
        ) : null}

        {!isLoading && !isError && recentExams.length === 0 ? (
          <p className="home-sidebar__hint">No decks yet. Your comeback arc starts with one upload.</p>
        ) : null}

        {!isLoading && !isError && recentExams.length > 0 ? (
          <ul
            className="home-sidebar__recent-list"
            style={{ maxHeight: "none", overflow: "visible" }}
          >
            {recentExams.map((exam) => (
              <li key={exam.exam_id}>
                <motion.button
                  className="home-sidebar__recent-item"
                  type="button"
                  onClick={() => handleExamSelect(exam.exam_id)}
                  whileHover={{ x: 4 }}
                  whileTap={{ scale: 0.99 }}
                >
                  <span className="home-sidebar__recent-title">{exam.title || "Untitled study quest"}</span>
                  <span className="home-sidebar__recent-meta">{formatTimestamp(exam.updated_at)}</span>
                </motion.button>
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      <div ref={accountRef} className="home-sidebar__account-wrap">
        <AnimatePresence>
          {isAccountMenuOpen ? (
          <motion.div
            className="home-sidebar__account-menu"
            role="menu"
            aria-label="Account options"
            initial={{ opacity: 0, y: 8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.98 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
          >
            <Link className="home-sidebar__account-link" href="/profile" role="menuitem" onClick={onNavigate}>
              Profile nook
            </Link>
            <Link className="home-sidebar__account-link" href="/settings" role="menuitem" onClick={onNavigate}>
              Settings studio
            </Link>
          </motion.div>
          ) : null}
        </AnimatePresence>

        <button
          className="home-sidebar__account-button"
          type="button"
          aria-haspopup="menu"
          aria-expanded={isAccountMenuOpen}
          onClick={() => setIsAccountMenuOpen((prev) => !prev)}
        >
          <span>Account</span>
        </button>
      </div>
    </motion.aside>
  );
}
