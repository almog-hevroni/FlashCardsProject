"use client";

import { motion } from "framer-motion";
import { PanelRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getExamProgress } from "@/lib/api/client";
import { mapApiError } from "@/lib/api/ui-error";
import { InlineError } from "@/components/common/inline-error";

type ProgressPanelProps = {
  examId: string;
  userId: string;
  isOpen: boolean;
  onToggle?: () => void;
};

function toPercent(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "0%";
  }
  return `${Math.round(value * 100)}%`;
}

function toNumericPercent(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

export function ProgressPanel({ examId, userId, isOpen, onToggle }: ProgressPanelProps) {
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["exam-progress", examId, userId],
    queryFn: () => getExamProgress(examId, userId),
    retry: 1,
  });
  const progressError = isError ? mapApiError(error, "exam.progress.load") : null;

  return (
    <aside
      className={`progress-panel${isOpen ? " progress-panel--open" : ""}`}
      aria-label="Study progress"
    >
      {!isOpen ? (
        <button
          className="progress-panel__toggle"
          type="button"
          onClick={onToggle}
          aria-label="Show progress sidebar"
          title="Show progress sidebar"
        >
          <PanelRight className="progress-panel__toggle-icon" size={18} aria-hidden="true" />
        </button>
      ) : null}

      {isOpen ? (
        <div className="progress-panel__content">
          <div className="progress-panel__header">
            <h2 className="progress-panel__title">Progress</h2>
            <button
              className="progress-panel__toggle"
              type="button"
              onClick={onToggle}
              aria-label="Hide progress sidebar"
              title="Hide progress sidebar"
            >
              <PanelRight className="progress-panel__toggle-icon progress-panel__toggle-icon--open" size={18} aria-hidden="true" />
            </button>
          </div>

          {isLoading ? <p className="progress-panel__hint">Loading progress...</p> : null}

          {progressError ? (
            <div>
              <InlineError
                message={progressError.message}
                onRetry={progressError.canRetry ? () => void refetch() : undefined}
                messageClassName="progress-panel__error"
                retryClassName="progress-panel__retry"
              />
            </div>
          ) : null}

          {!isLoading && !isError ? (
            <div className="progress-panel__summary progress-panel__summary--magic">
              <div>
                <span className="progress-panel__label">Overall mastery</span>
                <span className="progress-panel__value">{toPercent(data?.overall_proficiency ?? null)}</span>
              </div>
              <span className="progress-panel__sparkles" aria-hidden="true">
                ✦
              </span>
            </div>
          ) : null}

          {!isLoading && !isError && (data?.topics.length ?? 0) === 0 ? (
            <p className="progress-panel__hint">No topic progress yet. Rate cards to build progress.</p>
          ) : null}

          {!isLoading && !isError && (data?.topics.length ?? 0) > 0 ? (
            <ul className="progress-panel__list">
              {data?.topics.map((topic) => (
                <li key={topic.topic_id} className="progress-panel__item">
                  <div className="progress-panel__item-head">
                    <p className="progress-panel__topic">{topic.topic_label}</p>
                    <p className="progress-panel__meta">{topic.n_reviews ?? 0} reviews</p>
                  </div>
                  <div className="progress-panel__magic-track" aria-hidden="true">
                    <motion.span
                      className="progress-panel__magic-fill"
                      initial={{ width: "0%" }}
                      animate={{ width: `${toNumericPercent(topic.proficiency)}%` }}
                      transition={{ duration: 0.6, ease: "easeOut" }}
                    />
                    <motion.span
                      className="progress-panel__magic-spark"
                      initial={{ x: 0 }}
                      animate={{ x: `${toNumericPercent(topic.proficiency)}%` }}
                      transition={{
                        duration: 2,
                        repeat: Number.POSITIVE_INFINITY,
                        repeatType: "reverse",
                        ease: "easeInOut",
                      }}
                    >
                      ✦
                    </motion.span>
                  </div>
                  <span className="progress-panel__score">{toPercent(topic.proficiency)}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </aside>
  );
}
