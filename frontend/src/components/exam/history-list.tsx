"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  Brain,
  CheckCircle2,
  ChevronDown,
  HelpCircle,
  RotateCcw,
  ScrollText,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { InlineError } from "@/components/common/inline-error";
import { IdeaIcon } from "@/components/icons/idea-icon";
import {
  getExamById,
  getPresentedHistory,
  type Card,
  type ReviewRating,
} from "@/lib/api/client";
import { mapApiError } from "@/lib/api/ui-error";
import { useGuestSession } from "@/lib/session/guest-session";

type HistoryListProps = {
  examId: string;
};

function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "Unknown";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function presentedAtFromCard(card: Card): string {
  const presentedAt = card.info?.presented_at;
  return typeof presentedAt === "string" ? presentedAt : card.created_at;
}

function ratingFromCard(card: Card): ReviewRating | null {
  const infoRating =
    typeof card.info?.rating === "string"
      ? card.info.rating
      : typeof card.info?.last_rating === "string"
        ? card.info.last_rating
        : typeof card.info?.review_rating === "string"
          ? card.info.review_rating
          : null;
  if (
    infoRating === "i_knew_it" ||
    infoRating === "almost_knew" ||
    infoRating === "learned_now" ||
    infoRating === "dont_understand"
  ) {
    return infoRating;
  }
  return null;
}

const ratingUiMap: Record<
  ReviewRating,
  {
    label: string;
    icon: typeof CheckCircle2;
    iconClassName: string;
    badgeClassName: string;
  }
> = {
  i_knew_it: {
    label: "Nailed it",
    icon: CheckCircle2,
    iconClassName: "text-emerald-500",
    badgeClassName: "border border-emerald-200 bg-emerald-50 text-emerald-700",
  },
  almost_knew: {
    label: "Nearly there",
    icon: HelpCircle,
    iconClassName: "text-amber-500",
    badgeClassName: "border border-amber-200 bg-amber-50 text-amber-700",
  },
  learned_now: {
    label: "Just learned it",
    icon: Brain,
    iconClassName: "text-sky-500",
    badgeClassName: "border border-sky-200 bg-sky-50 text-sky-700",
  },
  dont_understand: {
    label: "Still mysterious",
    icon: AlertCircle,
    iconClassName: "text-rose-500",
    badgeClassName: "border border-rose-200 bg-rose-50 text-rose-700",
  },
};

const sidebarListMotion = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.05,
    },
  },
};

const sidebarItemMotion = {
  hidden: { opacity: 0, x: -20 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.24, ease: "easeOut" as const },
  },
};

export function HistoryList({ examId }: HistoryListProps) {
  const { userId } = useGuestSession();
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null);
  const [isAnswerVisible, setIsAnswerVisible] = useState(false);
  const [activeProofIndex, setActiveProofIndex] = useState<number | null>(0);

  const examQuery = useQuery({
    queryKey: ["exam-by-id", examId, userId],
    queryFn: () => getExamById(examId, userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const historyQuery = useQuery({
    queryKey: ["presented-history", examId, userId],
    queryFn: () => getPresentedHistory(examId, userId),
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const cards = useMemo(() => historyQuery.data ?? [], [historyQuery.data]);
  const resolvedSelectedCardId = useMemo(() => {
    if (
      selectedCardId &&
      cards.some((card) => card.card_id === selectedCardId)
    ) {
      return selectedCardId;
    }
    return cards[0]?.card_id ?? null;
  }, [cards, selectedCardId]);

  const selectedCard = useMemo(
    () => cards.find((card) => card.card_id === resolvedSelectedCardId) ?? null,
    [cards, resolvedSelectedCardId],
  );
  const selectedRating = selectedCard ? ratingFromCard(selectedCard) : null;
  const selectedRatingUi = selectedRating ? ratingUiMap[selectedRating] : null;
  const difficultyLabel =
    selectedCard &&
    typeof selectedCard.info?.card_type === "string" &&
    selectedCard.info.card_type === "diagnostic"
      ? "Calibrating your brilliance"
      : selectedCard
        ? `Difficulty ${selectedCard.difficulty}`
        : "Difficulty: mysterious";

  const examError = examQuery.isError
    ? mapApiError(examQuery.error, "exam.workspace.load_exam")
    : null;
  const historyError = historyQuery.isError
    ? mapApiError(historyQuery.error, "exam.workspace.load_session")
    : null;

  return (
    <section
      className="h-[calc(100vh-4rem)] overflow-hidden bg-gradient-to-br from-orange-50 via-sky-50 to-rose-50 p-3 md:p-4"
      aria-label="Exam history"
    >
      <div className="grid h-full grid-cols-1 gap-3 lg:grid-cols-[22rem_minmax(0,1fr)]">
        <aside className="flex h-full min-h-0 flex-col overflow-hidden rounded-3xl border border-blue-100/80 bg-white/80 p-3 shadow-xl shadow-blue-900/10 backdrop-blur-xl">
          <div className="mb-3 flex items-start justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-slate-900">
                {examQuery.data?.title ??
                  (examQuery.isLoading ? "Loading your trail..." : "Study trail")}
              </p>
              <p className="text-xs text-slate-500">
                Revisit the cards you have already met
              </p>
            </div>
            <Link
              className="inline-flex shrink-0 items-center rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
              href={`/exams/${examId}`}
            >
              Back to deck
            </Link>
          </div>

          {examError ? (
            <InlineError
              message={examError.message}
              onRetry={
                examError.canRetry ? () => void examQuery.refetch() : undefined
              }
            />
          ) : null}
          {historyError ? (
            <InlineError
              message={historyError.message}
              onRetry={
                historyError.canRetry
                  ? () => void historyQuery.refetch()
                  : undefined
              }
            />
          ) : null}

          {!historyQuery.isError && historyQuery.isLoading ? (
            <p className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
              Gathering the cards you have seen...
            </p>
          ) : null}
          {!historyQuery.isLoading &&
          !historyQuery.isError &&
          cards.length === 0 ? (
            <p className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
              No cards have walked across the stage yet.
            </p>
          ) : null}

          {!historyQuery.isLoading &&
          !historyQuery.isError &&
          cards.length > 0 ? (
            <motion.ol
              className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1"
              aria-label="Presented cards list"
              initial="hidden"
              animate="visible"
              variants={sidebarListMotion}
            >
              {cards.map((card, index) => {
                const cardRating = ratingFromCard(card);
                const ratingUi = cardRating ? ratingUiMap[cardRating] : null;
                const RatingIcon = ratingUi?.icon ?? ScrollText;
                const isActive = resolvedSelectedCardId === card.card_id;
                return (
                  <motion.li
                    key={`${card.card_id}-${index}`}
                    variants={sidebarItemMotion}
                  >
                    <button
                      type="button"
                      className={`group w-full rounded-2xl border px-3 py-3 text-left transition ${
                        isActive
                          ? "border-indigo-300 bg-indigo-50 shadow-md shadow-indigo-900/10"
                          : "border-blue-100 bg-white/85 shadow-sm hover:border-indigo-200 hover:bg-sky-50"
                      }`}
                      onClick={() => {
                        setSelectedCardId(card.card_id);
                        setIsAnswerVisible(false);
                      }}
                      aria-current={isActive ? "true" : undefined}
                    >
                      <div className="flex items-start gap-2.5">
                        <span
                          className={`mt-0.5 ${ratingUi?.iconClassName ?? "text-slate-400"}`}
                        >
                          <RatingIcon size={18} />
                        </span>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                            #{cards.length - index} •{" "}
                            {card.topic_label ?? "General topic"}
                          </p>
                          <p className="mt-1 line-clamp-2 text-sm font-semibold text-slate-800">
                            {card.question}
                          </p>
                          <div className="mt-2 flex items-center justify-between gap-2">
                            <p className="truncate text-xs text-slate-500">
                              {formatDateTime(presentedAtFromCard(card))}
                            </p>
                            <span
                              className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-medium ${
                                ratingUi?.badgeClassName ??
                                "border border-slate-300 bg-slate-100 text-slate-600"
                              }`}
                            >
                              {ratingUi?.label ?? "Not rated"}
                            </span>
                          </div>
                        </div>
                      </div>
                    </button>
                  </motion.li>
                );
              })}
            </motion.ol>
          ) : null}
        </aside>

        <main
          className="relative h-full min-h-0 overflow-hidden rounded-3xl border border-blue-100/80 p-4 shadow-xl shadow-blue-900/10 md:p-6"
          style={{
            background:
              "radial-gradient(circle at 92% 7%, rgba(101, 228, 178, 0.24), transparent 38%), radial-gradient(circle at 12% 94%, rgba(255, 216, 77, 0.24), transparent 34%), linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(239,246,255,0.86) 100%)",
          }}
        >
          <div
            className="flex h-full min-h-0 items-center justify-center"
            aria-live="polite"
          >
            {!selectedCard ? (
              <div className="rounded-2xl border border-slate-300 bg-white/90 px-6 py-8 text-center text-slate-700 shadow-sm">
                <p className="text-lg font-semibold text-slate-800">
                  Nothing to review yet
                </p>
                <p className="mt-2 text-sm text-slate-500">
                  Once history arrives, pick a card and stroll through your past brilliance.
                </p>
              </div>
            ) : (
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={selectedCard.card_id}
                  className="h-full w-full max-w-4xl"
                  initial={{ opacity: 0, y: 26 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -26 }}
                  transition={{ duration: 0.28, ease: "easeInOut" }}
                >
                  <div
                    className="mx-auto h-full max-h-[700px] w-full"
                    style={{ perspective: 1200 }}
                  >
                    <motion.div
                      className="relative h-full w-full cursor-pointer"
                      role="button"
                      tabIndex={0}
                      onClick={() =>
                        setIsAnswerVisible((previous) => !previous)
                      }
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          setIsAnswerVisible((previous) => !previous);
                        }
                      }}
                      animate={{ rotateY: isAnswerVisible ? 180 : 0 }}
                      transition={{ duration: 0.52, ease: "easeInOut" }}
                      style={{ transformStyle: "preserve-3d" }}
                      aria-label={
                        isAnswerVisible
                          ? "Show question side"
                          : "Show answer side"
                      }
                    >
                      <article
                        className="absolute inset-0 flex h-full flex-col rounded-3xl border border-slate-300/75 p-4 md:p-6"
                        style={{
                          background:
                            "radial-gradient(circle at 88% 8%, rgba(255, 216, 77, 0.28), transparent 34%), radial-gradient(circle at 10% 86%, rgba(56, 189, 248, 0.18), transparent 34%), linear-gradient(180deg, #ffffff 0%, #f4f9ff 100%)",
                          backfaceVisibility: "hidden",
                        }}
                      >
                        <div className="mb-4 flex items-center justify-between gap-2">
                          <span className="rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-semibold tracking-wide text-sky-800">
                            QUESTION
                          </span>
                          <span className="rounded-full border border-violet-200 bg-violet-50 px-3 py-1 text-sm font-medium text-violet-700">
                            {difficultyLabel}
                          </span>
                          <span
                            className={`rounded-full px-3 py-1 text-sm font-medium ${
                              selectedRatingUi?.badgeClassName ??
                              "border border-slate-300 bg-slate-100 text-slate-600"
                            }`}
                          >
                            {selectedRatingUi?.label ?? "Not rated"}
                          </span>
                        </div>

                        <div className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-1 text-center">
                          <p className="max-w-3xl whitespace-pre-wrap text-xl font-semibold leading-relaxed text-slate-800 md:text-3xl">
                            {selectedCard.question}
                          </p>
                        </div>

                        <p className="mt-3 text-center text-xs font-medium uppercase tracking-wide text-slate-500">
                          Tap for the satisfying reveal
                        </p>
                      </article>

                      <article
                        className="absolute inset-0 flex h-full flex-col rounded-3xl border border-slate-300/75 p-4 md:p-6"
                        style={{
                          background:
                            "radial-gradient(circle at 86% 9%, rgba(101, 228, 178, 0.26), transparent 34%), radial-gradient(circle at 10% 88%, rgba(139, 92, 246, 0.16), transparent 34%), linear-gradient(180deg, #ffffff 0%, #eef6ff 100%)",
                          backfaceVisibility: "hidden",
                          transform: "rotateY(180deg)",
                        }}
                      >
                        <div className="mb-4 flex items-center justify-between gap-2">
                          <span className="rounded-full border border-teal-200 bg-teal-50 px-2.5 py-1 text-[11px] font-semibold tracking-wide text-teal-800">
                              ANSWER &amp; RECEIPTS
                          </span>
                          <span className="rounded-full border border-violet-200 bg-violet-50 px-3 py-1 text-sm font-medium text-violet-700">
                            {difficultyLabel}
                          </span>
                          <span
                            className={`rounded-full px-3 py-1 text-sm font-medium ${
                              selectedRatingUi?.badgeClassName ??
                              "border border-slate-300 bg-slate-100 text-slate-600"
                            }`}
                          >
                            {selectedRatingUi?.label ?? "Not rated"}
                          </span>
                        </div>

                        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-1">
                          <p className="whitespace-pre-wrap text-lg leading-relaxed text-slate-800 md:text-2xl">
                            {selectedCard.answer}
                          </p>

                          <div className="border-t border-slate-200 pt-4">
                            <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                              <motion.span
                                className="inline-flex h-10 w-10 items-center justify-center"
                                aria-hidden="true"
                                animate={{
                                  opacity: [0.82, 1, 0.82],
                                  filter: [
                                    "drop-shadow(0 0 0 rgba(245, 197, 66, 0))",
                                    "drop-shadow(0 0 6px rgba(245, 197, 66, 0.42))",
                                    "drop-shadow(0 0 0 rgba(245, 197, 66, 0))",
                                  ],
                                }}
                                transition={{
                                  duration: 3.6,
                                  ease: "easeInOut",
                                  repeat: Number.POSITIVE_INFINITY,
                                }}
                              >
                                <IdeaIcon className="h-10 w-10" />
                              </motion.span>
                              Supporting receipts
                            </p>
                          </div>

                          {selectedCard.proofs.length > 0 ? (
                            <ul className="space-y-2">
                              {selectedCard.proofs.map((proof, proofIndex) => (
                                <li
                                  key={`${proof.doc_id}-${proof.start}-${proof.end}-${proofIndex}`}
                                  className="overflow-hidden rounded-xl border border-slate-200 bg-white/80"
                                >
                                  <button
                                    type="button"
                                    className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      setActiveProofIndex((previous) =>
                                        previous === proofIndex
                                          ? null
                                          : proofIndex,
                                      );
                                    }}
                                  >
                                    <span className="text-sm font-semibold text-slate-700">
                                      Receipt {proofIndex + 1} - {proof.doc_id}
                                    </span>
                                    <ChevronDown
                                      size={16}
                                      className={`text-slate-500 transition-transform ${
                                        activeProofIndex === proofIndex
                                          ? "rotate-180"
                                          : ""
                                      }`}
                                    />
                                  </button>

                                  <AnimatePresence initial={false}>
                                    {activeProofIndex === proofIndex ? (
                                      <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: "auto", opacity: 1 }}
                                        exit={{ height: 0, opacity: 0 }}
                                        transition={{
                                          duration: 0.2,
                                          ease: "easeInOut",
                                        }}
                                        className="overflow-hidden border-t border-slate-200"
                                      >
                                        <div className="space-y-2 px-3 py-3">
                                          <p className="text-xs font-semibold text-slate-500">
                                            Source {proof.doc_id}
                                            {proof.page !== null
                                              ? ` • Page ${proof.page}`
                                              : ""}
                                            {` • Span ${proof.start}-${proof.end}`}
                                          </p>
                                          <p className="whitespace-pre-wrap text-sm text-slate-700">
                                            {proof.text}
                                          </p>
                                        </div>
                                      </motion.div>
                                    ) : null}
                                  </AnimatePresence>
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                              No receipts attached to this card.
                            </p>
                          )}
                        </div>

                        <button
                          className="mt-4 inline-flex w-fit items-center gap-2 rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            setIsAnswerVisible(false);
                          }}
                        >
                          <RotateCcw size={14} />
                          Show the question
                        </button>
                      </article>
                    </motion.div>
                  </div>
                </motion.div>
              </AnimatePresence>
            )}
          </div>
        </main>
      </div>
    </section>
  );
}
