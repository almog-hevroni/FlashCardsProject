"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import type { Card } from "@/lib/api/client";
import { IdeaIcon } from "@/components/icons/idea-icon";
import { RatingControls } from "@/components/exam/rating-controls";
import type { ReviewRating } from "@/lib/api/client";

type FlashcardPlayerProps = {
  card: Card | null;
  isAnswerVisible: boolean;
  canRateCurrentCard: boolean;
  selectedRating: ReviewRating | null;
  isRatingPending: boolean;
  isNextEnabled: boolean;
  statusMessage: string | null;
  onToggleAnswer: () => void;
  onShowProofs: (card: Card) => void;
  onRate: (rating: ReviewRating) => void;
  onLoadPrevious: () => void;
  onLoadNext: () => void;
  isPreviousEnabled: boolean;
};

export function FlashcardPlayer({
  card,
  isAnswerVisible,
  canRateCurrentCard,
  selectedRating,
  isRatingPending,
  isNextEnabled,
  statusMessage,
  onToggleAnswer,
  onShowProofs,
  onRate,
  onLoadPrevious,
  onLoadNext,
  isPreviousEnabled,
}: FlashcardPlayerProps) {
  if (!card) {
    return (
      <section
        className="flashcard-player flashcard-player--empty"
        aria-live="polite"
      >
        <h1 className="flashcard-player__empty-title">No cards yet</h1>
        <p className="flashcard-player__hint">
          {statusMessage ?? "No cards are available for this exam yet."}
        </p>
        <div className="flashcard-player__nav">
          <button
            className="flashcard-player__nav-button"
            type="button"
            onClick={onLoadPrevious}
            disabled={!isPreviousEnabled}
          >
            Previous
          </button>
          <button
            className="flashcard-player__nav-button flashcard-player__nav-button--primary"
            type="button"
            onClick={onLoadNext}
          >
            Next
          </button>
        </div>
      </section>
    );
  }

  const cardType =
    typeof card.info?.card_type === "string" ? card.info.card_type : null;
  const difficultyLabel =
    cardType === "diagnostic"
      ? "Getting to know you ✨"
      : `Difficulty ${card.difficulty}`;
  const [isProofsHover, setIsProofsHover] = useState(false);

  return (
    <section className="flashcard-player" aria-label="Flashcard player">
      <div className="flashcard-player__meta-row">
        <span className="flashcard-player__badge flashcard-player__badge--topic">
          Topic: {card.topic_label ?? "General topic"}
        </span>
        <span className="flashcard-player__badge flashcard-player__badge--difficulty">
          {difficultyLabel}
        </span>
      </div>

      <div className="flashcard-player__flip-shell">
        <motion.div
          className="flashcard-player__flip-card"
          role="button"
          tabIndex={0}
          onClick={(event) => {
            if (
              (event.target as HTMLElement).closest("[data-no-flip='true']")
            ) {
              return;
            }
            onToggleAnswer();
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              onToggleAnswer();
            }
          }}
          animate={{ rotateY: isAnswerVisible ? 180 : 0 }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
          aria-label={
            isAnswerVisible ? "Show question side" : "Show answer side"
          }
        >
          <article
            className="flashcard-player__face flashcard-player__face--front"
            style={{ pointerEvents: isAnswerVisible ? "none" : "auto" }}
          >
            <span className="flashcard-player__face-label">Question</span>
            <div className="flashcard-player__front-content">
              <p className="flashcard-player__content flashcard-player__content--question">
                {card.question}
              </p>
            </div>
            <p className="flashcard-player__face-hint">Tap to flip</p>
          </article>
          <article
            className="flashcard-player__face flashcard-player__face--back"
            style={{ pointerEvents: isAnswerVisible ? "auto" : "none" }}
          >
            <span className="flashcard-player__face-label">Answer</span>
            <div className="flashcard-player__back-content">
              <p className="flashcard-player__content">{card.answer}</p>
            </div>
            <div className="flashcard-player__back-footer">
              <motion.button
                className="flashcard-player__proofs-button flex items-center gap-2"
                type="button"
                data-no-flip="true"
                onHoverStart={() => setIsProofsHover(true)}
                onHoverEnd={() => setIsProofsHover(false)}
                onClick={(event) => {
                  event.stopPropagation();
                  onShowProofs(card);
                }}
              >
                <motion.span
                  className="inline-flex h-10 w-10 items-center justify-center"
                  animate={
                    isProofsHover
                      ? {
                          scale: [1, 1.08, 1],
                          filter: [
                            "drop-shadow(0 0 0 rgba(245, 197, 66, 0))",
                            "drop-shadow(0 0 8px rgba(245, 197, 66, 0.72))",
                            "drop-shadow(0 0 4px rgba(245, 197, 66, 0.4))",
                          ],
                        }
                      : {
                          scale: 1,
                          filter: "drop-shadow(0 0 0 rgba(245, 197, 66, 0))",
                        }
                  }
                  transition={{
                    duration: 1.05,
                    ease: "easeInOut",
                    repeat: isProofsHover ? Number.POSITIVE_INFINITY : 0,
                  }}
                >
                  <IdeaIcon className="h-10 w-10" />
                </motion.span>
                View proofs
              </motion.button>
              <p className="flashcard-player__face-hint flashcard-player__face-hint--back">
                Show question
              </p>
              <span
                className="flashcard-player__back-footer-spacer"
                aria-hidden="true"
              />
            </div>
          </article>
        </motion.div>
      </div>

      {canRateCurrentCard || selectedRating ? (
        <RatingControls
          canRateCurrentCard={canRateCurrentCard}
          selectedRating={selectedRating}
          isPending={isRatingPending}
          onRate={onRate}
        />
      ) : null}
      {!canRateCurrentCard && !selectedRating ? (
        <p className="flashcard-player__readonly">
          This card is read-only and cannot be rated.
        </p>
      ) : null}
      {statusMessage ? (
        <p className="flashcard-player__hint">{statusMessage}</p>
      ) : null}

      <footer className="flashcard-player__nav">
        <button
          className="flashcard-player__nav-button"
          type="button"
          onClick={onLoadPrevious}
          disabled={!isPreviousEnabled}
        >
          Previous
        </button>
        <button
          className="flashcard-player__nav-button flashcard-player__nav-button--primary"
          type="button"
          onClick={onLoadNext}
          disabled={!isNextEnabled || isRatingPending}
        >
          Next
        </button>
      </footer>
    </section>
  );
}
