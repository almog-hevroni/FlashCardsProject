"use client";

import { useEffect, useState } from "react";
import { motion, useAnimationControls } from "framer-motion";
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
  isPreparingNextCard: boolean;
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
  isPreparingNextCard,
  isNextEnabled,
  statusMessage,
  onToggleAnswer,
  onShowProofs,
  onRate,
  onLoadPrevious,
  onLoadNext,
  isPreviousEnabled,
}: FlashcardPlayerProps) {
  const [isProofsHover, setIsProofsHover] = useState(false);
  const [displayedAnswerVisible, setDisplayedAnswerVisible] = useState(isAnswerVisible);
  const cardControls = useAnimationControls();

  useEffect(() => {
    let isCancelled = false;

    async function runFlip() {
      if (displayedAnswerVisible === isAnswerVisible) {
        await cardControls.start({
          rotateY: 0,
          transition: { duration: 0.12, ease: "easeOut" },
        });
        return;
      }

      const direction = isAnswerVisible ? 1 : -1;
      await cardControls.start({
        rotateY: direction * 90,
        scale: 0.985,
        transition: { duration: 0.18, ease: "easeIn" },
      });
      if (isCancelled) {
        return;
      }
      setDisplayedAnswerVisible(isAnswerVisible);
      cardControls.set({ rotateY: direction * -90 });
      await cardControls.start({
        rotateY: 0,
        scale: 1,
        transition: { duration: 0.22, ease: [0.22, 1, 0.36, 1] },
      });
    }

    void runFlip();

    return () => {
      isCancelled = true;
    };
  }, [cardControls, displayedAnswerVisible, isAnswerVisible]);

  if (!card) {
    return (
      <motion.section
        className="flashcard-player flashcard-player--empty"
        aria-live="polite"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
      >
        <h1 className="flashcard-player__empty-title">No cards on stage yet</h1>
        <p className="flashcard-player__hint">
          {statusMessage ?? "This deck is still warming up its vocabulary."}
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
            disabled={isPreparingNextCard}
          >
            {isPreparingNextCard ? "Preparing a clever card..." : "Next card"}
          </button>
        </div>
      </motion.section>
    );
  }

  const cardType =
    typeof card.info?.card_type === "string" ? card.info.card_type : null;
  const difficultyLabel =
    cardType === "diagnostic"
      ? "Calibrating your brilliance"
      : `Difficulty ${card.difficulty}`;

  return (
    <motion.section
      className="flashcard-player"
      aria-label="Flashcard player"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.38, ease: "easeOut" }}
    >
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
          aria-label={
            isAnswerVisible ? "Show question side" : "Show answer side"
          }
        >
          <motion.article
            animate={cardControls}
            className={`flashcard-player__face ${
              displayedAnswerVisible
                ? "flashcard-player__face--back"
                : "flashcard-player__face--front"
            }`}
            initial={false}
            style={{
              backfaceVisibility: "hidden",
              transformOrigin: "center center",
              transformPerspective: 1200,
            }}
          >
            {!displayedAnswerVisible ? (
              <>
                <span className="flashcard-player__face-label">Question</span>
                <div className="flashcard-player__front-content">
                  <p className="flashcard-player__content flashcard-player__content--question">
                    {card.question}
                  </p>
                </div>
                <p className="flashcard-player__face-hint">Tap to reveal the plot twist</p>
              </>
            ) : (
              <>
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
                    Show receipts
                  </motion.button>
                  <p className="flashcard-player__face-hint flashcard-player__face-hint--back">
                    Back to the question
                  </p>
                  <span
                    className="flashcard-player__back-footer-spacer"
                    aria-hidden="true"
                  />
                </div>
              </>
            )}
          </motion.article>
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
          This read-only card is a museum piece today, so ratings are closed.
        </p>
      ) : null}
      {statusMessage ? (
        <p className="flashcard-player__hint">{statusMessage}</p>
      ) : null}
      {isPreparingNextCard ? (
        <p className="flashcard-player__hint" aria-live="polite">
          Preparing a card that matches your current level. Tiny academic chef is plating.
        </p>
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
          disabled={!isNextEnabled || isRatingPending || isPreparingNextCard}
        >
          {isPreparingNextCard ? "Preparing..." : "Next card"}
        </button>
      </footer>
    </motion.section>
  );
}
