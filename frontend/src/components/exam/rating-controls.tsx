"use client";

import { motion } from "framer-motion";
import type { ReviewRating } from "@/lib/api/client";

type RatingOption = {
  value: ReviewRating;
  title: string;
  subtitle: string;
  tone: "green" | "yellow" | "blue" | "red";
};

const RATING_OPTIONS: RatingOption[] = [
  {
    value: "i_knew_it",
    title: "I knew it",
    subtitle: "Felt easy",
    tone: "green",
  },
  {
    value: "almost_knew",
    title: "Almost knew",
    subtitle: "Need a quick refresh",
    tone: "yellow",
  },
  {
    value: "learned_now",
    title: "Learned now",
    subtitle: "Just understood it",
    tone: "blue",
  },
  {
    value: "dont_understand",
    title: "Don't understand",
    subtitle: "Need more support",
    tone: "red",
  },
];

type RatingControlsProps = {
  canRateCurrentCard: boolean;
  selectedRating: ReviewRating | null;
  isPending: boolean;
  onRate: (rating: ReviewRating) => void;
};

export function RatingControls({
  canRateCurrentCard,
  selectedRating,
  isPending,
  onRate,
}: RatingControlsProps) {
  function renderBubble(option: RatingOption, isSelected: boolean) {
    const isDisabled = isSelected || !canRateCurrentCard || isPending;
    return (
      <motion.button
        key={option.value}
        className={`rating-controls__bubble rating-controls__bubble--${option.tone}${
          isSelected ? " rating-controls__bubble--selected" : ""
        }`}
        type="button"
        onClick={() => onRate(option.value)}
        disabled={isDisabled}
        animate={isSelected ? { y: 0 } : { y: [0, -4, 0] }}
        transition={
          isSelected
            ? { duration: 0.2, ease: "easeOut" }
            : {
                duration: 4,
                repeat: Number.POSITIVE_INFINITY,
                ease: "easeInOut",
              }
        }
        whileHover={isSelected ? undefined : { y: -6 }}
        whileTap={isSelected ? undefined : { y: -2 }}
      >
        <span className="rating-controls__bubble-title">{option.title}</span>
        <span className="rating-controls__bubble-subtitle">
          {option.subtitle}
        </span>
      </motion.button>
    );
  }

  const selectedOption = selectedRating
    ? (RATING_OPTIONS.find((option) => option.value === selectedRating) ?? null)
    : null;

  return (
    <div
      className="rating-controls"
      role="group"
      aria-label="Rate your answer confidence"
    >
      {selectedOption
        ? renderBubble(selectedOption, true)
        : RATING_OPTIONS.map((option) => renderBubble(option, false))}
    </div>
  );
}
