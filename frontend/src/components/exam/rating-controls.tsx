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
    title: "Nailed it",
    subtitle: "Brain did a victory lap",
    tone: "green",
  },
  {
    value: "almost_knew",
    title: "Nearly there",
    subtitle: "One polished reminder, please",
    tone: "yellow",
  },
  {
    value: "learned_now",
    title: "Just learned it",
    subtitle: "Fresh knowledge, handle gently",
    tone: "blue",
  },
  {
    value: "dont_understand",
    title: "Still mysterious",
    subtitle: "Send reinforcements",
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
  function renderBubble(option: RatingOption, isSelected: boolean, index: number) {
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
        animate={
          isSelected
            ? { y: -2, scale: 1.02 }
            : { y: [0, -5, 0], scale: [1, 1.025, 1] }
        }
        transition={
          isSelected
            ? { duration: 0.2, ease: "easeOut" }
            : {
                duration: 4.8,
                delay: index * 0.24,
                repeat: Number.POSITIVE_INFINITY,
                ease: "easeInOut",
              }
        }
        whileHover={isSelected ? undefined : { y: -8, scale: 1.055 }}
        whileTap={isSelected ? undefined : { y: -2, scale: 0.98 }}
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
      aria-label="Rate how confident you feel"
    >
      {selectedOption
        ? renderBubble(selectedOption, true, 0)
        : RATING_OPTIONS.map((option, index) => renderBubble(option, false, index))}
    </div>
  );
}
