"use client";

import { motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";

const TICK_MS = 120;
const FINAL_TOUCHES_PROGRESS_PER_MS = 0.000025; // 0.025% per second (very slow creep)

const MAGIC_PHASES = [
  {
    key: "uploading",
    startMs: 0,
    endMs: 5_000,
    startProgress: 0,
    endProgress: 15,
    text: "Uploading files securely... 🔒",
  },
  {
    key: "analyzing",
    startMs: 5_000,
    endMs: 40_000,
    startProgress: 15,
    endProgress: 45,
    text: "AI is analyzing your documents... 🧠",
  },
  {
    key: "weaving",
    startMs: 40_000,
    endMs: 105_000,
    startProgress: 45,
    endProgress: 85,
    text: "Weaving the magic & generating questions... ✨",
  },
  {
    key: "final-touches",
    startMs: 105_000,
    endMs: Number.POSITIVE_INFINITY,
    startProgress: 85,
    endProgress: 90,
    text: "Almost there, applying final touches... 🪄",
  },
] as const;

function getPhase(elapsedMs: number) {
  return MAGIC_PHASES.find((phase) => elapsedMs >= phase.startMs && elapsedMs < phase.endMs) ?? MAGIC_PHASES[MAGIC_PHASES.length - 1];
}

function getProgressForElapsed(elapsedMs: number) {
  const phase = getPhase(elapsedMs);

  if (phase.key === "final-touches") {
    const finalElapsedMs = Math.max(0, elapsedMs - phase.startMs);
    const progressed = phase.startProgress + finalElapsedMs * FINAL_TOUCHES_PROGRESS_PER_MS;
    return Math.min(phase.endProgress, progressed);
  }

  const phaseDuration = Math.max(1, phase.endMs - phase.startMs);
  const elapsedInPhase = Math.min(phaseDuration, Math.max(0, elapsedMs - phase.startMs));
  const ratio = elapsedInPhase / phaseDuration;
  return phase.startProgress + ratio * (phase.endProgress - phase.startProgress);
}

export function MagicUploadProgress() {
  const [elapsedMs, setElapsedMs] = useState(0);
  const [progress, setProgress] = useState(0);
  const pauseUntilMsRef = useRef(0);
  const nextPauseStartMsRef = useRef(3_300);

  useEffect(() => {
    const startedAt = Date.now();
    pauseUntilMsRef.current = 0;
    nextPauseStartMsRef.current = 2_800 + Math.random() * 2_000;

    const intervalId = window.setInterval(() => {
      const elapsed = Date.now() - startedAt;
      setElapsedMs(elapsed);

      if (elapsed >= nextPauseStartMsRef.current && elapsed >= pauseUntilMsRef.current) {
        pauseUntilMsRef.current = elapsed + 350 + Math.random() * 220;
        nextPauseStartMsRef.current = elapsed + 3_200 + Math.random() * 4_100;
      }

      const isPaused = elapsed < pauseUntilMsRef.current;
      if (!isPaused) {
        setProgress(getProgressForElapsed(elapsed));
      }
    }, TICK_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  const activePhase = useMemo(() => getPhase(elapsedMs), [elapsedMs]);
  const progressText = `${Math.round(progress)}%`;

  return (
    <div className="home-upload-card__magic-progress" role="status" aria-live="polite">
      <motion.p
        key={activePhase.text}
        className="home-upload-card__magic-progress-phase"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      >
        <span>{activePhase.text}</span>
        {activePhase.key === "final-touches" ? (
          <span className="home-upload-card__magic-progress-dots" aria-hidden="true">
            ...
          </span>
        ) : null}
      </motion.p>

      <div className="home-upload-card__magic-progress-track" aria-hidden="true">
        <motion.div
          className="home-upload-card__magic-progress-fill"
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.42, ease: "easeOut" }}
        />
      </div>
      <span className="home-upload-card__magic-progress-value">{progressText}</span>
    </div>
  );
}
