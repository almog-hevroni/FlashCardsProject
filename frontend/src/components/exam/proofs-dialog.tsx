"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { useEffect, useState } from "react";
import type { Card, ProofSpan } from "@/lib/api/client";

type ProofsDialogProps = {
  isOpen: boolean;
  card: Card | null;
  userId: string;
  onClose: () => void;
};

function hasExactOffsets(proof: ProofSpan) {
  return (
    Number.isFinite(proof.start) &&
    Number.isFinite(proof.end) &&
    proof.end > proof.start
  );
}

function formatSourceLabel(docId: string): string {
  if (!docId) {
    return "Unknown";
  }
  if (/^https?:\/\//i.test(docId)) {
    try {
      const parsed = new URL(docId);
      const lastSegment = parsed.pathname.split("/").filter(Boolean).at(-1);
      return lastSegment || parsed.hostname;
    } catch {
      return docId;
    }
  }
  return docId;
}

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_API_BASE_URL;

function buildSourceUrl(proof: ProofSpan, card: Card, userId: string): string {
  const isHttpUrl = /^https?:\/\//i.test(proof.doc_id);
  const url = isHttpUrl
    ? new URL(proof.doc_id)
    : new URL(
        `/documents/${encodeURIComponent(proof.doc_id)}/source`,
        apiBaseUrl,
      );
  if (!isHttpUrl) {
    url.searchParams.set("exam_id", card.exam_id);
    url.searchParams.set("user_id", userId);
  }
  if (proof.page !== null) {
    url.hash = `page=${proof.page}`;
  }
  if (hasExactOffsets(proof)) {
    url.searchParams.set("start", String(proof.start));
    url.searchParams.set("end", String(proof.end));
  }
  return url.toString();
}

export function ProofsDialog({
  isOpen,
  card,
  userId,
  onClose,
}: ProofsDialogProps) {
  const [openProofKeys, setOpenProofKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!isOpen) {
      setOpenProofKeys(new Set());
    }
  }, [isOpen, card?.card_id]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    document.addEventListener("keydown", handleEscape);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen, onClose]);

  if (!isOpen || !card) {
    return null;
  }

  function toggleProof(key: string) {
    setOpenProofKeys((previous) => {
      const next = new Set(previous);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  return (
    <div
      className="proofs-dialog__backdrop"
      role="presentation"
      onClick={onClose}
    >
      <section
        className="proofs-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Proofs and source context"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="proofs-dialog__header">
          <div>
            <h2 className="proofs-dialog__title">Proofs</h2>
            <p className="proofs-dialog__subtitle">{card.question}</p>
          </div>
          <button
            className="proofs-dialog__close"
            type="button"
            onClick={onClose}
          >
            Close
          </button>
        </header>

        <div className="proofs-dialog__content">
          {card.proofs.length === 0 ? (
            <p className="proofs-dialog__empty">
              No proofs are available for this card yet.
            </p>
          ) : (
            <ul className="proofs-dialog__list">
              {card.proofs.map((proof, proofIndex) => {
                const jumpUrl = buildSourceUrl(proof, card, userId);
                const hasOffsets = hasExactOffsets(proof);
                const sourceLabel = formatSourceLabel(proof.doc_id);
                const pageLabel = proof.page !== null ? String(proof.page) : "Unavailable";
                const spanLabel = hasOffsets ? `${proof.start}-${proof.end}` : "Unavailable";
                const proofKey = `${proof.doc_id}:${proof.page ?? "na"}:${proof.start}:${proof.end}`;
                const panelId = `proof-panel-${proofIndex}`;
                const isOpenProof = openProofKeys.has(proofKey);
                return (
                  <li key={proofKey} className="proofs-dialog__item">
                    <button
                      type="button"
                      className="proofs-dialog__trigger"
                      aria-expanded={isOpenProof}
                      aria-controls={panelId}
                      onClick={() => toggleProof(proofKey)}
                    >
                      <span className="proofs-dialog__trigger-title">
                        Proof {proofIndex + 1} - {sourceLabel}
                      </span>
                      <ChevronDown
                        size={16}
                        className={`proofs-dialog__trigger-chevron${
                          isOpenProof ? " proofs-dialog__trigger-chevron--open" : ""
                        }`}
                        aria-hidden="true"
                      />
                    </button>
                    <AnimatePresence initial={false}>
                      {isOpenProof ? (
                        <motion.div
                          id={panelId}
                          className="proofs-dialog__panel"
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2, ease: "easeInOut" }}
                        >
                          <div className="proofs-dialog__panel-inner">
                            <blockquote className="proofs-dialog__quote">
                              {proof.text}
                            </blockquote>
                            <p className="proofs-dialog__meta-inline">
                              Source: {sourceLabel} • Page: {pageLabel} • Span: {spanLabel}
                            </p>
                            <a
                              className="proofs-dialog__jump"
                              href={jumpUrl}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Jump to source
                            </a>
                          </div>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}
