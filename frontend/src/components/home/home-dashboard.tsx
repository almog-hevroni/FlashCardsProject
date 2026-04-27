"use client";

import { useRef, useState } from "react";
import { ExamHistorySidebar } from "@/components/home/exam-history-sidebar";
import { UploadExamForm } from "@/components/home/upload-exam-form";

const UPLOAD_SECTION_ID = "new-exam-upload";

export function HomeDashboard() {
  const uploadSectionRef = useRef<HTMLElement | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  function focusUploadArea() {
    uploadSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    uploadSectionRef.current?.focus();
    setIsSidebarOpen(false);
  }

  return (
    <div className="home-dashboard">
      <button
        className="home-dashboard__sidebar-toggle"
        type="button"
        onClick={() => setIsSidebarOpen((previous) => !previous)}
        aria-expanded={isSidebarOpen}
        aria-controls="home-sidebar"
      >
        {isSidebarOpen ? "Close menu" : "Menu"}
      </button>

      <ExamHistorySidebar
        className={`home-dashboard__sidebar${isSidebarOpen ? " home-dashboard__sidebar--open" : ""}`}
        onNewExam={focusUploadArea}
        onNavigate={() => setIsSidebarOpen(false)}
      />

      <main className="home-dashboard__content">
        <UploadExamForm
          id={UPLOAD_SECTION_ID}
          sectionRef={uploadSectionRef}
          onFocusWithin={() => setIsSidebarOpen(false)}
        />
      </main>
    </div>
  );
}
