"use client";

import { useState } from "react";
import { ExamHistorySidebar } from "@/components/home/exam-history-sidebar";
import { UploadExamForm } from "@/components/home/upload-exam-form";

export function HomeDashboard() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="home-dashboard">
      <button
        className="home-dashboard__sidebar-toggle"
        type="button"
        onClick={() => setIsSidebarOpen((previous) => !previous)}
        aria-expanded={isSidebarOpen}
        aria-controls="home-sidebar"
      >
        {isSidebarOpen ? "Close deck drawer" : "Deck drawer"}
      </button>

      <ExamHistorySidebar
        className={`home-dashboard__sidebar${isSidebarOpen ? " home-dashboard__sidebar--open" : ""}`}
        onNavigate={() => setIsSidebarOpen(false)}
      />

      <main className="home-dashboard__content">
        <UploadExamForm
          onFocusWithin={() => setIsSidebarOpen(false)}
        />
      </main>
    </div>
  );
}
