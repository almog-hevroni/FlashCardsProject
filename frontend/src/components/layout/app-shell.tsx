import Link from "next/link";
import type { PropsWithChildren } from "react";

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="app-shell">
      <header className="app-shell__header">
        <Link href="/" className="app-shell__brand app-shell__brand-link cursor-pointer" aria-label="Go to home">
          FlashCards
        </Link>
      </header>
      <main className="app-shell__main">{children}</main>
    </div>
  );
}
