"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type PropsWithChildren } from "react";
import { GuestSessionProvider } from "@/lib/session/guest-session";

export function AppProviders({ children }: PropsWithChildren) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <GuestSessionProvider>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </GuestSessionProvider>
  );
}
