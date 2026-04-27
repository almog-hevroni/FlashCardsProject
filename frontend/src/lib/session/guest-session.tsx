"use client";

import { createContext, useContext, useMemo, type PropsWithChildren } from "react";

type GuestSession = {
  userId: string;
  mode: "guest";
};

const DEFAULT_GUEST_SESSION: GuestSession = {
  userId: "guest",
  mode: "guest",
};

const GuestSessionContext = createContext<GuestSession>(DEFAULT_GUEST_SESSION);

export function GuestSessionProvider({ children }: PropsWithChildren) {
  const session = useMemo<GuestSession>(() => DEFAULT_GUEST_SESSION, []);
  return <GuestSessionContext.Provider value={session}>{children}</GuestSessionContext.Provider>;
}

export function useGuestSession() {
  return useContext(GuestSessionContext);
}
