import type { Metadata } from "next";
import { Wix_Madefor_Display, Wix_Madefor_Text } from "next/font/google";
import type { PropsWithChildren } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { AppProviders } from "@/components/providers/app-providers";
import "./globals.css";

const wixMadeforDisplay = Wix_Madefor_Display({
  subsets: ["latin"],
  variable: "--font-wix-display",
  display: "swap",
});

const wixMadeforText = Wix_Madefor_Text({
  subsets: ["latin"],
  variable: "--font-wix-text",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FlashCards",
  description: "Guest-first study workspace.",
};

export default function RootLayout({ children }: Readonly<PropsWithChildren>) {
  return (
    <html lang="en">
      <body className={`${wixMadeforDisplay.variable} ${wixMadeforText.variable}`}>
        <AppProviders>
          <AppShell>{children}</AppShell>
        </AppProviders>
      </body>
    </html>
  );
}
