"use client";

import { useEffect, useMemo, useState } from "react";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";

export function MagicSparklesBackground() {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    void initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    }).then(() => setIsReady(true));
  }, []);

  const options = useMemo(
    () => ({
      fullScreen: { enable: false },
      detectRetina: true,
      fpsLimit: 60,
      background: { color: "transparent" },
      particles: {
        number: {
          value: 400,
          density: {
            enable: true,
            width: 1200,
            height: 900,
          },
        },
        color: {
          value: ["#ffffff", "#d6f3ff", "#9cdfff", "#77d3ff"],
        },
        shape: {
          type: "circle",
        },
        opacity: {
          value: { min: 0.12, max: 0.68 },
          animation: {
            enable: true,
            speed: 0.92,
            sync: false,
            startValue: "random" as const,
            minimumValue: 0.1,
          },
        },
        size: {
          value: { min: 1.1, max: 3.8 },
          animation: {
            enable: true,
            speed: 2.1,
            sync: false,
            startValue: "random" as const,
            minimumValue: 0.7,
          },
        },
        move: {
          enable: true,
          direction: "top" as const,
          speed: { min: 0.08, max: 0.34 },
          random: true,
          straight: false,
          outModes: { default: "out" as const },
        },
        twinkle: {
          particles: {
            enable: true,
            color: "#ffffff",
            frequency: 0.11,
            opacity: 1,
          },
        },
      },
      interactivity: {
        events: {
          onHover: { enable: false, mode: "repulse" },
          onClick: { enable: false, mode: "push" },
          resize: { enable: true },
        },
      },
    }),
    [],
  );

  if (!isReady) {
    return <div className="magic-sparkles-background" aria-hidden="true" />;
  }

  return (
    <div className="magic-sparkles-background" aria-hidden="true">
      <Particles
        className="magic-sparkles-background__canvas"
        options={options}
      />
    </div>
  );
}
