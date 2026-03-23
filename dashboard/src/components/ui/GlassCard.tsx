import React from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  glow?: 'cyan' | 'magenta' | 'green' | 'red' | 'yellow';
}

export function GlassCard({ children, className = '', glow }: GlassCardProps) {
  const glowColors = {
    cyan: 'shadow-[0_0_30px_rgba(6,182,212,0.3)] border-cyan-400/30',
    magenta: 'shadow-[0_0_30px_rgba(236,72,153,0.3)] border-pink-400/30',
    green: 'shadow-[0_0_30px_rgba(34,197,94,0.3)] border-green-400/30',
    red: 'shadow-[0_0_30px_rgba(239,68,68,0.3)] border-red-400/30',
    yellow: 'shadow-[0_0_30px_rgba(234,179,8,0.3)] border-yellow-400/30',
  };

  return (
    <div
      className={`
        backdrop-blur-xl bg-slate-900/60 border border-slate-700/50
        rounded-xl p-6
        ${glow ? glowColors[glow] : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
}