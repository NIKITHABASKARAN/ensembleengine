import React from 'react';

interface NeonButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'cyan' | 'magenta' | 'green' | 'red' | 'yellow';
  active?: boolean;
  className?: string;
}

export function NeonButton({
  children,
  onClick,
  variant = 'cyan',
  active = false,
  className = '',
}: NeonButtonProps) {
  const variants = {
    cyan: active
      ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_20px_rgba(6,182,212,0.5)]'
      : 'border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 hover:shadow-[0_0_15px_rgba(6,182,212,0.3)]',
    magenta: active
      ? 'bg-pink-500/20 border-pink-400 text-pink-300 shadow-[0_0_20px_rgba(236,72,153,0.5)]'
      : 'border-pink-500/50 text-pink-400 hover:bg-pink-500/10 hover:shadow-[0_0_15px_rgba(236,72,153,0.3)]',
    green: active
      ? 'bg-green-500/20 border-green-400 text-green-300 shadow-[0_0_20px_rgba(34,197,94,0.5)]'
      : 'border-green-500/50 text-green-400 hover:bg-green-500/10 hover:shadow-[0_0_15px_rgba(34,197,94,0.3)]',
    red: active
      ? 'bg-red-500/20 border-red-400 text-red-300 shadow-[0_0_20px_rgba(239,68,68,0.5)]'
      : 'border-red-500/50 text-red-400 hover:bg-red-500/10 hover:shadow-[0_0_15px_rgba(239,68,68,0.3)]',
    yellow: active
      ? 'bg-yellow-500/20 border-yellow-400 text-yellow-300 shadow-[0_0_20px_rgba(234,179,8,0.5)]'
      : 'border-yellow-500/50 text-yellow-400 hover:bg-yellow-500/10 hover:shadow-[0_0_15px_rgba(234,179,8,0.3)]',
  };

  return (
    <button
      onClick={onClick}
      className={`
        px-6 py-3 border-2 rounded-lg font-semibold
        transition-all duration-300 ease-out
        ${variants[variant]}
        ${className}
      `}
    >
      {children}
    </button>
  );
}