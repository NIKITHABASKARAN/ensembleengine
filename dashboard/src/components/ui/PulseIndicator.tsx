import React from 'react';

interface PulseIndicatorProps {
  color?: 'cyan' | 'magenta' | 'green' | 'red' | 'yellow';
  size?: 'sm' | 'md' | 'lg';
}

export function PulseIndicator({ color = 'cyan', size = 'md' }: PulseIndicatorProps) {
  const colors = {
    cyan: 'bg-cyan-400',
    magenta: 'bg-pink-400',
    green: 'bg-green-400',
    red: 'bg-red-400',
    yellow: 'bg-yellow-400',
  };

  const sizes = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4',
  };

  return (
    <div className="relative flex items-center justify-center">
      <div className={`${sizes[size]} ${colors[color]} rounded-full animate-pulse`}></div>
      <div
        className={`absolute ${sizes[size]} ${colors[color]} rounded-full animate-ping opacity-75`}
      ></div>
    </div>
  );
}