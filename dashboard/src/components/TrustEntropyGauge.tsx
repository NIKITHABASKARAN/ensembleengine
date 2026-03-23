import React from 'react';
import { Droplet } from 'lucide-react';
import { GlassCard } from './ui/GlassCard';

interface TrustEntropyGaugeProps {
  entropy: number;
}

export function TrustEntropyGauge({ entropy }: TrustEntropyGaugeProps) {
  const percentage = Math.min(Math.max(entropy * 100, 0), 100);

  const getEntropyColor = (value: number) => {
    if (value < 0.3) return { bg: 'bg-green-500', glow: 'shadow-[0_0_40px_rgba(34,197,94,0.6)]', text: 'text-green-300' };
    if (value < 0.6) return { bg: 'bg-yellow-500', glow: 'shadow-[0_0_40px_rgba(234,179,8,0.6)]', text: 'text-yellow-300' };
    return { bg: 'bg-red-500', glow: 'shadow-[0_0_40px_rgba(239,68,68,0.6)]', text: 'text-red-300' };
  };

  const getEntropyLabel = (value: number) => {
    if (value < 0.3) return 'Low Disagreement';
    if (value < 0.6) return 'Moderate Disagreement';
    return 'High Disagreement';
  };

  const colors = getEntropyColor(entropy);

  return (
    <GlassCard>
      <div className="flex items-center gap-3 mb-6">
        <Droplet className="w-6 h-6 text-blue-400" />
        <h3 className="text-lg font-bold text-blue-300">Trust Entropy</h3>
      </div>

      <div className="flex flex-col items-center">
        <div className="relative w-48 h-48 mb-6">
          <div className="absolute inset-0 rounded-full border-4 border-slate-700/50"></div>

          <svg className="absolute inset-0 transform -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="#1e293b"
              strokeWidth="8"
            />
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="url(#entropyGradient)"
              strokeWidth="8"
              strokeDasharray={`${percentage * 2.827} 282.7`}
              strokeLinecap="round"
              className={`transition-all duration-1000 ${colors.glow}`}
            />
            <defs>
              <linearGradient id="entropyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor={entropy < 0.3 ? '#22c55e' : entropy < 0.6 ? '#eab308' : '#ef4444'} />
                <stop offset="100%" stopColor={entropy < 0.3 ? '#16a34a' : entropy < 0.6 ? '#ca8a04' : '#dc2626'} />
              </linearGradient>
            </defs>
          </svg>

          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className={`text-4xl font-bold ${colors.text} animate-pulse`}>
              {percentage.toFixed(0)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Model Disagreement</div>
          </div>
        </div>

        <div className={`text-center px-6 py-3 rounded-lg backdrop-blur-sm ${colors.bg} bg-opacity-20 border border-current ${colors.text}`}>
          <div className="font-semibold">{getEntropyLabel(entropy)}</div>
          <div className="text-xs opacity-75 mt-1">
            {entropy < 0.3 ? 'Models are aligned' : entropy < 0.6 ? 'Some uncertainty detected' : 'Significant model variance'}
          </div>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-slate-700/50">
        <div className="text-xs text-slate-500">
          <p className="mb-2">
            <span className="font-semibold text-slate-400">Entropy Score:</span> Measures disagreement between ML models in the ensemble
          </p>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>0-30%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <span>30-60%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span>60-100%</span>
            </div>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}