import React from 'react';
import { Plane, MapPin, Clock, Gauge } from 'lucide-react';
import { TravelAnomaly } from '../types';
import { GlassCard } from './ui/GlassCard';

interface GeographicVelocityProps {
  travelAnomaly: TravelAnomaly | null;
}

export function GeographicVelocity({ travelAnomaly }: GeographicVelocityProps) {
  if (!travelAnomaly?.detected) {
    return null;
  }

  return (
    <div className="animate-[slideIn_0.5s_ease-out]">
      <GlassCard glow="red">
        <div className="flex items-center gap-3 mb-4">
          <div className="relative">
            <Plane className="w-6 h-6 text-red-400 animate-pulse" />
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
          </div>
          <h3 className="text-lg font-bold text-red-300">Geographic Velocity Alert</h3>
        </div>

        <div className="space-y-4">
          <div className="backdrop-blur-sm bg-red-900/20 border border-red-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <MapPin className="w-4 h-4 text-red-400" />
              <span className="text-sm font-semibold text-red-300">Impossible Travel Detected</span>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-slate-500">From:</span>
                <span className="text-slate-200 font-mono">{travelAnomaly.from_location}</span>
              </div>
              <div className="flex items-center justify-center">
                <div className="text-red-400">↓</div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-500">To:</span>
                <span className="text-slate-200 font-mono">{travelAnomaly.to_location}</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="backdrop-blur-sm bg-slate-800/40 border border-slate-700/30 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <Gauge className="w-4 h-4 text-yellow-400" />
                <span className="text-xs text-slate-400">Velocity</span>
              </div>
              <div className="text-lg font-bold text-yellow-300">
                {travelAnomaly.velocity_kph?.toLocaleString()} km/h
              </div>
            </div>

            <div className="backdrop-blur-sm bg-slate-800/40 border border-slate-700/30 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-cyan-400" />
                <span className="text-xs text-slate-400">Time Delta</span>
              </div>
              <div className="text-lg font-bold text-cyan-300">
                {travelAnomaly.time_delta_hours?.toFixed(1)}h
              </div>
            </div>
          </div>

          <div className="backdrop-blur-sm bg-slate-800/40 border border-red-500/20 rounded-lg p-3">
            <div className="text-xs text-slate-400 mb-2">
              This travel pattern is physically impossible for a human. The calculated velocity exceeds realistic transportation speeds between these locations.
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-red-400 font-semibold">High-confidence anomaly</span>
            </div>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}