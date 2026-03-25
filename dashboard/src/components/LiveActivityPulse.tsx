import React from 'react';
import { Activity, Shield, ShieldAlert, ShieldX } from 'lucide-react';
import { SecurityEvent } from '../types';
import { GlassCard } from './ui/GlassCard';
import { PulseIndicator } from './ui/PulseIndicator';

interface LiveActivityPulseProps {
  events: SecurityEvent[];
}

export function LiveActivityPulse({ events }: LiveActivityPulseProps) {
  const getVerdictIcon = (verdict: string) => {
    switch (verdict) {
      case 'ALLOW':
        return <Shield className="w-5 h-5 text-green-400" />;
      case 'MFA':
        return <ShieldAlert className="w-5 h-5 text-yellow-400" />;
      case 'DENY':
        return <ShieldX className="w-5 h-5 text-red-400" />;
      default:
        return <Shield className="w-5 h-5 text-slate-400" />;
    }
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'ALLOW':
        return 'green';
      case 'MFA':
        return 'yellow';
      case 'DENY':
        return 'red';
      default:
        return 'cyan';
    }
  };

  const getRiskScoreColor = (score: number) => {
    if (score < 0.3) return 'text-green-400';
    if (score < 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="h-full flex flex-col">
      <GlassCard className="flex-1 flex flex-col overflow-hidden" glow="magenta">
        <div className="shrink-0 mb-4">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-6 h-6 text-pink-400" />
            <h2 className="text-xl font-bold text-pink-300">Live Activity Pulse</h2>
          </div>
          <p className="text-sm text-slate-400">
            Real-time resource access logs and security events
          </p>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto space-y-3 pr-2">
          {events.length === 0 ? (
            <div className="flex items-center justify-center h-full text-slate-500">
              <div className="text-center">
                <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>Waiting for activity...</p>
              </div>
            </div>
          ) : (
            events.map((event) => (
              <div
                key={event.id}
                className="backdrop-blur-sm bg-slate-800/40 border border-slate-700/30 rounded-lg p-4 hover:bg-slate-800/60 transition-all"
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5">{getVerdictIcon(event.verdict)}</div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-slate-200">{event.user_id}</span>
                      <PulseIndicator
                        color={getVerdictColor(event.verdict) as 'cyan' | 'magenta' | 'green' | 'red' | 'yellow'}
                        size="sm"
                      />
                    </div>
                    <div className="text-sm text-slate-400 mb-2">
                      <span className="text-cyan-400">{event.action}</span>
                      <span className="mx-2">•</span>
                      <span className="text-slate-300">{event.resource}</span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-500">{event.timestamp}</span>
                      <div className="flex items-center gap-3">
                        <span className={`font-mono ${getRiskScoreColor(event.risk_score)}`}>
                          Risk: {(event.risk_score * 100).toFixed(0)}%
                        </span>
                        <span
                          className={`
                            px-2 py-1 rounded font-semibold
                            ${
                              event.verdict === 'ALLOW'
                                ? 'bg-green-500/20 text-green-300'
                                : event.verdict === 'MFA'
                                  ? 'bg-yellow-500/20 text-yellow-300'
                                  : 'bg-red-500/20 text-red-300'
                            }
                          `}
                        >
                          {event.verdict}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </GlassCard>
    </div>
  );
}