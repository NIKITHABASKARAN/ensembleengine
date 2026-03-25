import React from 'react';
import { Brain, Zap, Target } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { NeuralAnalysisResponse } from '../types';
import { GlassCard } from './ui/GlassCard';

interface NeuralAnalysisProps {
  analysis: NeuralAnalysisResponse | null;
}

export function NeuralAnalysis({ analysis }: NeuralAnalysisProps) {
  const radarData = analysis
    ? [
        { model: 'LGBM', weight: analysis.ensemble_weights.lgbm * 100 },
        { model: 'XGB', weight: analysis.ensemble_weights.xgb * 100 },
        { model: 'ISO', weight: analysis.ensemble_weights.iso * 100 },
        { model: 'LSTM', weight: analysis.ensemble_weights.lstm * 100 },
        { model: 'GNN', weight: analysis.ensemble_weights.gnn * 100 },
      ]
    : [];

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'fast_path':
        return 'text-green-400';
      case 'deep_path':
        return 'text-yellow-400';
      case 'opa_enforcement':
        return 'text-red-400';
      default:
        return 'text-slate-400';
    }
  };

  const getStageLabel = (stage: string) => {
    switch (stage) {
      case 'fast_path':
        return 'Fast Path';
      case 'deep_path':
        return 'Deep Path';
      case 'opa_enforcement':
        return 'OPA Enforcement';
      default:
        return 'Unknown';
    }
  };

  return (
    <GlassCard>
      <div className="mb-4">
        <div className="flex items-center gap-3 mb-2">
          <Brain className="w-6 h-6 text-purple-400" />
          <h2 className="text-xl font-bold text-purple-300">Neural Analysis</h2>
        </div>
        <p className="text-sm text-slate-400">
          ML ensemble decision weights and processing pipeline
        </p>
      </div>

      {!analysis ? (
        <div className="flex items-center justify-center py-12 text-slate-500">
          <div className="text-center">
            <Brain className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No analysis data available</p>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="backdrop-blur-sm bg-slate-800/40 border border-slate-700/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                <span className="text-sm font-semibold text-slate-300">Processing Stage</span>
              </div>
              <span className={`text-lg font-bold ${getStageColor(analysis.processing_stage)}`}>
                {getStageLabel(analysis.processing_stage)}
              </span>
            </div>
            <div className="text-xs text-slate-500">
              {analysis.deep_path ? 'Deep analysis activated' : 'Standard processing'}
            </div>
          </div>

          <div className="backdrop-blur-sm bg-slate-800/40 border border-slate-700/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-5 h-5 text-cyan-400" />
              <span className="text-sm font-semibold text-slate-300">Ensemble Weights</span>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#475569" strokeDasharray="3 3" />
                <PolarAngleAxis
                  dataKey="model"
                  tick={{ fill: '#94a3b8', fontSize: 12 }}
                />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                />
                <Radar
                  name="Weight"
                  dataKey="weight"
                  stroke="#06b6d4"
                  fill="#06b6d4"
                  fillOpacity={0.3}
                  strokeWidth={2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="backdrop-blur-sm bg-slate-800/40 border border-slate-700/30 rounded-lg p-4">
            <div className="text-sm text-slate-300 mb-2 font-semibold">Behavioral Metrics</div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <span className="text-slate-500">Typing Cadence:</span>
                <span className="ml-2 text-cyan-400 font-mono">
                  {analysis.behavioral_analysis.typing_cadence.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Mouse Velocity:</span>
                <span className="ml-2 text-cyan-400 font-mono">
                  {analysis.behavioral_analysis.mouse_velocity.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Session Duration:</span>
                <span className="ml-2 text-cyan-400 font-mono">
                  {analysis.behavioral_analysis.session_duration.toFixed(0)}s
                </span>
              </div>
              <div>
                <span className="text-slate-500">Resource Pattern:</span>
                <span className="ml-2 text-cyan-400">
                  {analysis.behavioral_analysis.resource_pattern}
                </span>
              </div>
            </div>
          </div>

          <div className="backdrop-blur-sm bg-slate-800/40 border border-slate-700/30 rounded-lg p-4">
            <div className="text-sm text-slate-400">{analysis.explanation}</div>
          </div>
        </div>
      )}
    </GlassCard>
  );
}