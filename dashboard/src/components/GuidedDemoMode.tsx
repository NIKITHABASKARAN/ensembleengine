import React, { useState } from 'react';
import { BookOpen, X, ChevronRight, ChevronLeft } from 'lucide-react';
import { NeonButton } from './ui/NeonButton';
import { GlassCard } from './ui/GlassCard';

interface GuidedDemoModeProps {
  isActive: boolean;
  onClose: () => void;
  currentHighlight: 'persona' | 'activity' | 'neural' | null;
  onHighlightChange: (highlight: 'persona' | 'activity' | 'neural' | null) => void;
}

const demoStages = [
  {
    stage: 1,
    title: 'Entry: Fast Path Processing',
    highlight: 'persona' as const,
    description: 'Start by selecting a persona to simulate different security scenarios. The system processes normal users through a fast path with minimal latency.',
    details: [
      'Normal users are authenticated within milliseconds',
      'Basic checks include session validation and IP verification',
      'Low-risk scores bypass expensive ML models',
      'Fast path handles 95% of legitimate traffic',
    ],
  },
  {
    stage: 2,
    title: 'Verification: Deep Path Analysis',
    highlight: 'neural' as const,
    description: 'When anomalies are detected, the system activates deep path analysis using a machine learning ensemble to evaluate threat levels.',
    details: [
      'Ensemble combines 5 specialized ML models (LGBM, XGB, ISO, LSTM, GNN)',
      'Trust Entropy measures model disagreement',
      'Behavioral biometrics analyzed (typing, mouse patterns)',
      'Geographic velocity calculated for travel anomalies',
    ],
  },
  {
    stage: 3,
    title: 'Enforcement: OPA Policy Engine',
    highlight: 'activity' as const,
    description: 'The system enforces decisions through OPA policies. MFA challenges are issued for medium risk, while high-risk activities are blocked entirely.',
    details: [
      'ALLOW: Grants immediate access (Risk < 30%)',
      'MFA: Requires biometric verification (Risk 30-70%)',
      'DENY: Blocks access and triggers alerts (Risk > 70%)',
      'All decisions logged in the activity pulse feed',
    ],
  },
];

export function GuidedDemoMode({ isActive, onClose, currentHighlight, onHighlightChange }: GuidedDemoModeProps) {
  const [currentStage, setCurrentStage] = useState(0);

  if (!isActive) return null;

  const stage = demoStages[currentStage];

  const handleNext = () => {
    if (currentStage < demoStages.length - 1) {
      const nextStage = currentStage + 1;
      setCurrentStage(nextStage);
      onHighlightChange(demoStages[nextStage].highlight);
    } else {
      handleClose();
    }
  };

  const handlePrev = () => {
    if (currentStage > 0) {
      const prevStage = currentStage - 1;
      setCurrentStage(prevStage);
      onHighlightChange(demoStages[prevStage].highlight);
    }
  };

  const handleClose = () => {
    setCurrentStage(0);
    onHighlightChange(null);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-[fadeIn_0.3s_ease-out]">
      <div className="max-w-2xl w-full mx-4">
        <GlassCard glow="cyan">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <BookOpen className="w-6 h-6 text-cyan-400" />
              <h2 className="text-2xl font-bold text-cyan-300">Guided Demo</h2>
            </div>
            <button
              onClick={handleClose}
              className="text-slate-400 hover:text-slate-200 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          <div className="mb-6">
            <div className="flex items-center gap-2 mb-4">
              {demoStages.map((s, idx) => (
                <div key={s.stage} className="flex items-center flex-1">
                  <div
                    className={`
                      flex-1 h-2 rounded-full transition-all
                      ${idx <= currentStage ? 'bg-cyan-500' : 'bg-slate-700'}
                    `}
                  ></div>
                  {idx < demoStages.length - 1 && (
                    <ChevronRight className="w-4 h-4 text-slate-600 mx-1" />
                  )}
                </div>
              ))}
            </div>

            <div className="text-sm text-slate-400 mb-2">
              Stage {stage.stage} of {demoStages.length}
            </div>
            <h3 className="text-2xl font-bold text-slate-200 mb-4">{stage.title}</h3>
            <p className="text-slate-300 mb-6 leading-relaxed">{stage.description}</p>

            <div className="backdrop-blur-sm bg-slate-800/40 border border-slate-700/30 rounded-lg p-6">
              <div className="text-sm font-semibold text-slate-300 mb-3">Key Points:</div>
              <ul className="space-y-2">
                {stage.details.map((detail, idx) => (
                  <li key={idx} className="flex items-start gap-3 text-sm text-slate-400">
                    <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span>{detail}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="flex items-center justify-between gap-4">
            <NeonButton
              variant="cyan"
              onClick={handlePrev}
              className={currentStage === 0 ? 'opacity-50 cursor-not-allowed' : ''}
            >
              <div className="flex items-center gap-2">
                <ChevronLeft className="w-4 h-4" />
                Previous
              </div>
            </NeonButton>

            <div className="text-xs text-slate-500">
              {currentHighlight && (
                <span className="px-3 py-1 rounded bg-cyan-500/20 text-cyan-300">
                  Highlighting: {currentHighlight}
                </span>
              )}
            </div>

            <NeonButton variant="cyan" onClick={handleNext}>
              <div className="flex items-center gap-2">
                {currentStage === demoStages.length - 1 ? 'Finish' : 'Next'}
                {currentStage < demoStages.length - 1 && <ChevronRight className="w-4 h-4" />}
              </div>
            </NeonButton>
          </div>
        </GlassCard>
      </div>
    </div>
  );
}