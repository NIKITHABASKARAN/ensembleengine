import React, { useEffect, useState } from 'react';
import { Fingerprint, Shield, CheckCircle } from 'lucide-react';

interface BiometricScanAnimationProps {
  isActive: boolean;
  onComplete: () => void;
}

export function BiometricScanAnimation({ isActive, onComplete }: BiometricScanAnimationProps) {
  const [stage, setStage] = useState<'scanning' | 'verifying' | 'complete'>('scanning');

  useEffect(() => {
    if (!isActive) {
      setStage('scanning');
      return;
    }

    const scanTimer = setTimeout(() => setStage('verifying'), 2000);
    const verifyTimer = setTimeout(() => setStage('complete'), 4000);
    const completeTimer = setTimeout(onComplete, 5000);

    return () => {
      clearTimeout(scanTimer);
      clearTimeout(verifyTimer);
      clearTimeout(completeTimer);
    };
  }, [isActive, onComplete]);

  if (!isActive) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-xl animate-[fadeIn_0.3s_ease-out]">
      <div className="relative">
        <div className="absolute inset-0 animate-pulse">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-purple-500/20 blur-3xl"></div>
        </div>

        <div className="relative z-10 flex flex-col items-center">
          {stage === 'scanning' && (
            <>
              <div className="relative mb-8">
                <Fingerprint className="w-48 h-48 text-cyan-400 animate-pulse" />
                <div className="absolute inset-0 border-4 border-cyan-400/30 rounded-full animate-ping"></div>
              </div>
              <div className="text-center">
                <h2 className="text-3xl font-bold text-cyan-300 mb-2 animate-pulse">
                  Biometric Verification Required
                </h2>
                <p className="text-lg text-slate-400">Scanning fingerprint...</p>
              </div>
            </>
          )}

          {stage === 'verifying' && (
            <>
              <div className="relative mb-8">
                <Shield className="w-48 h-48 text-blue-400 animate-pulse" />
                <div className="absolute inset-0">
                  <div className="absolute inset-0 border-4 border-blue-400/50 rounded-full animate-spin"></div>
                </div>
              </div>
              <div className="text-center">
                <h2 className="text-3xl font-bold text-blue-300 mb-2">
                  Analyzing Identity
                </h2>
                <p className="text-lg text-slate-400">Processing biometric data...</p>
                <div className="mt-4 flex gap-2 justify-center">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.1s]"></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                </div>
              </div>
            </>
          )}

          {stage === 'complete' && (
            <>
              <div className="relative mb-8">
                <CheckCircle className="w-48 h-48 text-green-400 animate-[zoomIn_0.5s_ease-out]" />
                <div className="absolute inset-0 bg-green-400/20 rounded-full blur-2xl animate-pulse"></div>
              </div>
              <div className="text-center">
                <h2 className="text-3xl font-bold text-green-300 mb-2">
                  Verification Complete
                </h2>
                <p className="text-lg text-slate-400">Identity confirmed</p>
              </div>
            </>
          )}
        </div>

        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] pointer-events-none">
          <div className="absolute inset-0 border border-cyan-400/20 rounded-full animate-[expand_3s_ease-out_infinite]"></div>
          <div className="absolute inset-0 border border-cyan-400/20 rounded-full animate-[expand_3s_ease-out_infinite] [animation-delay:1s]"></div>
          <div className="absolute inset-0 border border-cyan-400/20 rounded-full animate-[expand_3s_ease-out_infinite] [animation-delay:2s]"></div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes expand {
          0% { transform: scale(0.8); opacity: 0.5; }
          100% { transform: scale(2); opacity: 0; }
        }
        @keyframes zoomIn {
          0% { transform: scale(0); opacity: 0; }
          50% { transform: scale(1.1); }
          100% { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}